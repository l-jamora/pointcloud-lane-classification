"""Training loop for the BEV CNN (M3).

Fits `src/cnn.py`'s LaneCNN on the train split and monitors loss on the val
split every epoch, using the frozen split from `src/data.py`. The test split
is never touched here -- it stays sealed until M5.

Three decisions that are specific to this dataset:

1. **Grids are converted once, up front, and kept in RAM.** `points_to_bev`
   re-reads a ~1 MB .npy and bins every point; doing that inside the epoch
   loop would repeat the same work 30+ times. The whole train split as grids
   is roughly 100 MB of float32, which fits comfortably.

2. **One tile per batch** (`BATCH_SIZE = 1`). Grids vary in size per tile
   (see notebooks/04_bev_cnn.ipynb) and a tensor batch must be one rectangular
   block, so batching would require zero-padding to a common H, W. That
   padding is not inert -- see `LaneCNN`'s docstring for the measurement: it
   made 136 of 213 val predictions depend on which tiles shared their batch.
   Feeding one grid at a time removes the question entirely, and costs only
   ~3.2 min instead of ~1.3 min for 40 epochs on CPU. `LaneCNN` normalises
   with `nn.GroupNorm`, which works per sample, so a batch of one is not a
   degenerate case for it the way it would be for `nn.BatchNorm2d`.

3. **The loss is class-weighted**, mirroring `class_weight="balanced"` in
   `src/baseline.py`: without it the network can score ~35% accuracy by
   always guessing 2lanes (the largest class) and the rare classes
   (transition, crossing) contribute too little gradient to be learned.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.bev import RESOLUTION_M, build_bev_grids, mirror_x
from src.cnn import DEFAULT_CHANNELS, LaneCNN
from src.data import CLASSES, get_splits
from src.metrics import evaluate

SEED = 42
EPOCHS = 40
BATCH_SIZE = 1  # see module docstring: >1 requires padding, which corrupts predictions
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0  # M4 tunes this; 0.0 makes AdamW identical to plain Adam


def class_weights(labels: np.ndarray, n_classes: int = len(CLASSES)) -> torch.Tensor:
    """Inverse-frequency weight per class: `n_samples / (n_classes * count_c)`.

    This is sklearn's "balanced" formula. A class with half the average count
    gets twice the weight, so one misclassified crossing tile costs the loss
    as much as several misclassified 2lanes tiles. Averaged over samples the
    weights come to exactly 1, so the loss keeps the scale of an unweighted
    one -- the weighting only redistributes it between classes.
    """
    # Clamp to 1 so a class absent from `labels` yields a large-but-finite
    # weight instead of inf (inf * 0 samples = nan loss). All 6 classes are
    # present in the real train split; this only guards subsets and tests.
    counts = np.maximum(np.bincount(labels, minlength=n_classes), 1)
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def _run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates.

    Training and validation differ in exactly one thing here -- whether
    gradients are computed and applied -- so they share one function instead of
    two near-identical loops that can drift apart. `model.train(is_train)` is
    still set for correctness, but with GroupNorm and no dropout the network
    computes the same thing in both modes; `tests/test_train.py` pins that.

    Returns the epoch loss plus true and predicted labels. The epoch loss is
    normalised by the *summed class weights*, not by the sample count, because
    that is what `CrossEntropyLoss(reduction="mean")` divides each batch by
    when `weight=` is set: per the PyTorch docs the denominator is
    `sum_n w_{y_n}`, not `N`. Re-weighting each batch by its sample count
    would mix two different normalisations and produce an epoch loss that no
    single batch actually measured.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_weight, y_true, y_pred = 0.0, 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()  # gradient of the loss w.r.t. every parameter
                optimizer.step()  # nudge each parameter down its gradient

            # Undo the batch's own normalisation before accumulating, so the
            # epoch loss is one weighted mean over all samples rather than a
            # mean of per-batch means with different denominators.
            batch_weight = (
                float(criterion.weight[y].sum()) if criterion.weight is not None else len(y)
            )
            total_loss += loss.item() * batch_weight
            total_weight += batch_weight
            y_true.append(y.numpy())
            y_pred.append(logits.argmax(dim=1).numpy())

    return total_loss / total_weight, np.concatenate(y_true), np.concatenate(y_pred)


def predict(model: LaneCNN, grids: list[np.ndarray]) -> np.ndarray:
    """Label predictions for a trained model on any list of BEV grids.

    Separate from `train()`'s internal val evaluation (which only ever
    scores the split `train()` itself built, for checkpoint selection) so an
    already-trained model can be scored against a *different* split -- e.g.
    the sealed test set at M5 -- without retraining. One grid at a time,
    same reasoning as `train()`'s `batch_size=1`: grids vary in size, so
    there is no batch to stack.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for grid in grids:
            logits = model(torch.from_numpy(grid).unsqueeze(0))
            preds.append(int(logits.argmax(dim=1).item()))
    return np.array(preds, dtype=np.int64)


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = SEED,
    splits: dict | None = None,
    verbose: bool = True,
    resolution: float = RESOLUTION_M,
    weight_decay: float = WEIGHT_DECAY,
    augment: bool = False,
    channels: tuple[int, ...] = DEFAULT_CHANNELS,
) -> dict:
    """Train LaneCNN on the train split, monitoring val loss each epoch.

    Returns `{"model", "history", "report"}`. `model` holds the weights from
    the epoch with the *lowest val loss*, not the last epoch: val loss usually
    bottoms out and then climbs again while train loss keeps falling, which is
    the network starting to memorise the 1140 training tiles. Keeping the best
    checkpoint is early stopping without stopping early -- we still run all
    epochs so the full curve is visible in `history`.

    The M4 tuning knobs -- `resolution`, `weight_decay`, `augment`, `channels`
    -- all default to the M3 settings, so calling `train()` with no arguments
    still reproduces the M3 result (best epoch 25, val accuracy 0.770).
    `src/tune.py` is what sweeps them.
    """
    torch.manual_seed(seed)  # reproducible weight init and batch shuffling
    if splits is None:
        splits = get_splits()

    train_grids, train_labels, _ = build_bev_grids(splits["train"], resolution=resolution)
    val_grids, val_labels, _ = build_bev_grids(splits["val"], resolution=resolution)

    if augment:
        # Mirroring across the road is label-preserving (see bev.mirror_x), so
        # each tile yields a second, different training example for free. Only
        # the train split is augmented -- val has to stay the same 213 tiles
        # every run or the metric stops being comparable across configs.
        train_grids = train_grids + [mirror_x(g) for g in train_grids]
        train_labels = np.concatenate([train_labels, train_labels])
        # Doubling every class leaves the class *ratios* untouched, so
        # class_weights below returns exactly what it would have returned on
        # the un-augmented labels.

    # No collate_fn: PyTorch's default stacks same-shaped samples and raises on
    # mismatched ones, which is the behaviour we want -- at batch_size=1 every
    # "batch" is one grid, and any larger batch fails loudly instead of being
    # silently padded (see module docstring).
    train_loader = DataLoader(
        list(zip(train_grids, train_labels)),
        batch_size=batch_size,
        shuffle=True,  # re-shuffled every epoch, so the update order varies
    )
    val_loader = DataLoader(
        list(zip(val_grids, val_labels)),
        batch_size=batch_size,
        shuffle=False,  # order is irrelevant when only measuring
    )

    model = LaneCNN(
        in_channels=train_grids[0].shape[0], n_classes=len(CLASSES), channels=channels
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels))
    # AdamW rather than Adam: it applies weight decay straight to the weights
    # instead of folding it into the gradient (and from there into Adam's
    # momentum and variance estimates, where it gets rescaled per-parameter and
    # stops acting like the L2 penalty it is meant to be). At weight_decay=0.0
    # the two are identical, so the M3 default is unaffected.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best = [], {"val_loss": float("inf")}
    for epoch in range(1, epochs + 1):
        train_loss, y_tr, p_tr = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, y_va, p_va = _run_epoch(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": float((y_tr == p_tr).mean()),
            "val_acc": float((y_va == p_va).mean()),
        }
        history.append(row)

        if val_loss < best["val_loss"]:
            best = {
                "val_loss": val_loss,
                "epoch": epoch,
                # .clone() because state_dict() shares storage with the live
                # model -- without it, later epochs would overwrite the copy.
                "state": {k: v.clone() for k, v in model.state_dict().items()},
            }

        if verbose:
            print(
                f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"train_acc={row['train_acc']:.3f}  val_acc={row['val_acc']:.3f}"
                + ("  <- best" if best["epoch"] == epoch else "")
            )

    model.load_state_dict(best["state"])
    _, y_va, p_va = _run_epoch(model, val_loader, criterion)
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "report": evaluate(y_va, p_va, CLASSES),
        # Reported because `augment` silently changes it: the notebook and
        # tests both need to see that mirroring actually doubled the split
        # rather than being accepted and ignored.
        "n_train": len(train_grids),
        "n_val": len(val_grids),
    }


if __name__ == "__main__":
    result = train()
    report = result["report"]
    w = report["weighted"]
    print(
        f"\nbest epoch: {result['best_epoch']}  "
        f"accuracy={report['accuracy']:.3f} "
        f"weighted_precision={w['precision']:.3f} "
        f"weighted_recall={w['recall']:.3f} "
        f"weighted_iou={w['iou']:.3f}"
    )
    print("\nvs M2 baseline (random_forest, val): accuracy=0.765 weighted_iou=0.629")
