"""Self-checks for the parts of src/train.py that are easy to get silently wrong.

Run with: .venv-linux/bin/python tests/test_train.py

Silently wrong is the risk being covered here: neither of these defects raises
an exception. They just produce a number that looks plausible and is not the
number you think you are reading.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bev import mirror_x  # noqa: E402
from src.cnn import LaneCNN  # noqa: E402
from src.data import get_splits  # noqa: E402
from src.train import _run_epoch, class_weights, train  # noqa: E402


def test_class_weights():
    labels = np.array([0, 0, 0, 1])  # class 0 three times as common as class 1
    w = class_weights(labels, n_classes=2)

    assert np.isclose(w[1] / w[0], 3.0)  # rarer class weighted proportionally higher
    # Sample-averaged weight is 1: the loss keeps its unweighted scale overall,
    # weighting only shifts it between classes.
    counts = np.bincount(labels)
    assert np.isclose((counts * w.numpy()).sum() / len(labels), 1.0)


def test_epoch_loss_matches_a_single_whole_split_forward_pass():
    """The epoch loss must equal the loss of one pass over all samples at once.

    `CrossEntropyLoss(reduction="mean")` with `weight=` divides each batch by
    the *summed class weights* in that batch, not by its sample count (PyTorch
    docs: the denominator is `sum_n w_{y_n}`). Accumulating `loss * len(y)` and
    dividing by N therefore mixes two normalisations and yields a loss no batch
    ever measured. This catches that.
    """
    torch.manual_seed(0)
    grids = [np.random.default_rng(i).random((3, 24, 24)).astype(np.float32) for i in range(8)]
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2])

    model = LaneCNN(in_channels=3, n_classes=6)
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels, n_classes=6))

    loader = DataLoader(list(zip(grids, labels)), batch_size=3, shuffle=False)
    epoch_loss, _, _ = _run_epoch(model, loader, criterion)

    with torch.no_grad():  # same data, one forward pass, no batching at all
        x = torch.from_numpy(np.stack(grids))
        reference = criterion(model(x), torch.from_numpy(labels)).item()

    assert np.isclose(epoch_loss, reference, atol=1e-6), (epoch_loss, reference)


def test_mismatched_grid_sizes_fail_loudly():
    """A batch of differently-sized grids must raise, never be silently padded.

    Zero-padding a batch to a common H, W looks harmless (an empty BEV cell is
    already 0) but is not: BatchNorm shifts the padded zeros to a non-zero
    activation, the next conv mixes that into real cells, and the final global
    average divides by a cell count including the padding. Measured on a
    6-epoch model, 136 of 213 val tiles changed prediction between batch size
    16 and 1. src/train.py uses batch_size=1 for this reason; this test pins
    the guarantee that a larger batch cannot quietly reintroduce the problem.
    """
    grids = [np.zeros((3, 9, 11), dtype=np.float32), np.zeros((3, 40, 33), dtype=np.float32)]
    loader = DataLoader(list(zip(grids, np.array([0, 1]))), batch_size=2)

    try:
        next(iter(loader))
    except RuntimeError:
        return
    raise AssertionError("expected mismatched grid sizes to raise, got a padded batch")


def test_train_and_eval_modes_agree():
    """`model.train()` and `model.eval()` must give the same logits.

    This is the guarantee that lets a val number be trusted: the weights being
    measured must behave the same way they behaved while being fitted. It only
    holds because `src/cnn.py` normalises per sample (GroupNorm). With
    `nn.BatchNorm2d` and `batch_size=1` it fails badly -- measured on an
    8-epoch model, the identical weights scored 0.662 val accuracy in train
    mode and 0.127 in eval mode, because BatchNorm's running averages, each
    collected from a single tile, did not match the per-tile statistics
    training had used.
    """
    grid = torch.from_numpy(
        np.random.default_rng(0).random((1, 3, 33, 70)).astype(np.float32)
    )
    model = LaneCNN(in_channels=3, n_classes=6)

    with torch.no_grad():
        model.train()
        train_mode = model(grid)
        model.eval()
        eval_mode = model(grid)

    assert torch.allclose(train_mode, eval_mode, atol=1e-6), (train_mode - eval_mode).abs().max()


# --- M4 tuning knobs -------------------------------------------------------


def test_mirror_x_is_its_own_inverse():
    """Mirroring twice returns the original grid, and shape never changes.

    The augmentation's whole claim is that a mirrored tile is the *same scene*
    viewed from the other side -- a real second example, not a corrupted one.
    An involution is the cheapest way to pin that the operation is a clean
    reflection and not, say, a roll or a transpose that would deform geometry
    and turn free data into label noise.
    """
    grid = np.random.default_rng(0).random((3, 12, 25)).astype(np.float32)
    once = mirror_x(grid)

    assert once.shape == grid.shape
    assert np.array_equal(mirror_x(once), grid)
    assert not np.array_equal(once, grid)  # 12x25 random grid is not symmetric
    # Reversing a numpy axis yields a negative-strided view that
    # torch.from_numpy rejects; mirror_x copies, so this must not raise.
    torch.from_numpy(once)


def test_mirror_flips_across_the_road_not_along_it():
    """The reflection must be left/right, leaving the driving direction alone.

    src/bev.py puts y (driving direction) on rows and x (across the road) on
    columns. Mirroring the wrong axis would reverse the direction of travel,
    which is *not* obviously label-preserving for transitions -- so this pins
    which axis moves rather than trusting the docstring.
    """
    grid = np.zeros((1, 4, 3), dtype=np.float32)
    grid[0, 0, 0] = 1.0  # first row, first column

    flipped = mirror_x(grid)

    assert flipped[0, 0, 2] == 1.0  # moved across the columns (x)
    assert flipped[0, 0, 0] == 0.0
    assert flipped[0, :, :].sum(axis=1).tolist() == [1.0, 0.0, 0.0, 0.0]  # row order intact


def test_channels_parameter_changes_capacity_and_still_forwards():
    """`channels` must alter the network and keep accepting any grid size.

    A shape-agnostic architecture is easy to break when it becomes
    configurable -- an off-by-one in the width list would wire the classifier
    to the wrong layer. The 9x11 grid also exercises the `ceil_mode` pooling
    path that keeps small tiles from collapsing to zero size.
    """
    grid = torch.from_numpy(np.random.default_rng(0).random((1, 3, 9, 11)).astype(np.float32))

    sizes = {}
    for channels in ((16, 32, 64), (16, 32, 64, 128), (32, 64, 128, 256)):
        model = LaneCNN(in_channels=3, n_classes=6, channels=channels)
        assert model(grid).shape == (1, 6)
        sizes[channels] = sum(p.numel() for p in model.parameters())

    assert sizes[(16, 32, 64)] < sizes[(16, 32, 64, 128)] < sizes[(32, 64, 128, 256)]


def test_augment_doubles_the_train_split_and_leaves_val_alone():
    """`augment=True` must add mirrored *training* tiles and touch nothing else.

    Two ways this silently goes wrong: the flag is accepted but ignored, so a
    whole stage of the M4 search measures nothing; or val gets augmented too,
    so val stops being the same fixed set of tiles and its score is no longer
    comparable across configs. Neither shows up in the printed loss.

    Uses six real tiles and one epoch -- enough to prove the wiring, cheap
    enough to stay in a test.
    """
    splits = get_splits()
    tiny = {"train": splits["train"][:4], "val": splits["val"][:2], "test": []}

    plain = train(epochs=1, splits=tiny, verbose=False)
    augmented = train(epochs=1, splits=tiny, verbose=False, augment=True)

    assert plain["n_train"] == 4
    assert augmented["n_train"] == 8  # mirrored copies added
    assert plain["n_val"] == augmented["n_val"] == 2  # val untouched


def test_weight_decay_default_leaves_m3_behaviour_unchanged():
    """AdamW at weight_decay=0.0 must match the Adam it replaced, step for step.

    src/train.py switched optimizer to get a decay knob. If that switch also
    moved the default trajectory, every M3 number in the docs would silently
    stop being reproducible -- so this pins that the default is a no-op change.
    """
    torch.manual_seed(0)
    model_a = LaneCNN(in_channels=3, n_classes=6)
    torch.manual_seed(0)
    model_b = LaneCNN(in_channels=3, n_classes=6)

    grid = torch.from_numpy(np.random.default_rng(0).random((1, 3, 20, 20)).astype(np.float32))
    y = torch.tensor([2])
    criterion = nn.CrossEntropyLoss()

    for model, optimizer in (
        (model_a, torch.optim.Adam(model_a.parameters(), lr=1e-3)),
        (model_b, torch.optim.AdamW(model_b.parameters(), lr=1e-3, weight_decay=0.0)),
    ):
        for _ in range(3):
            loss = criterion(model(grid), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for a, b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


if __name__ == "__main__":
    test_class_weights()
    test_train_and_eval_modes_agree()
    test_epoch_loss_matches_a_single_whole_split_forward_pass()
    test_mismatched_grid_sizes_fail_loudly()
    test_mirror_x_is_its_own_inverse()
    test_mirror_flips_across_the_road_not_along_it()
    test_channels_parameter_changes_capacity_and_still_forwards()
    test_augment_doubles_the_train_split_and_leaves_val_alone()
    test_weight_decay_default_leaves_m3_behaviour_unchanged()
    print("ok")
