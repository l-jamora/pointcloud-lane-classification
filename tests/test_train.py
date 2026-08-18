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

from src.cnn import LaneCNN  # noqa: E402
from src.train import _run_epoch, class_weights  # noqa: E402


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


if __name__ == "__main__":
    test_class_weights()
    test_train_and_eval_modes_agree()
    test_epoch_loss_matches_a_single_whole_split_forward_pass()
    test_mismatched_grid_sizes_fail_loudly()
    print("ok")
