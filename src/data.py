"""Data loading and split for the pointcloud lane classification dataset.

Each sample is a road block: a `.npy` array of shape (N_points, 22), where N
varies per tile (see dataset/Features.txt for the column layout). This module
is the single place that knows the folder layout and the frozen split, so
every later milestone (M2 baseline, M3 CNN, ...) reads the same samples the
same way.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Folder names double as class labels; order fixes the integer label mapping
# used everywhere else (e.g. y arrays for sklearn).
CLASSES = ["2lanes", "3lanes", "split4lanes", "split6lanes", "transition", "crossing"]
class_to_idx = {name: i for i, name in enumerate(CLASSES)}
idx_to_class = {i: name for i, name in enumerate(CLASSES)}

# Verbatim from dataset/Features.txt — the single source of truth for column
# indices, so downstream code can write FEATURE_NAMES.index("intensity")
# instead of a magic "9" that silently breaks if the dataset format changes.
FEATURE_NAMES = [
    "local_x", "local_y", "local_z",
    "red", "green", "blue",
    "global_x", "global_y", "global_z",
    "intensity", "n_returns",
    "planarity", "linearity", "sphericity", "verticality",
    "mean_intensity_0.3m_y", "mean_intensity_1.5m_y",
    "mean_intensity_0.3m_x", "mean_intensity_1.5m_x",
    "edge_area", "grid_index_0.3m", "intensity_gradient_pos",
]

DATASET_ROOT = Path(__file__).resolve().parent.parent / "dataset"
SPLITS_PATH = Path(__file__).resolve().parent.parent / "splits.json"


def list_samples(root: Path = DATASET_ROOT) -> list[tuple[str, str, int]]:
    """Scan class folders for .npy files without loading them.

    Returns (relative_path, class_name, label) tuples. Only paths are listed
    here (not loaded arrays) because the full dataset is ~1 GB and most
    operations (e.g. building the split) only need to know which files
    exist, not their contents.
    """
    root = Path(root)
    samples = []
    for class_name in CLASSES:
        for path in sorted((root / class_name).glob("*.npy")):
            rel_path = path.relative_to(root.parent).as_posix()
            samples.append((rel_path, class_name, class_to_idx[class_name]))
    return samples


def load_points(rel_path: str, root: Path = DATASET_ROOT) -> np.ndarray:
    """Load one road block's points. The only function that touches disk for tile data."""
    return np.load(root.parent / rel_path)


def get_splits(
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.80, 0.15, 0.05),
    path: Path = SPLITS_PATH,
) -> dict[str, list[tuple[str, str, int]]]:
    """Return the frozen {"train", "val", "test"} sample split.

    If `path` already exists, load and return it as-is — this is the freeze
    mechanism: once splits.json is committed, re-running this function (on
    any machine, by any teammate) reproduces the exact same test set instead
    of re-rolling it. Only builds a new split when no file exists yet.

    Splitting is stratified on class label via sklearn's train_test_split,
    which wraps ShuffleSplit and preserves per-class ratios when given
    `stratify=`. That matters here: classes range from 133 to 473 samples
    (~3.6:1), so a plain random split could easily under/over-represent a
    class in a given split (`random_split` below quantifies how badly).
    Splitting happens twice -- train vs rest, then val vs test -- because
    train_test_split only produces two groups at a time.
    """
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)

    samples = list_samples()
    labels = [s[2] for s in samples]

    _, val_ratio, test_ratio = ratios
    train, rest = train_test_split(
        samples, test_size=(val_ratio + test_ratio), stratify=labels, random_state=seed
    )
    rest_labels = [s[2] for s in rest]
    val, test = train_test_split(
        rest,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=rest_labels,
        random_state=seed,
    )

    splits = {"train": train, "val": val, "test": test}
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)
    return splits


def random_split(
    samples: list[tuple[str, str, int]] | None = None,
    ratios: tuple[float, float, float] = (0.80, 0.15, 0.05),
    seed: int = 0,
) -> dict[str, list[tuple[str, str, int]]]:
    """Build one *unstratified* random train/val/test split.

    Never frozen and never used for training or evaluation -- exists only to
    illustrate, by contrast, what `get_splits()`'s stratification protects
    against: an unlucky draw can leave a rare class (`crossing`, the
    smallest at 133 tiles) with very few val/test examples, where the
    frozen stratified split guarantees a fixed, proportional count every
    time. Comparing many `random_split` seeds against the one frozen
    `get_splits()` result quantifies how often, and how badly, an unlucky
    draw actually happens on this dataset.

    Same two-step `train_test_split` mechanics as `get_splits()` (train vs
    rest, then val vs test, since the function only splits two groups at a
    time), just without `stratify=` and without persisting to disk -- every
    call with a different `seed` gives a different split.
    """
    if samples is None:
        samples = list_samples()

    _, val_ratio, test_ratio = ratios
    train, rest = train_test_split(samples, test_size=(val_ratio + test_ratio), random_state=seed)
    val, test = train_test_split(
        rest, test_size=test_ratio / (val_ratio + test_ratio), random_state=seed
    )
    return {"train": train, "val": val, "test": test}
