"""BEV (bird's-eye-view) grid conversion for the M3 CNN.

Turns one tile's raw points into a fixed-channel, variable-size 2D grid by
binning points into `resolution`-metre cells over the local x/y plane
(driving direction is y). Each cell gets 3 channels:

- density: point count in the cell (compressed with log1p -- cell counts are
  heavy-tailed, and log1p keeps the scale comparable to the other channels
  while staying monotonic and leaving empty cells at exactly 0)
- mean_height: mean local_z (residual height after ground removal, so
  already a small, roughly zero-centred scale -- no extra transform needed)
- mean_intensity: mean intensity divided by 65535 (the sensor's native
  16-bit range per dataset/Features.txt) -- a fixed constant, not fit on any
  split, so it cannot leak train/val/test information the way a
  per-split min/max normalisation would

Grid extent is derived per tile, not cropped or padded to a common window:
across the train split, x_range spans roughly 5-30m and y_range roughly
2-25m (see notebooks/04_bev_cnn.ipynb), so a single fixed window would
either waste most of a typical grid on empty cells or crop the road out of
the widest tiles. src/cnn.py's network accepts any grid size for exactly
this reason.
"""

import numpy as np

from src.data import DATASET_ROOT, FEATURE_NAMES, load_points

RESOLUTION_M = 0.3  # default cell size in metres; ROADMAP M4 tunes this
CHANNEL_NAMES = ("density", "mean_height", "mean_intensity")

_X_IDX = FEATURE_NAMES.index("local_x")
_Y_IDX = FEATURE_NAMES.index("local_y")
_Z_IDX = FEATURE_NAMES.index("local_z")
_INTENSITY_IDX = FEATURE_NAMES.index("intensity")
_INTENSITY_MAX = 65535.0


def points_to_bev(points: np.ndarray, resolution: float = RESOLUTION_M) -> np.ndarray:
    """Convert one tile's raw points into a `(3, H, W)` BEV grid.

    H, W come from the tile's own x/y extent at `resolution` metres per
    cell, so grid size varies tile to tile (see module docstring). Empty
    cells (no points) are 0 in every channel.
    """
    x = points[:, _X_IDX]
    y = points[:, _Y_IDX]
    z = points[:, _Z_IDX]
    intensity = points[:, _INTENSITY_IDX]

    col = np.floor((x - x.min()) / resolution).astype(np.int64)
    row = np.floor((y - y.min()) / resolution).astype(np.int64)
    n_rows, n_cols = int(row.max()) + 1, int(col.max()) + 1

    density = np.zeros((n_rows, n_cols), dtype=np.float64)
    height_sum = np.zeros((n_rows, n_cols), dtype=np.float64)
    intensity_sum = np.zeros((n_rows, n_cols), dtype=np.float64)

    np.add.at(density, (row, col), 1.0)
    np.add.at(height_sum, (row, col), z)
    np.add.at(intensity_sum, (row, col), intensity)

    occupied = density > 0
    mean_height = np.divide(height_sum, density, out=np.zeros_like(height_sum), where=occupied)
    mean_intensity = np.divide(intensity_sum, density, out=np.zeros_like(intensity_sum), where=occupied)

    grid = np.stack([np.log1p(density), mean_height, mean_intensity / _INTENSITY_MAX], axis=0)
    return grid.astype(np.float32)


def build_bev_grids(
    samples: list[tuple[str, str, int]], resolution: float = RESOLUTION_M, root=DATASET_ROOT
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Convert a list of (rel_path, class_name, label) samples to BEV grids.

    Returns a Python list of `(3, H, W)` arrays -- not a stacked ndarray,
    since grids have different H, W per tile -- a label array, and the
    source paths. `samples` is typically one split from `src.data.get_splits()`.
    """
    grids, labels, paths = [], [], []
    for rel_path, _class_name, label in samples:
        points = load_points(rel_path, root=root)
        grids.append(points_to_bev(points, resolution=resolution))
        labels.append(label)
        paths.append(rel_path)
    return grids, np.array(labels, dtype=np.int64), paths


def mirror_x(grid: np.ndarray) -> np.ndarray:
    """Flip a `(C, H, W)` BEV grid across the road, for M4 augmentation.

    `points_to_bev` puts y (the driving direction) on rows and x (across the
    road) on columns, so reversing the last axis mirrors left/right and leaves
    the direction of travel untouched. That is label-preserving for all six
    classes -- a 4-lane split seen from the opposite side is still a 4-lane
    split -- which is what makes this a valid way to double the training set
    rather than a way to inject label noise.

    `.copy()` because reversing a numpy axis produces a negative-strided view,
    and `torch.from_numpy` rejects those.
    """
    return grid[:, :, ::-1].copy()
