"""Per-cloud feature engineering for the M2 baseline (ROADMAP M2).

Road blocks are point clouds with a variable number of points, but sklearn
classifiers expect one fixed-length row per sample. This module bridges that
gap: `extract_features` turns a single (N_points, 22) array into one feature
vector by aggregating a subset of the raw per-point columns (see
`dataset/Features.txt`) with summary statistics, plus a few explicitly
engineered geometric features.

Column choices are informed by notebooks/01_eda.ipynb:
- `global_x`, `global_y`, `global_z` are excluded on purpose. They encode
  *where* a tile sits on the map, not what it looks like. A classifier that
  latched onto absolute position could fit the training tiles perfectly
  without learning anything about lane geometry, and would fail on a road
  segment the model hasn't seen before.
- `local_y` is excluded from the per-column stats because blocks are cut to
  a fixed length along the driving direction (see CLAUDE.md) and carries
  little class signal on its own; we still use it once, to compute
  `y_range`, as a sanity check.
- `grid_index_0.3m` is a per-point bookkeeping index (which 0.3m grid cell a
  point falls into), not a physical quantity, so aggregating it directly
  (e.g. its mean) is meaningless. We don't use it here.
- `mean_intensity_0.3m_y`, `mean_intensity_0.3m_x`, and `edge_area` have a
  mean of ~0 in *every* class (see 01_eda.ipynb section 2) — a strong hint
  they are already centered/differenced signals. Their `mean` stat is kept
  below for completeness, but expect `std`/percentiles to carry the actual
  signal for these three columns.
"""

import numpy as np

from src.data import DATASET_ROOT, FEATURE_NAMES, load_points

# Raw columns aggregated with generic distribution stats: geometry
# descriptors, colour, intensity, return count, the along/across-track
# mean-intensity grids (lane-marking signal), and the two "edge" columns.
AGG_COLUMNS = [
    "local_x",  # position across the road -> lane count/width signal
    "local_z",  # residual elevation after ground removal (curbs, poles, ...)
    "red", "green", "blue",
    "intensity",
    "n_returns",
    "planarity", "linearity", "sphericity", "verticality",
    "mean_intensity_0.3m_y", "mean_intensity_1.5m_y",
    "mean_intensity_0.3m_x", "mean_intensity_1.5m_x",
    "edge_area",
    "intensity_gradient_pos",
]

# mean/std describe the bulk of the distribution; percentiles are added
# because these columns can be heavy-tailed (e.g. intensity in the EDA
# stats has max = 65535 far above the mean) and robust quantiles capture
# shape that mean/std alone would miss.
STAT_NAMES = ("mean", "std", "p5", "p25", "p50", "p75", "p95")
_PERCENTILES = (5, 25, 50, 75, 95)

EXTRA_FEATURE_NAMES = ("n_points", "x_range", "y_range", "point_density")


def engineered_feature_names() -> list[str]:
    """Column names for the vector `extract_features` returns, in order."""
    names = [f"{col}_{stat}" for col in AGG_COLUMNS for stat in STAT_NAMES]
    names.extend(EXTRA_FEATURE_NAMES)
    return names


def _column_stats(col: np.ndarray) -> np.ndarray:
    """mean, std, and the 5/25/50/75/95th percentiles of one raw column."""
    percentiles = np.percentile(col, _PERCENTILES)
    return np.array([col.mean(), col.std(), *percentiles])


def extract_features(points: np.ndarray) -> np.ndarray:
    """Aggregate one road block's raw points into a fixed-length feature vector.

    `points` is the (N_points, 22) array `src.data.load_points` returns.
    Output length and order match `engineered_feature_names()`.
    """
    stats = [
        _column_stats(points[:, FEATURE_NAMES.index(col)]) for col in AGG_COLUMNS
    ]

    x = points[:, FEATURE_NAMES.index("local_x")]
    y = points[:, FEATURE_NAMES.index("local_y")]
    # Road width: lane count is expected to correlate with how far points
    # spread across the road (x-axis), since driving direction is along y.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    n_points = points.shape[0]
    # Points per unit area: crossings/transitions may have different point
    # density than a straight, uniformly-scanned lane segment.
    point_density = n_points / (x_range * y_range + 1e-6)

    extra = np.array([n_points, x_range, y_range, point_density])
    return np.concatenate([*stats, extra])


def build_feature_matrix(
    samples: list[tuple[str, str, int]], root=DATASET_ROOT
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (X, y, paths) for a list of (rel_path, class_name, label) samples.

    `samples` is typically one split from `src.data.get_splits()`, e.g.
    `get_splits()["train"]`. Loads and discards each tile's raw points one
    at a time (not all at once) since the full dataset is ~1 GB.
    """
    n_features = len(engineered_feature_names())
    X = np.zeros((len(samples), n_features))
    y = np.zeros(len(samples), dtype=int)
    paths = []
    for i, (rel_path, _class_name, label) in enumerate(samples):
        points = load_points(rel_path, root=root)
        X[i] = extract_features(points)
        y[i] = label
        paths.append(rel_path)
    return X, y, paths
