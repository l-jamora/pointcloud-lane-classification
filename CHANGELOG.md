# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [Unreleased]

### Added
- `src/data.py` — dataset loader: class/label mapping, `list_samples`, `load_points`,
  and a frozen stratified 80/15/5 `get_splits` (stratified on class label because of
  the ~3.6:1 class imbalance)
- `notebooks/01_eda.ipynb` — M1 EDA: class distribution, per-feature stats per class,
  BEV + intensity visualizations for sample tiles
- `src/features.py` — M2 per-cloud feature engineering: aggregates 16 raw columns
  (mean/std/percentiles) plus explicit `n_points`/`x_range`/`y_range`/`point_density`
  into one fixed-length vector per tile; `global_x/y/z` deliberately excluded to avoid
  location leakage
- `notebooks/02_feature_engineering.ipynb` — builds the feature table on the train
  split and sanity-checks it: `x_range` isolates `2lanes` but doesn't cleanly rank the
  other classes by lane count, intensity/geometry stats carry more of that signal, and
  a correlation check flags `mean`/`p50` pairs (and `x_range`/planarity/linearity) as
  redundant
- `splits.json` — frozen train/val/test split (committed so the test set is fixed
  across the team from M1 onward)
- `requirements.txt`, `.gitignore`
- `claude/plans/` — local Claude plan documentation (gitignored)

---

## [0.1.0] - 2026-06-23

### Added
- Initial project structure
- Dataset: 1.05 GB, 6 classes, 22 features per point (`.npy` format)
- `docs/General_Suggestions.md` — project brief from course instructors
- `ROADMAP.md`, `CHANGELOG.md`, `CLAUDE.md`
