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
- `src/metrics.py` — accuracy/precision/recall/IoU (weighted & per-class), computed
  directly from a confusion matrix rather than via `sklearn.metrics`, shared across
  milestones so M3's CNN can be compared against M2 on identical metrics
- `src/baseline.py` — M2 baseline: trains Random Forest and SVM (`StandardScaler` +
  `SVC`, both `class_weight="balanced"`) on `src/features.py` vectors, evaluates on the
  val split, and selects the best model by weighted IoU
- `notebooks/03_baseline.ipynb` — trains/compares both baselines, cross-checks
  `src/metrics.py` against `sklearn.metrics` (exact match), and picks `random_forest`
  (val accuracy 0.765, weighted IoU 0.629) over `svm`; `transition`/`crossing` are the
  weakest classes for both models, consistent with `02_feature_engineering.ipynb`'s
  finding that `x_range` doesn't cleanly separate them — the concrete baseline for M3
  to beat
- `notebooks/00_project_summary.ipynb` — presentation-ready condensation of
  `01_eda.ipynb`–`03_baseline.ipynb`: at-a-glance dataset stats, class distribution and
  one example tile per class (shared physical scale), the `x_range` finding, and the
  baseline model comparison/confusion matrix/per-class IoU; ends with a key-takeaways
  and M3 next-steps summary
- `splits.json` — frozen train/val/test split (committed so the test set is fixed
  across the team from M1 onward)
- `requirements.txt`, `.gitignore`
- `.gitattributes` — forces LF line endings for all text files on every platform,
  so Windows checkouts no longer show whole-file diffs on cross-OS edits
- `claude/plans/` — local Claude plan documentation (gitignored)

---

## [0.1.0] - 2026-06-23

### Added
- Initial project structure
- Dataset: 1.05 GB, 6 classes, 22 features per point (`.npy` format)
- `docs/General_Suggestions.md` — project brief from course instructors
- `ROADMAP.md`, `CHANGELOG.md`, `CLAUDE.md`
