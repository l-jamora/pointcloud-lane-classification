# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [Unreleased]

### Added
- `src/data.py` — dataset loader: class/label mapping, `list_samples`, `load_points`,
  and a frozen stratified 80/15/5 `get_splits` (stratified on class label because of
  the ~3.6:1 class imbalance)
- `src/data.py::random_split` — an *unstratified* random split with the same 80/15/5
  mechanics as `get_splits`, never frozen and never used for training/evaluation; exists
  only to quantify, by contrast, what stratification protects against
- `notebooks/01_eda.ipynb` — M1 EDA: class distribution, per-feature stats per class,
  BEV + intensity visualizations for sample tiles; §5 compares the frozen stratified split
  against 500 random-split seeds — a random split's worst-represented test class has a
  median of 5 tiles vs. the stratified split's guaranteed 7, and 91% of random seeds do at
  least as badly, confirming stratification isn't a marginal choice on this dataset
- `notebooks/00_project_summary.ipynb` — condensed version of the same stratified-vs-random
  comparison, inserted after the dataset section
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
- `src/bev.py` — M3 BEV grid conversion: bins a tile's points into a variable-size,
  3-channel (density, mean height, mean intensity) grid at a configurable cell
  resolution (default 0.3m); grid extent is read off each tile rather than cropped
  or padded to one fixed window, since train-split extents vary widely (x_range
  ~5-30m, y_range ~2-25m)
- `src/cnn.py` — M3 `LaneCNN`: a small purpose-built CNN (~99K params, vs. ~11.2M for
  a from-scratch ResNet-18) with global adaptive pooling before the classifier, so it
  accepts any grid `H, W` from `src/bev.py` with no cropping/padding/resizing;
  `ceil_mode=True` on every pool keeps small grids (down to ~7 cells) from collapsing
  to a zero spatial dimension
- `notebooks/04_bev_cnn.ipynb` — implements and verifies the first two M3 sub-goals
  only (BEV converter + CNN architecture; training is a later step): measures tile
  extent to justify the variable-size grid design, visualizes BEV channels for an
  example tile, and stress-tests the network's forward pass on the smallest/largest
  real grids in the train split
- `src/train.py` — M3 training loop: class-weighted `CrossEntropyLoss` + Adam, one tile
  per batch (grids vary in size and padding them corrupts predictions — see below), BEV
  grids converted once up front instead of per epoch, val loss/accuracy measured every
  epoch and the lowest-val-loss checkpoint restored at the end
- `tests/test_train.py` — pins the three invariants that make the val number meaningful:
  the epoch loss equals a single whole-split forward pass, a batch of mismatched grid
  sizes raises rather than being silently padded, and `train()`/`eval()` return identical
  logits
- `notebooks/04_bev_cnn.ipynb` — extended with the remaining two M3 sub-goals: training on
  the frozen 80/15 split with train-vs-val loss curves, and the comparison against M2.
  Result: the CNN's best checkpoint (epoch 25) reaches val accuracy 0.770 / weighted IoU
  0.638 vs. the random forest's 0.765 / 0.629 — level-pegging rather than a win, since the
  accuracy gap is a single tile out of 213. Train accuracy reaches 0.94 while val plateaus
  near 0.75, so it still overfits 1140 tiles. M4 levers: augmentation, more BEV channels,
  regularisation
- `splits.json` — frozen train/val/test split (committed so the test set is fixed
  across the team from M1 onward)
- `requirements.txt`, `.gitignore`
- `.gitattributes` — forces LF line endings for all text files on every platform,
  so Windows checkouts no longer show whole-file diffs on cross-OS edits
- `claude/plans/` — local Claude plan documentation (gitignored)
- `src/tune.py` — M4 hyperparameter search: a staged coordinate search (resolution → lr →
  capacity → data strategy, each stage carrying its winner forward as the next stage's
  control) plus `repeat()`, which re-runs a config across seeds to size the noise floor.
  Runs five configurations at a time in separate processes at four threads each; the
  module docstring records the benchmark showing why this beats one 14-thread run, and
  why the machine's GPU cannot help a `batch_size=1` workload
- `results/m4_tuning.json` — every run of the M4 search, written incrementally, so the
  notebook and M5 can re-read results without re-running anything
- `src/bev.py` — `mirror_x`, label-preserving reflection across the road for augmentation
- `src/cnn.py` — `LaneCNN(channels=...)` makes depth and width searchable; the default
  `(16, 32, 64, 128)` reproduces the M3 architecture exactly
- `src/train.py` — `resolution`, `weight_decay`, `augment` and `channels` parameters, plus
  `n_train`/`n_val` in the returned dict; every default reproduces M3
- `notebooks/05_optimization.ipynb` — M4: the search, the seed-variance measurement, the
  M2 error analysis, the rejected feature refinement, and the best configurations
- `tests/test_train.py` — tests for the new knobs: `mirror_x` is an involution and flips
  the across-road axis (not the driving direction), `channels` changes capacity while
  still accepting any grid size, `augment` doubles the train split and leaves val alone,
  and AdamW at `weight_decay=0.0` is step-for-step identical to the Adam it replaced

### Changed
- `src/baseline.py` — both baselines tuned by manual, targeted search around the M4 error
  analysis, scored against val (not an automated `GridSearchCV` — no such search exists in
  the repo, and the notebook this was originally written from turned out to have never
  actually been executed). Random forest `n_estimators=200 → 500` and `max_features='sqrt' →
  0.3` (val weighted IoU 0.612 → 0.645, mean over 10 seeds); SVM `C=1 → 100` and
  `gamma='scale' → 0.01` (0.492 → 0.625, single deterministic fit — corrected from an
  unverified 0.484 once `notebooks/05_optimization.ipynb` was actually re-run and the M2-default
  SVM number was computed for the first time). `max_features` was the decisive one for the
  forest: the M4 importance analysis found no dominant feature — the top 20 of 123 carry just
  0.34 — so offering each split only ~11 candidates was discarding usable signal
- `src/train.py` — optimizer `Adam` → `AdamW`, to get a correctly decoupled weight-decay
  knob. Identical behaviour at the default `weight_decay=0.0`, pinned by a test
- `ROADMAP.md` — M3's reported CNN result flagged as superseded: 0.638 weighted IoU was a
  single lucky run, and the same config averages 0.594 ± 0.031 over 5 seeds

### Measured, and rejected
- **12 "change along the road" features** (per-third `x_range`/`intensity_mean`/`point_share`
  plus their spread), motivated by the error analysis: `transition` (IoU 0.396) and
  `crossing` (0.333) are exactly the classes whose labels depend on how the road changes
  along the tile, and no existing feature could express that. The forest ranked the new
  columns as its most important of all 135 — and val IoU *fell* 0.612 → 0.596, losing on 7
  of 10 seeds. Reverted. Feature importance measures in-sample split quality, not
  generalisation; judging by importance or by a single seed would have shipped a regression
  and reported it as a win
- **Mirror augmentation** (0.606 ± 0.034) and **weight decay 1e-4** (0.605 ± 0.025), both
  chosen to attack M3's overfitting: each narrowed the train/val gap slightly (0.093 → 0.083
  and 0.088) without improving accuracy on the 0.5 m grid (0.619 ± 0.012). The gap was a
  symptom of limited data, not of a missing regulariser
- **Balanced subsampling** (0.572) lost to the class-weighted loss already in place

### Fixed
- `src/cnn.py` — replaced `nn.BatchNorm2d` with `nn.GroupNorm` (parameter count unchanged
  at ~99K). BatchNorm is incompatible with this milestone's variable-size BEV grids:
  batching them requires zero-padding to a common H, W, and the norm layer's learned
  offset turns the padded zeros into a non-zero constant that the next conv mixes into
  real cells — 136 of 213 val tiles changed prediction between batch size 16 and 1.
  Avoiding the padding with `batch_size=1` then broke BatchNorm the other way, since it
  normalises per-tile while training but by accumulated running averages at `eval()`: the
  same 8-epoch weights scored 0.662 val accuracy in train mode and 0.127 in eval mode.
  GroupNorm normalises within each sample, so it is batch-size independent and identical
  in both modes. None of this raised an exception — it produced plausible-looking
  accuracies that were not measuring what they claimed to
- `src/train.py` — epoch loss is normalised by the summed class weights, not the sample
  count: `CrossEntropyLoss(reduction="mean")` with `weight=` divides each batch by
  `sum_n w_{y_n}` per the PyTorch docs, so scaling by batch size and dividing by N mixed
  two normalisations and produced a loss no batch had measured — the same loss that
  selected the "best" checkpoint
- `src/train.py` — `class_weights` clamps class counts to a minimum of 1, so a class
  absent from a label subset yields a finite weight instead of `inf` (and a `nan` loss)

---

## [0.1.0] - 2026-06-23

### Added
- Initial project structure
- Dataset: 1.05 GB, 6 classes, 22 features per point (`.npy` format)
- `docs/General_Suggestions.md` — project brief from course instructors
- `ROADMAP.md`, `CHANGELOG.md`, `CLAUDE.md`
