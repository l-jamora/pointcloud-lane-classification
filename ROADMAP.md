# Roadmap

Semester project — RWTH ML for Civil Engineering.
Target: classify road scenes from airborne LiDAR point clouds.

**Split:** 80% train / 15% val / 5% test

### Where the code lives

`src/` is flat on purpose — modules are grouped by responsibility, not by
milestone, because most of them are used across several. This table is the
milestone mapping.

| File | Milestones | Role |
|------|-----------|------|
| `src/data.py` | M1–M5 | Sample listing, `.npy` loading, class constants, frozen 80/15/5 split; `random_split` (M1 only, illustrative) |
| `src/metrics.py` | M2–M5 | Accuracy / precision / recall / IoU — one implementation so all models are scored identically |
| `src/features.py` | M2, M4 | Per-cloud aggregate feature vectors for the sklearn baseline |
| `src/baseline.py` | M2, M4, M5 | Random Forest / SVM training and model selection |
| `src/bev.py` | M3–M5 | Point cloud → bird's-eye-view grid conversion |
| `src/cnn.py` | M3–M5 | `LaneCNN` architecture |
| `src/train.py` | M3–M5 | CNN training loop, class weighting, val-loss monitoring, `predict` for scoring a trained model on any split |
| `src/tune.py` | M4 | Staged coordinate search + seed repeats; writes `results/m4_tuning.json` |
| `src/final_eval.py` | M5 | Runs the M4-winning configs once against the sealed test split; writes `results/m5_test_eval.json` |
| `tests/test_train.py` | M3–M5 | Invariants for the training loop (loss normalisation, batch-size constraint, train/eval consistency, tuning knobs, `predict` vs. `train()`'s own val report) |
| `results/m4_tuning.json` | M4, M5 | Every run of the M4 search, so results are re-readable without re-running |
| `results/m5_test_eval.json` | M5 | The one-time sealed-test evaluation, so the comparison table is re-readable without re-running |

---

## M1 — Setup & EDA
**Goal:** Understand the data before modelling.

- [x] Write data loader for `.npy` files
- [x] Compute per-feature statistics (mean, std, range) per class
- [x] Visualize sample point clouds (BEV + intensity overlay)
- [x] Document class distribution and imbalance

---

## M2 — Baseline (1D Feature Extraction)
**Goal:** Fast, interpretable first result using per-cloud feature vectors.

- [x] Engineer per-cloud aggregate features (mean, std, percentiles of key channels)
- [x] Train sklearn classifiers (Random Forest, SVM, etc.)
- [x] Evaluate: accuracy, precision, recall, IoU (weighted & per-class)
- [x] Select best baseline model

---

## M3 — BEV CNN
**Goal:** Leverage spatial structure via 2D occupancy grids.

- [x] Implement BEV converter (xy-plane occupancy grid, configurable resolution)
- [x] Choose/implement CNN architecture (e.g. ResNet-18 via PyTorch)
- [x] Train with 80/15 split; monitor val loss
- [x] Compare metrics against M2 baseline — CNN val accuracy 0.770 / weighted IoU 0.638
      vs. random forest 0.765 / 0.629. Level-pegging, not a win: the accuracy gap is
      one tile out of 213. Still overfits (train acc 0.94); M4 levers below
- [!] **Superseded by M4.** Both numbers above are single runs. Over 5 seeds the same
      CNN config averages 0.594 ± 0.031 weighted IoU, so 0.638 was a lucky draw — see
      the M4 section for the corrected comparison.

---

## M4 — Optimization
**Goal:** Squeeze performance through tuning and data strategy.

- [x] Search hyperparameters (learning rate, grid resolution, model depth) — staged
      coordinate search in `src/tune.py`, not a full grid: with 213 val tiles a
      27-run cross product resolves differences smaller than the noise floor
- [x] Experiment: full dataset vs. balanced subsample — full wins; balanced
      subsampling (0.572) lost to the class-weighted loss already in place
- [x] Review & refine feature engineering based on M2/M3 errors — errors localised to
      `transition` (IoU 0.396) and `crossing` (0.333); the refinement this suggested was
      implemented, measured, and **rejected** for making val IoU worse (see below)
- [x] Document best configs — `notebooks/05_optimization.ipynb` §5

### The M4 result that matters

M4's first finding was that M3's numbers could not be trusted, because **nothing in
the project had ever measured run-to-run variance**. Holding the config and seed
fixed and changing only the thread count moved weighted IoU from 0.638 to 0.587
(parallel float reductions are not associative, and 40 epochs amplify the last bits).

Re-run over 5 seeds:

| config | val weighted IoU |
|---|---|
| tuned random forest (M4) | **0.645 ± 0.011** (10 seeds) |
| CNN, 0.5 m grid (M4 best) | 0.619 ± 0.012 |
| tuned SVM (M4) | 0.625 |
| CNN, M3 config | 0.594 ± 0.031 |

So M3's reported 0.638 was a lucky single run, and **the tuned baseline is now ahead
of the CNN**, not behind it. Rule adopted for M5: a single-run difference below
~0.06 weighted IoU is not a finding.

### Best configs

| | M2/M3 | M4 |
|---|---|---|
| BEV resolution | 0.3 m | **0.5 m** (only CNN change that survived seed testing) |
| CNN lr / channels | 1e-3 / (16,32,64,128) | unchanged — alternatives were worse |
| CNN data strategy | full + weighted loss | unchanged — augmentation, weight decay and balanced subsampling all failed |
| random forest | `n_estimators=200`, `max_features='sqrt'` | **`n_estimators=500`, `max_features=0.3`** |
| svm | `C=1`, `gamma='scale'` | **`C=100`, `gamma=0.01`** (IoU 0.492 → 0.625, single fit) |

Negative results worth keeping: mirror augmentation and weight decay both narrowed the
train/val gap without improving accuracy, and 12 "change along the road" features that
the forest ranked as its *most important* made validation IoU worse — high importance
measures in-sample split quality, not generalisation.

---

## M5 — Final Evaluation & Report
**Goal:** Honest test-set assessment and clean submission.

- [x] Run final models on held-out test set (no further tuning after this) — random forest
      refit over 10 seeds, SVM once (deterministic), CNN retrained over 5 seeds; see
      `src/final_eval.py` / `results/m5_test_eval.json`
- [x] Comparison table: all approaches × all metrics, with spreads — `notebooks/06_final_evaluation.ipynb`
- [ ] Clean up code; add docstrings to public functions
- [ ] Write final report

### Test-set result

| model | val weighted IoU | test weighted IoU |
|---|---|---|
| **random forest** | 0.645 ± 0.011 (10 seeds) | **0.656 ± 0.017** (10 seeds) |
| svm | 0.625 (1 fit) | 0.523 (1 fit) |
| cnn (0.5 m grid) | 0.619 ± 0.012 (5 seeds) | 0.583 ± 0.023 (5 seeds) |

**The tuned random forest is the project's final model** — 0.073 ahead of the CNN on test,
clearing M4's own "0.06 is not a finding" bar. It did not drop from val to test at all
(within its own seed spread); the CNN dropped as M4 predicted (checkpoint-selection
optimism); the SVM dropped hardest (0.625 → 0.523) — `C=100` won on val by trading away
regularisation, a val-overfit only a sealed, untouched split could expose. `split4lanes`
(8 test tiles) was unexpectedly the weakest class on test for every model — read as sampling
noise from a small per-class test count, not a new failure mode, per `01_eda.ipynb` §5.
