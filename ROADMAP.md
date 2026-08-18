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
| `src/data.py` | M1–M5 | Sample listing, `.npy` loading, class constants, frozen 80/15/5 split |
| `src/metrics.py` | M2, M3, M5 | Accuracy / precision / recall / IoU — one implementation so all models are scored identically |
| `src/features.py` | M2, M4 | Per-cloud aggregate feature vectors for the sklearn baseline |
| `src/baseline.py` | M2, M5 | Random Forest / SVM training and model selection |
| `src/bev.py` | M3–M5 | Point cloud → bird's-eye-view grid conversion |
| `src/cnn.py` | M3–M5 | `LaneCNN` architecture |
| `src/train.py` | M3–M5 | CNN training loop, class weighting, val-loss monitoring |
| `tests/test_train.py` | M3–M5 | Invariants for the training loop (loss normalisation, batch-size constraint, train/eval consistency) |

M4 and M5 add no new modules: M4 tunes the M2/M3 code above, M5 runs it once
against the sealed test set.

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

---

## M4 — Optimization
**Goal:** Squeeze performance through tuning and data strategy.

- [ ] Grid search hyperparameters (learning rate, grid resolution, model depth)
- [ ] Experiment: full dataset vs. balanced subsample
- [ ] Review & refine feature engineering based on M2/M3 errors
- [ ] Document best configs

---

## M5 — Final Evaluation & Report
**Goal:** Honest test-set assessment and clean submission.

- [ ] Run final models on held-out test set (no further tuning after this)
- [ ] Comparison table: all approaches × all metrics
- [ ] Clean up code; add docstrings to public functions
- [ ] Write final report
