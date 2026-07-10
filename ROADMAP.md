# Roadmap

Semester project — RWTH ML for Civil Engineering.
Target: classify road scenes from airborne LiDAR point clouds.

**Split:** 80% train / 15% val / 5% test

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

- [ ] Engineer per-cloud aggregate features (mean, std, percentiles of key channels)
- [ ] Train sklearn classifiers (Random Forest, SVM, etc.)
- [ ] Evaluate: accuracy, precision, recall, IoU (weighted & per-class)
- [ ] Select best baseline model

---

## M3 — BEV CNN
**Goal:** Leverage spatial structure via 2D occupancy grids.

- [ ] Implement BEV converter (xy-plane occupancy grid, configurable resolution)
- [ ] Choose/implement CNN architecture (e.g. ResNet-18 via PyTorch)
- [ ] Train with 80/15 split; monitor val loss
- [ ] Compare metrics against M2 baseline

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
