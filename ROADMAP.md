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
| `src/tune.py` | M4, M5 | Staged coordinate search + seed repeats; writes `results/m4_tuning.json` |
| `tests/test_train.py` | M3–M5 | Invariants for the training loop (loss normalisation, batch-size constraint, train/eval consistency, tuning knobs) |
| `results/m4_tuning.json` | M4, M5 | Every run of the M4 search, so results are re-readable without re-running |

M5 adds no new modules: it runs the M4 configurations once against the sealed
test set.

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
| svm | `C=1`, `gamma='scale'` | **`C=100`, `gamma=0.01`** (IoU 0.484 → 0.625) |

Negative results worth keeping: mirror augmentation and weight decay both narrowed the
train/val gap without improving accuracy, and 12 "change along the road" features that
the forest ranked as its *most important* made validation IoU worse — high importance
measures in-sample split quality, not generalisation.

---

## M5 — Final Evaluation & Report
**Goal:** Honest test-set assessment and clean submission.

- [ ] Run final models on held-out test set (no further tuning after this) — report the
      CNN over several seeds, not as a single run (see M4)
- [ ] Comparison table: all approaches × all metrics, with spreads
- [ ] Clean up code; add docstrings to public functions
- [ ] Write final report
