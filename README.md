# pointcloud-lane-classification

Classify road scenes from airborne LiDAR point clouds. RWTH semester project — ML for Civil Engineering.
Learning-first: every implementation decision should be understood by the team, not just running.

---

## Classes

| Folder | Class |
|--------|-------|
| `2lanes` | 2-lane road |
| `3lanes` | 3-lane road |
| `split4lanes` | 4-lane road with median >2m |
| `split6lanes` | 6-lane road with median >2m |
| `transition` | Lane transition |
| `crossing` | Road crossing |

## Dataset

- Format: `.npy` arrays, road blocks oriented along y-axis
- Size: ~1.05 GB across 6 class folders
- Features: 22 per point — xyz (local & global), RGB, intensity, planarity, linearity, sphericity, verticality, mean intensity grids, edge area, gradient positions
- See [`dataset/Features.txt`](dataset/Features.txt) for full feature index

## Approaches

1. **1D feature vectors** — per-cloud aggregates (mean/std/percentiles) + sklearn classifiers
2. **BEV occupancy grid** — 2D xy-plane projection → CNN (e.g. ResNet)
3. **Direct point cloud** — custom data loader + point-based model *(optional)*

## Evaluation

Train / val / test split: **80 / 15 / 5**
Metrics: accuracy, precision, recall, IoU (per-class and weighted)

## Team

Luis Jamora · Lukas Hammerschick

## Links

- [ROADMAP.md](ROADMAP.md) — milestone plan
- [CHANGELOG.md](CHANGELOG.md) — version history
- [docs/General_Suggestions.md](docs/General_Suggestions.md) — project brief
