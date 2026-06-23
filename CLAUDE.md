# CLAUDE.md

## Project
Classify road scenes (lane count, transitions, crossings) from airborne LiDAR point clouds. RWTH semester project.

## Team
Luis Jamora · Lukas Hammerschick

## Dataset
- Path: `dataset/`
- Format: `.npy` NumPy arrays, one file per road block sample
- Classes (6): `2lanes`, `3lanes`, `split4lanes`, `split6lanes`, `transition`, `crossing`
- Features (22): see `dataset/Features.txt` — includes xyz, RGB, intensity, geometric descriptors, mean intensity grids
- Size: ~1.05 GB total; road blocks vary in point count

## Key Files
- `docs/General_Suggestions.md` — full project brief and task instructions
- `ROADMAP.md` — milestone plan (M1–M5)
- `CHANGELOG.md` — version history
- `dataset/Features.txt` — feature index reference

## Learning Goals
This is a university learning project. When writing or explaining code, prioritize clarity and understanding over brevity. Always explain *why* a design decision was made, not just what it does. Avoid black-box solutions — if a library function is used, explain what it's doing under the hood. The team wants to understand every part of the codebase themselves.

## Dev Notes
- Python stack: numpy, scikit-learn, PyTorch (or TensorFlow for CNN work)
- Road blocks are pre-aligned: driving direction along y-axis, ground points removed
- Do not commit dataset files or large binaries; `.npy` files stay local
- Train/val/test split: 80/15/5 — freeze test set; never tune against it
