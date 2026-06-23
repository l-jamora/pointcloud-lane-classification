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

## Dev Guidelines

### Commit Messages
Follow `type(scope): description` convention. Keep the description short and imperative.

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `refactor` | Code restructure, no behaviour change |
| `docs` | Documentation only |
| `data` | Dataset changes |
| `chore` | Tooling, config, cleanup |

Examples:
```
feat(bev_converter.py): add configurable grid resolution
fix(dataloader.py): handle variable point counts per sample
refactor(baseline.py): extract feature extraction to separate function
docs(CHANGELOG.md): add M2 baseline release entry
```

### Documentation
Update `CHANGELOG.md` and relevant docs with every commit. Documentation changes should be included in the same commit as the code change — not as a follow-up.

### Issues
Each issue must include:

```markdown
## Context
<what, why, any relevant screenshots or data>

## Acceptance Criteria
- [ ] criterion 1
- [ ] criterion 2
```

- Wording: short and specific — enough for the other team member to pick it up cold
- Assign to the relevant milestone (M1–M5) and a label (`bug`, `feat`, `data`, etc.)

## Dev Notes
- Python stack: numpy, scikit-learn, PyTorch (or TensorFlow for CNN work)
- Road blocks are pre-aligned: driving direction along y-axis, ground points removed
- Do not commit dataset files or large binaries; `.npy` files stay local
- Train/val/test split: 80/15/5 — freeze test set; never tune against it
