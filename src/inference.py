"""Model persistence and inference on new, unlabeled data.

Training and evaluation (`src/baseline.py`, `src/train.py`, `src/final_eval.py`)
never save a model to disk -- every number in this project comes from a
freshly-fit model, scored once and discarded. That is fine for measuring
performance, but a model that will be evaluated by someone else, on data
this project never sees, has to still exist after the process that trained
it exits. `save_models` fits and persists one copy of each final model;
`predict_new` loads a saved model and scores an arbitrary folder of tiles
that may have no ground-truth labels or class-folder structure at all.
"""

from pathlib import Path

import joblib
import numpy as np
import torch

from src.baseline import get_models
from src.bev import points_to_bev
from src.cnn import LaneCNN
from src.data import CLASSES, get_splits
from src.features import build_feature_matrix, extract_features
from src.final_eval import CNN_RESOLUTION
from src.train import predict as predict_cnn
from src.train import train

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _paths(models_dir: Path) -> dict[str, Path]:
    return {
        "random_forest": models_dir / "random_forest.joblib",
        "svm": models_dir / "svm.joblib",
        "cnn": models_dir / "cnn.pt",
    }


def save_models(splits: dict | None = None, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    """Fit and persist one copy of each final model, trained the same way M5 scored them.

    Random forest and SVM: `get_models()`'s default seed (`RANDOM_STATE`),
    fit on train -- one of the ten random-forest seeds `final_eval.py`
    already scored on test, and exactly the SVM fit it scored (the SVM's
    decision boundary does not depend on seed). The CNN: `train()`'s own
    default seed, at the M4-winning 0.5m resolution.

    Deliberately not a retrain on train+val+test: the point of saving a
    model is to hand over the *same object* M5's reported numbers describe,
    not a different one that merely shares its hyperparameters. Val is
    still used inside `train()` to pick the CNN's best checkpoint, same as
    every other CNN run in this project.

    `models_dir` defaults to the committed `models/` folder; tests pass a
    scratch directory instead so a test run can never overwrite the real,
    already-evaluated artifacts.
    """
    if splits is None:
        splits = get_splits()
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)
    paths = _paths(models_dir)

    X_train, y_train, _ = build_feature_matrix(splits["train"])

    models = get_models()
    models["random_forest"].fit(X_train, y_train)
    # compress=3: cuts the forest's ~500 trees from ~18 MB to ~4 MB on disk for a
    # negligible, one-time load-time cost -- worth it for something meant to be committed.
    joblib.dump(models["random_forest"], paths["random_forest"], compress=3)

    models["svm"].fit(X_train, y_train)
    joblib.dump(models["svm"], paths["svm"], compress=3)

    cnn_result = train(resolution=CNN_RESOLUTION, splits=splits, verbose=False)
    torch.save(
        {"state_dict": cnn_result["model"].state_dict(), "resolution": CNN_RESOLUTION},
        paths["cnn"],
    )

    return paths


def list_unlabeled_samples(root) -> list[Path]:
    """Every `.npy` file under `root`, found recursively.

    Unlike `src.data.list_samples`, which expects the six `CLASSES`
    subfolders this project's own dataset uses, held-out evaluation data may
    not be labelled at all -- that is the point of it -- so nothing here
    reads or requires a class name. A flat folder of tiles and a
    class-labelled one both work the same way.
    """
    root = Path(root)
    return sorted(root.rglob("*.npy"))


def predict_new(model_name: str, data_root, models_dir: Path = MODELS_DIR) -> dict[str, str]:
    """Load a saved model and predict a class for every tile under `data_root`.

    `model_name` is one of `"random_forest"`, `"svm"`, `"cnn"`.  `data_root`
    needs no class-folder structure or ground-truth labels -- see
    `list_unlabeled_samples`. `models_dir` must match whatever `save_models`
    was given. Returns `{file_path: predicted_class_name}`.
    """
    tile_paths = list_unlabeled_samples(data_root)
    if not tile_paths:
        raise FileNotFoundError(f"no .npy files found under {data_root}")
    model_paths = _paths(Path(models_dir))

    if model_name in ("random_forest", "svm"):
        model = joblib.load(model_paths[model_name])
        X = np.stack([extract_features(np.load(p)) for p in tile_paths])
        y_pred = model.predict(X)
    elif model_name == "cnn":
        checkpoint = torch.load(model_paths["cnn"], weights_only=True)
        model = LaneCNN()
        model.load_state_dict(checkpoint["state_dict"])
        grids = [points_to_bev(np.load(p), resolution=checkpoint["resolution"]) for p in tile_paths]
        y_pred = predict_cnn(model, grids)
    else:
        raise ValueError(f"unknown model_name: {model_name!r}, expected random_forest/svm/cnn")

    return {str(path): CLASSES[label] for path, label in zip(tile_paths, y_pred)}


if __name__ == "__main__":
    paths = save_models()
    for name, path in paths.items():
        print(f"{name}: {path} ({path.stat().st_size / 1024:.0f} KB)")
