"""Self-checks for src/inference.py's save/load round trip.

Run with the project's virtualenv active: `python tests/test_inference.py`
(`.venv\\Scripts\\python` on Windows, `.venv/bin/python` on Linux/Mac).

The risk here is specific to persistence: a model can fit and predict
correctly in memory and still come back different after a save/load cycle
(wrong file, stale checkpoint, a preprocessing step that did not get saved
alongside the estimator). None of that raises an exception either -- it
just quietly hands back a worse or different model than the one that was
actually evaluated.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bev import points_to_bev  # noqa: E402
from src.data import CLASSES, get_splits, load_points  # noqa: E402
from src.baseline import get_models  # noqa: E402
from src.features import build_feature_matrix, extract_features  # noqa: E402
from src.train import predict as predict_cnn  # noqa: E402
from src.train import train  # noqa: E402
import src.inference as inference  # noqa: E402


def _tiny_splits():
    splits = get_splits()
    return {"train": splits["train"][:40], "val": splits["val"][:20], "test": []}


def test_predict_new_matches_the_in_memory_model_it_saved():
    """A save/load round trip must reproduce the exact predictions of the model that was saved.

    Fits all three models once, predicts directly in memory, then goes
    through `save_models` -> `predict_new` for the same tiles and checks
    the two sets of predictions are identical file-for-file. This is the
    one test that would catch a save path pointing at the wrong file, a
    feature-extraction mismatch between training and inference, or a CNN
    checkpoint saved with the wrong resolution.
    """
    splits = _tiny_splits()
    eval_samples = get_splits()["val"][-5:]
    eval_paths = [Path(p) for p, _c, _l in eval_samples]

    scratch = Path(tempfile.mkdtemp())
    models_dir = Path(tempfile.mkdtemp())  # never the real models/ -- see save_models' docstring
    try:
        for p in eval_paths:
            shutil.copy(p, scratch / p.name)

        X_train, y_train, _ = build_feature_matrix(splits["train"])
        models = get_models()
        models["random_forest"].fit(X_train, y_train)
        models["svm"].fit(X_train, y_train)
        cnn_result = train(resolution=inference.CNN_RESOLUTION, splits=splits, verbose=False)

        direct = {}
        for p in eval_paths:
            points = load_points(str(p))
            x = extract_features(points).reshape(1, -1)
            grid = points_to_bev(points, resolution=inference.CNN_RESOLUTION)
            direct[p.name] = {
                "random_forest": CLASSES[models["random_forest"].predict(x)[0]],
                "svm": CLASSES[models["svm"].predict(x)[0]],
                "cnn": CLASSES[predict_cnn(cnn_result["model"], [grid])[0]],
            }

        inference.save_models(splits=splits, models_dir=models_dir)

        for model_name in ("random_forest", "svm", "cnn"):
            loaded = {
                Path(k).name: v
                for k, v in inference.predict_new(model_name, scratch, models_dir=models_dir).items()
            }
            for fname in direct:
                assert loaded[fname] == direct[fname][model_name], (model_name, fname)
    finally:
        shutil.rmtree(scratch)
        shutil.rmtree(models_dir)


def test_list_unlabeled_samples_needs_no_class_folders():
    """A flat folder of tiles, with no class subfolders, must still be found.

    Held-out evaluation data may arrive exactly like this -- unlabeled, so
    no `CLASSES`-named folder to sort into -- unlike this project's own
    dataset layout, which `src.data.list_samples` assumes.
    """
    splits = get_splits()
    real_paths = [Path(p) for p, _c, _l in splits["val"][:3]]

    scratch = Path(tempfile.mkdtemp())
    try:
        for p in real_paths:
            shutil.copy(p, scratch / p.name)  # flat: no class subfolder

        found = inference.list_unlabeled_samples(scratch)
        assert {p.name for p in found} == {p.name for p in real_paths}
    finally:
        shutil.rmtree(scratch)


def test_predict_new_raises_on_empty_folder():
    """An empty or wrong data_root must fail loudly, not return an empty, silently-useless dict."""
    empty = Path(tempfile.mkdtemp())
    try:
        try:
            inference.predict_new("random_forest", empty)
        except FileNotFoundError:
            return
        raise AssertionError("expected predict_new to raise on a folder with no .npy files")
    finally:
        shutil.rmtree(empty)


if __name__ == "__main__":
    test_list_unlabeled_samples_needs_no_class_folders()
    test_predict_new_raises_on_empty_folder()
    test_predict_new_matches_the_in_memory_model_it_saved()
    print("ok")
