"""Baseline: interpretable per-cloud classifiers.

Trains scikit-learn classifiers on the fixed-length feature vectors from
`src/features.py` and evaluates them on the validation split. The frozen
test split from `src/data.py` stays untouched until final evaluation

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import CLASSES, get_splits
from src.features import build_feature_matrix
from src.metrics import evaluate

RANDOM_STATE = 42


def get_models() -> dict:
    """Candidate classifiers for the baseline.

    Random Forest splits on per-feature thresholds, so it is scale
    invariant and needs no preprocessing. SVC is distance-based: without
    scaling, a column like intensity (range ~19000-65535) would dominate
    the decision boundary over a column like planarity (range ~0-1)
    regardless of which one actually separates the classes, so it is
    wrapped in a StandardScaler first.

    `class_weight="balanced"` reweights both models inversely to class
    frequency during training, to offset the ~3.6:1 class imbalance
    documented in notebooks/01_eda.ipynb. M4 additionally experiments with
    balanced subsampling of the dataset itself, a different lever on the
    same problem.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(class_weight="balanced", random_state=RANDOM_STATE),
        ),
    }


def train_and_evaluate(models: dict | None = None, splits: dict | None = None) -> dict:
    """Fit each model on the train split, evaluate it on the val split.

    Returns `{model_name: {"model": fitted estimator, "report": evaluate() dict}}`.
    """
    if splits is None:
        splits = get_splits()
    if models is None:
        models = get_models()

    X_train, y_train, _ = build_feature_matrix(splits["train"])
    X_val, y_val, _ = build_feature_matrix(splits["val"])

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[name] = {"model": model, "report": evaluate(y_val, y_pred, CLASSES)}
    return results


def select_best(results: dict, metric: str = "iou") -> str:
    """Name of the model with the highest val-split weighted `metric`.

    `metric` is a key under each report's `"weighted"` dict, e.g. "iou",
    "precision", or "recall". Weighted IoU is the default because it
    penalizes both false positives and false negatives per class at once,
    the strictest of the three under class imbalance.
    """
    return max(results, key=lambda name: results[name]["report"]["weighted"][metric])


if __name__ == "__main__":
    results = train_and_evaluate()
    for name, result in results.items():
        report = result["report"]
        w = report["weighted"]
        print(
            f"{name}: accuracy={report['accuracy']:.3f} "
            f"weighted_precision={w['precision']:.3f} "
            f"weighted_recall={w['recall']:.3f} "
            f"weighted_iou={w['iou']:.3f}"
        )
    print(f"\nbest model by weighted IoU: {select_best(results)}")
