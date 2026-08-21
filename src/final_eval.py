"""M5 final test-set evaluation.

Runs the three M4-winning configurations against the sealed test split, for
the first and only time in the project. Every hyperparameter below was
already fixed by M4 (`ROADMAP.md` "Best configs") using only train/val --
nothing here is chosen or adjusted based on what the test split returns.
`ROADMAP.md`/`CLAUDE.md`: freeze the test set, never tune against it.

Random forest and CNN are each refit/retrained over several seeds and
scored individually, because both have a randomised training procedure (the
forest's bootstrap/feature sampling; the CNN's weight init and shuffling,
plus the thread-count sensitivity M4 measured) -- a single seed is not a
reliable number for either, the same reason M4 stopped trusting single runs.
The SVM is fit once: its decision boundary is deterministic given fixed
data and hyperparameters, so repeating it would only measure floating-point
noise, not signal.
"""

import json
from pathlib import Path

import numpy as np

from src.baseline import get_models
from src.bev import build_bev_grids
from src.data import CLASSES, get_splits
from src.features import build_feature_matrix
from src.metrics import evaluate
from src.train import predict, train

RF_SEEDS = tuple(range(10))  # matches notebooks/05_optimization.ipynb's iou_over_seeds
CNN_SEEDS = (0, 1, 2, 3, 4)  # matches src/tune.py's repeat() default
CNN_RESOLUTION = 0.5  # the only CNN change that survived M4 seed testing

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "m5_test_eval.json"


def evaluate_random_forest(splits: dict) -> list[dict]:
    """Fit the M4-tuned forest (`n_estimators=500, max_features=0.3`) over `RF_SEEDS`, score each on test."""
    X_train, y_train, _ = build_feature_matrix(splits["train"])
    X_test, y_test, _ = build_feature_matrix(splits["test"])

    reports = []
    for seed in RF_SEEDS:
        model = get_models(seed=seed)["random_forest"]
        model.fit(X_train, y_train)
        reports.append(evaluate(y_test, model.predict(X_test), CLASSES))
    return reports


def evaluate_svm(splits: dict) -> dict:
    """Fit the M4-tuned SVM (`C=100, gamma=0.01`) once, score it on test."""
    X_train, y_train, _ = build_feature_matrix(splits["train"])
    X_test, y_test, _ = build_feature_matrix(splits["test"])

    model = get_models()["svm"]
    model.fit(X_train, y_train)
    return evaluate(y_test, model.predict(X_test), CLASSES)


def evaluate_cnn(splits: dict) -> list[dict]:
    """Train the M4-best CNN (0.5m grid) over `CNN_SEEDS`, score each trained model on test.

    Each seed still trains with the ordinary train/val split internally
    (val picks the best checkpoint, same as every other CNN run in this
    project) -- only the final scoring at the end targets test instead of
    val, via `predict()` rather than `train()`'s own (val-only) report.
    """
    test_grids, y_test, _ = build_bev_grids(splits["test"], resolution=CNN_RESOLUTION)

    reports = []
    for seed in CNN_SEEDS:
        result = train(resolution=CNN_RESOLUTION, seed=seed, splits=splits, verbose=False)
        y_pred = predict(result["model"], test_grids)
        reports.append(evaluate(y_test, y_pred, CLASSES))
    return reports


def _jsonable(report: dict) -> dict:
    """`evaluate()`'s confusion_matrix is an ndarray; everything else is already plain Python."""
    return {**report, "confusion_matrix": report["confusion_matrix"].tolist()}


def main(splits: dict | None = None) -> dict:
    """Run all three evaluations once, write them to `RESULTS_PATH`, and return them.

    Written to disk (like `results/m4_tuning.json`) so `notebooks/06_final_evaluation.ipynb`
    reads this instead of re-running it -- the test split should be touched
    by code exactly once, not once per notebook execution.
    """
    if splits is None:
        splits = get_splits()

    results = {
        "random_forest": {"seeds": list(RF_SEEDS), "reports": [_jsonable(r) for r in evaluate_random_forest(splits)]},
        "svm": {"report": _jsonable(evaluate_svm(splits))},
        "cnn": {
            "seeds": list(CNN_SEEDS),
            "resolution": CNN_RESOLUTION,
            "reports": [_jsonable(r) for r in evaluate_cnn(splits)],
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    results = main()
    rf_iou = np.array([r["weighted"]["iou"] for r in results["random_forest"]["reports"]])
    cnn_iou = np.array([r["weighted"]["iou"] for r in results["cnn"]["reports"]])
    svm_iou = results["svm"]["report"]["weighted"]["iou"]

    print(f"random forest (test, {len(RF_SEEDS)} seeds): weighted IoU {rf_iou.mean():.3f} +/- {rf_iou.std(ddof=1):.3f}")
    print(f"svm (test, single fit): weighted IoU {svm_iou:.3f}")
    print(f"cnn (test, {len(CNN_SEEDS)} seeds, {CNN_RESOLUTION}m grid): weighted IoU {cnn_iou.mean():.3f} +/- {cnn_iou.std(ddof=1):.3f}")
    print(f"\nwritten to {RESULTS_PATH}")
