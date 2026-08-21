"""Classification metrics shared across milestones.

Operates on integer label arrays only (`y_true`, `y_pred`), not on any
model-specific object, so the same functions apply to the sklearn
baselines and later the CNN without modification.

Implemented directly from the confusion matrix instead of calling
`sklearn.metrics.classification_report`/`jaccard_score`, so the formulas
behind precision/recall/IoU are visible rather than hidden inside a library
call. Cross-checked once against sklearn's own `accuracy_score`/
`precision_score`/`recall_score`/`jaccard_score` (`average="weighted"`) and
found to match exactly.
"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Row = true class, column = predicted class. cm[i, j] = count of true-i predicted-as-j."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def _per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, and IoU (Jaccard index) per class, derived from `cm`.

    For class c: TP = cm[c, c], FP = column sum minus TP (other classes
    predicted as c), FN = row sum minus TP (class c predicted as something
    else).
    - precision = TP / (TP + FP): of tiles predicted class c, fraction
      actually class c.
    - recall = TP / (TP + FN): of tiles actually class c, fraction predicted
      correctly.
    - IoU = TP / (TP + FP + FN): overlap between the predicted-c set and the
      true-c set, over their union -- stricter than precision or recall
      alone since it is penalized by both kinds of error at once.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)  # number of true instances per class

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)

    return {"precision": precision, "recall": recall, "iou": iou, "support": support}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Full metrics report: overall accuracy, per-class, and support-weighted averages.

    Weighted averages sum each class's score times its support (number of
    true instances), divided by total support -- so a class with more val
    samples counts more, the same convention as sklearn's
    `average="weighted"`. That matters here because of the dataset's ~3.6:1
    class imbalance across the 6 road-scene classes: unweighted averaging
    would let a class with 30 val samples move the score as much as one
    with 200.
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    per_class = _per_class_scores(cm)
    support = per_class["support"]
    total = int(support.sum())

    accuracy = float(np.trace(cm) / total)
    weighted = {
        metric: float(np.sum(per_class[metric] * support) / total)
        for metric in ("precision", "recall", "iou")
    }

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class": {
            class_names[i]: {
                "precision": float(per_class["precision"][i]),
                "recall": float(per_class["recall"][i]),
                "iou": float(per_class["iou"][i]),
                "support": int(support[i]),
            }
            for i in range(n_classes)
        },
        "weighted": weighted,
    }
