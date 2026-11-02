"""Patient-conditional event ranking on fixed, reading-free windows."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .temporal import change_pairs


def event_metrics(time, reference, prediction, test_mask, reading_mask,
                  direction="drop", threshold_mmhg=10., min_each_class=5):
    if direction not in ("drop", "rise") or threshold_mmhg <= 0 or min_each_class < 1:
        raise ValueError("invalid event definition")
    y, p = np.asarray(reference, float), np.asarray(prediction, float)
    if y.shape != p.shape or y.shape != np.asarray(time).shape:
        raise ValueError("matching time, reference and prediction vectors required")
    a, b = change_pairs(time, test_mask, reading_mask, exclude_readings=True)
    dy, dp = y[b] - y[a], p[b] - p[a]
    if not np.isfinite(dy).all() or not np.isfinite(dp).all():
        raise ValueError("event endpoints must be finite")
    labels = dy <= -threshold_mmhg if direction == "drop" else dy >= threshold_mmhg
    scores = -dp if direction == "drop" else dp
    positive, negative = int(labels.sum()), int((~labels).sum())
    eligible = min(positive, negative) >= min_each_class
    return dict(direction=direction, windows=len(labels), positives=positive, negatives=negative,
                eligible=eligible,
                auroc=float(roc_auc_score(labels, scores)) if eligible else float("nan"),
                average_precision=float(average_precision_score(labels, scores)) if eligible else float("nan"))
