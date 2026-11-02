"""Patient-level error, Pearson correlation and direction concordance."""
import numpy as np
from .temporal import between_reading_segments, change_pairs


def pearson(prediction, reference):
    p, y = np.asarray(prediction, float), np.asarray(reference, float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("correlation inputs must be matching vectors")
    if len(p) < 2 or not np.isfinite(p).all() or not np.isfinite(y).all():
        return float("nan")
    if np.std(p) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(p, y)[0, 1])


def patient_metrics(time, reference, prediction, test_mask, reading_mask,
                    concordance_exclude_readings=False,run_id=None,include_changes=True):
    y, p = np.asarray(reference, float), np.asarray(prediction, float)
    test = np.asarray(test_mask, bool)
    if y.shape != p.shape or y.shape != test.shape or not test.any():
        raise ValueError("matching arrays and at least one test observation required")
    if not np.isfinite(y[test]).all() or not np.isfinite(p[test]).all():
        raise ValueError("all test references and predictions must be finite")
    z = []
    for q in between_reading_segments(time, y, test, reading_mask,run_id=run_id):
        r = pearson(p[q], y[q])
        if np.isfinite(r):
            z.append(np.arctanh(np.clip(r, -.999, .999)))
    mean_z = float(np.mean(z)) if z else float("nan")
    result=dict(mae=float(np.mean(abs(p[test]-y[test]))),r_test=pearson(p[test],y[test]),
        z_seg=mean_z,r_seg=float(np.tanh(mean_z)),valid_segments=len(z),test_blocks=int(test.sum()))
    if not include_changes:
        return result
    a, b = change_pairs(time, test, reading_mask, exclude_readings=concordance_exclude_readings,run_id=run_id)
    dy, dp = y[b] - y[a], p[b] - p[a]
    keep = abs(dy) >= 5.
    c = (float(np.mean(np.where(abs(dp[keep]) < 1e-9, .5, np.sign(dp[keep]) == np.sign(dy[keep]))))
         if keep.sum() >= 10 else float("nan"))
    result.update(concordance=c,change_pairs=int(keep.sum()))
    return result


def aggregate_patient_values(values, metric, aggregation=None):
    """MAE: mean; r_test/concordance: median; z_seg: tanh(median Fisher-z)."""
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    if not len(finite):
        return float("nan")
    if aggregation is not None:
        if aggregation not in ('mean','median'):
            raise ValueError('aggregation must be mean or median')
        value=float(np.mean(finite) if aggregation=='mean' else np.median(finite))
        return float(np.tanh(value)) if metric=='z_seg' else value
    if metric == "mae":
        return float(np.mean(finite))
    if metric == "z_seg":
        return float(np.tanh(np.median(finite)))
    if metric in ("r_test", "concordance", "auroc", "average_precision"):
        return float(np.median(finite))
    raise ValueError("unknown metric; r_seg aggregation requires patient z_seg")
