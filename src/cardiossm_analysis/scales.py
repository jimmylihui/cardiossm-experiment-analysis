"""Non-overlapping output averaging scales on aligned native observations."""
import numpy as np
from .metrics import pearson
from .temporal import validate_time


def scale_metrics(time, reference, prediction, test_mask, scales_s, gap_s=1800.):
    """Average prediction and target in the same non-overlapping bins per run.

    Use native-resolution observations, not an already 60min-smoothed curve.
    Averages include only observations explicitly marked as test. Bins do not
    straddle a run reset. This is an output-resolution analysis, not retraining
    an encoder at different input windows or changing its history length.
    """
    t = validate_time(time)
    y, p = np.asarray(reference, float), np.asarray(prediction, float)
    test = np.asarray(test_mask, bool)
    if any(v.shape != t.shape for v in (y, p, test)) or not test.any():
        raise ValueError("matching arrays and nonempty test mask required")
    if not np.isfinite(y[test]).all() or not np.isfinite(p[test]).all():
        raise ValueError("finite test predictions and references required")
    dt = np.diff(t)
    dt = dt[dt <= gap_s]
    if not len(dt):
        raise ValueError("insufficient native observations")
    native = float(np.median(dt))
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > gap_s) + 1, len(t)]
    rows = []
    for scale in scales_s:
        if not np.isfinite(scale) or scale < native - 1e-6:
            raise ValueError("averaging scale cannot be below native sampling interval")
        yy, pp = [], []
        for a, b in zip(cuts[:-1], cuts[1:]):
            bins = np.floor((t[a:b] - t[a]) / scale).astype(np.int64)
            for key in np.unique(bins[test[a:b]]):
                ids = np.flatnonzero((bins == key) & test[a:b]) + a
                yy.append(float(np.mean(y[ids]))); pp.append(float(np.mean(p[ids])))
        yy, pp = np.asarray(yy), np.asarray(pp)
        rows.append(dict(scale_s=float(scale), observations=len(yy),
                         mae=float(np.mean(abs(yy - pp))), correlation=pearson(pp, yy)))
    return rows
