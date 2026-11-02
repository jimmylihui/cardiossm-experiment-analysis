"""Descriptive feature screening with temporal separation and a joint max-null."""
import numpy as np
from .temporal import validate_time, rolling_average
from .metrics import pearson


def feature_correlations(features, reference, min_pairs=60, min_reference_sd=1.):
    x, y = np.asarray(features, float), np.asarray(reference, float)
    if x.ndim != 2 or y.shape != (len(x),):
        raise ValueError("features must be time-by-feature and reference a matching vector")
    r = np.full(x.shape[1], np.nan)
    for j in range(x.shape[1]):
        keep = np.isfinite(x[:, j]) & np.isfinite(y)
        if keep.sum() >= min_pairs and np.std(y[keep]) >= min_reference_sd:
            r[j] = pearson(x[keep, j], y[keep])
    return r


def heldout_screen(time, reference, features, train_fraction=.3, embargo_s=3600.,
                   null_resamples=499, seed=0,run_id=None):
    """Past-only 60min means, training-only feature selection, later evaluation.

    The max-null jointly circular-shifts all features within contiguous held-out
    chunks (>180s splits). Chunks need >=61 blocks; shifts are 30..L-30 blocks.
    This is the original nominal 2min-block design. Inputs must actually use
    retained 2min blocks, with completed-block availability timestamps.
    """
    t = validate_time(time)
    x = np.asarray(features, float)
    y = np.asarray(reference, float)
    if x.ndim != 2 or y.shape != t.shape or len(x) != len(t):
        raise ValueError("aligned time, reference and feature matrix required")
    if not 0 < train_fraction < 1 or embargo_s < 0 or null_resamples < 1:
        raise ValueError("invalid split, embargo or null resamples")
    split = int(train_fraction * len(t))
    if split < 1 or split >= len(t):
        raise ValueError("insufficient blocks for split")
    smooth = rolling_average(np.column_stack((y, x)), t,run_id=run_id)
    y, x = smooth[:, 0], smooth[:, 1:]
    later = np.flatnonzero((np.arange(len(t)) >= split) & (t > t[split - 1] + embargo_s))
    train_r = feature_correlations(x[:split], y[:split])
    eligible = np.flatnonzero(np.isfinite(train_r))
    if not len(eligible):
        return dict(status="insufficient_training", selected_feature=None)
    selected = int(eligible[np.argmax(abs(train_r[eligible]))])
    test_r = feature_correlations(x[later], y[later])
    result = dict(status="heldout_only", selected_feature=selected,
                  training_r=float(train_r[selected]), heldout_r=float(test_r[selected]),
                  train_blocks=split, heldout_blocks=len(later), null_p_selected=None)
    chunks = [q for q in np.split(later, np.flatnonzero(np.diff(t[later]) > 180) + 1) if len(q) >= 61]
    if not chunks:
        return result
    ids = np.concatenate(chunks)
    observed = feature_correlations(x[ids], y[ids])
    candidates = eligible[np.isfinite(observed[eligible])]
    if not len(candidates) or not np.isfinite(observed[selected]):
        return result
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(null_resamples):
        shifted = np.concatenate([np.roll(x[q], int(rng.integers(30, len(q) - 29)), axis=0) for q in chunks])
        rr = feature_correlations(shifted, y[ids])
        if not np.isfinite(rr[candidates]).any():
            raise ValueError("undefined correlation in null replicate")
        distribution.append(float(np.nanmax(abs(rr[candidates]))))
    dist = np.asarray(distribution)
    result.update(status="tested", null_blocks=len(ids), null_chunks=len(chunks),
                  null_observed_selected_r=float(observed[selected]),
                  null_p_selected=float((1 + np.sum(dist >= abs(observed[selected]))) / (null_resamples + 1)),
                  null_p_max=float((1 + np.sum(dist >= np.max(abs(observed[candidates])))) / (null_resamples + 1)))
    return result
