"""Time support, averaging and reading-excluded evaluation intervals."""
import numpy as np


def validate_time(time):
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("time must be a finite, strictly increasing vector in seconds")
    return t


def rolling_average(values, time, window_s=3600., mode="past_only", gap_s=1800.):
    """Run-aware mean on completed observations in (t-window,t], startup min_periods=1."""
    t = validate_time(time)
    x = np.asarray(values, dtype=float)
    if x.ndim not in (1, 2) or len(x) != len(t) or window_s <= 0 or gap_s <= 0:
        raise ValueError("incompatible values, time, window or gap")
    if mode != "past_only":
        raise ValueError("mode must be past_only")
    out = np.full_like(x, np.nan)
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > gap_s) + 1, len(t)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        tt = t[a:b]
        lo = np.searchsorted(tt, tt - window_s, side="right")
        hi = np.arange(1, b - a + 1)
        for j, (left, right) in enumerate(zip(lo, hi)):
            v = x[a + left:a + right]
            count = np.isfinite(v).sum(axis=0)
            total = np.where(np.isfinite(v), v, 0).sum(axis=0)
            out[a + j] = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    return out


def change_pairs(time, test_mask, reading_mask, steps=5, max_span_s=720., exclude_readings=True):
    """Endpoints separated by `steps` retained test blocks; inclusive reading exclusion."""
    t = validate_time(time)
    test = np.asarray(test_mask, bool)
    readings = np.asarray(reading_mask, bool)
    if test.shape != t.shape or readings.shape != t.shape or steps < 1:
        raise ValueError("masks must match time and steps must be positive")
    ids = np.flatnonzero(test)
    end = ids[steps:]
    start = ids[:-steps]
    keep = t[end] - t[start] <= max_span_s
    if exclude_readings:
        counts = np.r_[0, np.cumsum(readings)]
        keep &= counts[end + 1] - counts[start] == 0
    return start[keep], end[keep]


def between_reading_segments(time, reference, test_mask, reading_mask, gap_s=1800.,
                             min_blocks=5, min_span_s=3600., min_reference_sd=1.):
    """Exclude each reading block; split at gaps; apply original segment eligibility."""
    t = validate_time(time)
    y = np.asarray(reference, float)
    test = np.asarray(test_mask, bool)
    readings = np.asarray(reading_mask, bool)
    if any(a.shape != t.shape for a in (y, test, readings)):
        raise ValueError("reference and masks must match time")
    ids = np.flatnonzero(test)
    locations = np.flatnonzero(readings[ids])
    segments = []
    for a, b in zip(locations, list(locations[1:]) + [len(ids)]):
        part = ids[a + 1:b]
        for q in np.split(part, np.flatnonzero(np.diff(t[part]) > gap_s) + 1):
            if (len(q) >= min_blocks and t[q[-1]] - t[q[0]] >= min_span_s
                    and np.isfinite(y[q]).all() and np.std(y[q]) >= min_reference_sd):
                segments.append(q)
    return segments
