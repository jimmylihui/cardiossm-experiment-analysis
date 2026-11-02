"""Original retrospective offset + HR assembly, not a prospective filter claim."""
import numpy as np
from .temporal import validate_time


def offset_hr(time, reference, smoothed_reference, neural, hr_deviation,
              calibration_mask, reading_mask, qa=2., calibration_stride=15):
    """Return slow estimate and slow+HR, recording prediction BEFORE KF update.

    Neural input is an externally generated, already averaged MAP contribution.
    It is centered using the WHOLE supplied record, as in the retrospective
    protocol. Initial calibration uses smoothed MAP with observation variance 4;
    periodic updates use MAP-HR with initial-calibration residual variance.
    Setting neural to zero gives the matched no-BCG Offset+HR baseline.
    This function does not implement neural fitting or its reading-block order.
    """
    t = validate_time(time)
    y, ym, u, h = [np.asarray(v, float) for v in (reference, smoothed_reference, neural, hr_deviation)]
    cal, read = np.asarray(calibration_mask, bool), np.asarray(reading_mask, bool)
    if any(v.shape != t.shape for v in (y, ym, u, h, cal, read)):
        raise ValueError("all assembly inputs must match time")
    if not all(np.isfinite(v).all() for v in (y, ym, u, h)) or not cal.any():
        raise ValueError("finite inputs and a nonempty calibration set required")
    if qa < 0 or calibration_stride < 1:
        raise ValueError("invalid qa or calibration stride")
    observations = {int(i): (float(ym[i]), 4.) for i in np.flatnonzero(cal)[::calibration_stride]}
    residual_variance = float(np.var((y - ym - h)[cal]))
    for i in np.flatnonzero(read):
        observations[int(i)] = (float(y[i] - h[i]), residual_variance)
    centered = u - np.mean(u)
    offset, variance, started = 0., 10000., False
    slow = np.full(len(t), np.nan)
    for i in range(len(t)):
        variance += qa * (0. if i == 0 else t[i] - t[i - 1]) / 3600.
        if started:
            slow[i] = offset + centered[i]
        if i in observations:
            value, noise = observations[i]
            denominator = variance + noise
            gain = variance / denominator if denominator > 0 else 0.
            offset += gain * (value - centered[i] - offset)
            variance *= 1. - gain
            started = True
    return slow, slow + h


def select_qa_other_patients(selection_errors):
    """Other-patient outcome selection; RETROSPECTIVE, not inner validation.

    Rows must be unique patients and columns qa candidates. Every held-out
    patient's own row is excluded. This does not repair label leakage already
    present in how the supplied candidate errors were produced.
    """
    e = np.asarray(selection_errors, float)
    if e.ndim != 2 or len(e) < 2 or not np.isfinite(e).all():
        raise ValueError("finite patient-by-candidate errors, at least two patients required")
    return np.argmin(e.sum(axis=0)[None, :] - e, axis=1)
