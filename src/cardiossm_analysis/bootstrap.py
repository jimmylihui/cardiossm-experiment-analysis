"""Paired patient resampling. No blocks/windows are treated as independent people."""
import numpy as np
from .metrics import aggregate_patient_values


def paired_interval(first, second, metric, resamples=10000, seed=0):
    """Difference of cohort aggregates, preserving pair identity.

    Missingness is resolved on the common finite patient set. This reports the
    difference of medians for correlation, not the median patient delta-r.
    Appropriate when tuning is fixed independently. For outcome-tuned qa use
    candidate_interval so tuning is repeated inside each resample.
    """
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != b.shape or a.ndim != 1 or resamples < 1:
        raise ValueError("matching patient vectors and positive resamples required")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        raise ValueError("at least two finite patient pairs required")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(a), size=(resamples, len(a)))
    delta = np.array([aggregate_patient_values(a[q], metric) - aggregate_patient_values(b[q], metric) for q in draws])
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(difference=aggregate_patient_values(a, metric) - aggregate_patient_values(b, metric),
                ci95_low=float(lo), ci95_high=float(hi), valid_pairs=len(a), resamples=resamples,
                patients_improved=int(np.sum(a < b if metric == "mae" else a > b)))


def candidate_interval(first_values, first_errors, second_values, second_errors,
                       metric, resamples=10000, seed=0):
    """Reselect qa after removing ALL bootstrap copies of held-out identity.

    Each values/errors array is [unique patients, candidates], aligned across
    models. An all-one-identity draw has no other identities for tuning and is
    omitted explicitly. No sentinel qa choice or zero correlation is inserted.
    """
    av, ae, bv, be = [np.asarray(v, float) for v in (first_values, first_errors, second_values, second_errors)]
    if (av.ndim != 2 or bv.ndim != 2 or av.shape[1] < 1 or bv.shape[1] < 1
            or ae.shape != av.shape or bv.shape != be.shape or len(bv) != len(av)):
        raise ValueError("aligned patient-by-candidate arrays required")
    if len(av) < 2 or resamples < 1 or not np.isfinite(ae).all() or not np.isfinite(be).all():
        raise ValueError("finite selection errors and at least two patients required")
    from .calibration import select_qa_other_patients
    ids = np.arange(len(av))
    aa = av[ids, select_qa_other_patients(ae)]
    bb = bv[ids, select_qa_other_patients(be)]
    finite = np.isfinite(aa) & np.isfinite(bb)
    if finite.sum() < 2:
        raise ValueError("at least two finite selected patient pairs required")
    point = aggregate_patient_values(aa[finite], metric) - aggregate_patient_values(bb[finite], metric)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        draw = rng.integers(len(av), size=len(av))
        if len(np.unique(draw)) < 2:
            continue
        first, second = [], []
        for heldout in draw:
            others = draw[draw != heldout]
            ac = int(np.argmin(np.mean(ae[others], axis=0)))
            bc = int(np.argmin(np.mean(be[others], axis=0)))
            first.append(av[heldout, ac]); second.append(bv[heldout, bc])
        first, second = np.asarray(first), np.asarray(second)
        valid = np.isfinite(first) & np.isfinite(second)
        if valid.sum() >= 2:
            deltas.append(aggregate_patient_values(first[valid], metric) - aggregate_patient_values(second[valid], metric))
    if not deltas:
        raise ValueError("no valid bootstrap replicates")
    lo, hi = np.quantile(deltas, [.025, .975])
    return dict(difference=float(point), ci95_low=float(lo), ci95_high=float(hi),
                requested_resamples=resamples, valid_resamples=len(deltas), reselects_qa=True,
                heldout_exclusion="all copies of patient identity")
