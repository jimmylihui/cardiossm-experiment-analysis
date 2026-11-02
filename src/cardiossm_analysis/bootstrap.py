"""Paired patient resampling. No blocks/windows are treated as independent people."""
import numpy as np
from .metrics import aggregate_patient_values


def paired_interval(first, second, metric, resamples=10000, seed=0,
                    aggregation=None, contrast='difference_of_aggregates'):
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
    def difference(x,y):
        if contrast=='median_paired_delta':
            if metric=='z_seg':x,y=np.tanh(x),np.tanh(y)
            return float(np.median(x-y))
        if contrast!='difference_of_aggregates':raise ValueError('unknown contrast')
        return aggregate_patient_values(x,metric,aggregation)-aggregate_patient_values(y,metric,aggregation)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(a), size=(resamples, len(a)))
    delta = np.array([difference(a[q],b[q]) for q in draws])
    lo, hi = np.quantile(delta, [.025, .975])
    return dict(difference=difference(a,b),aggregation=aggregation or 'metric_default',contrast=contrast,
                ci95_low=float(lo), ci95_high=float(hi), valid_pairs=len(a), resamples=resamples,
                patients_improved=int(np.sum(a < b if metric == "mae" else a > b)))


def candidate_interval(first_values, first_errors, second_values, second_errors,
                       metric, resamples=10000, seed=0,aggregation=None,
                       contrast='difference_of_aggregates'):
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
    def difference(a,b):
        if contrast=='median_paired_delta':
            if metric=='z_seg':a,b=np.tanh(a),np.tanh(b)
            return float(np.median(a-b))
        if contrast!='difference_of_aggregates':raise ValueError('unknown contrast')
        return aggregate_patient_values(a,metric,aggregation)-aggregate_patient_values(b,metric,aggregation)
    point = difference(aa[finite],bb[finite])
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
            deltas.append(difference(first[valid],second[valid]))
    if not deltas:
        raise ValueError("no valid bootstrap replicates")
    lo, hi = np.quantile(deltas, [.025, .975])
    return dict(difference=float(point),aggregation=aggregation or 'metric_default',contrast=contrast,ci95_low=float(lo), ci95_high=float(hi),
                requested_resamples=resamples, valid_resamples=len(deltas), reselects_qa=True,
                heldout_exclusion="all copies of patient identity")


def holm_adjust(pvalues):
    """Adjust one explicitly supplied comparison family; do not choose a test."""
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('one finite p-value vector in [0,1] required')
    order=np.argsort(p,kind='stable');result=np.empty_like(p)
    result[order]=np.minimum(1.,np.maximum.accumulate(p[order]*np.arange(len(p),0,-1)))
    return result
