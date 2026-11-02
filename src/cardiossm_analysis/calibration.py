"""Offset + HR assembly with predictions recorded before reading updates."""
import numpy as np
from .temporal import validate_time
from .runs import record_runs


def offset_hr(time, reference, smoothed_reference, neural, hr_deviation,
              calibration_mask, reading_mask, qa=None, calibration_stride=15,
              reading_values=None, observation_variance=None):
    """Return level and level+HR using supplied past-only block contributions."""
    t = validate_time(time)
    y, ym, u, h = [np.asarray(v, float) for v in (reference, smoothed_reference, neural, hr_deviation)]
    cal, read = np.asarray(calibration_mask, bool), np.asarray(reading_mask, bool)
    if any(v.shape != t.shape for v in (y, ym, u, h, cal, read)):
        raise ValueError("all assembly inputs must match time")
    if not all(np.isfinite(v).all() for v in (y, ym, u, h)):
        raise ValueError("finite inputs required")
    if qa is None or not np.isfinite(qa) or qa < 0 or calibration_stride < 1:
        raise ValueError("invalid qa or calibration stride")
    observations = {int(i): (float(ym[i]), 4.) for i in np.flatnonzero(cal)[::calibration_stride]}
    if cal.any():
        residual_variance = float(np.var((y - ym - h)[cal]))
    else:
        if observation_variance is None or not np.isfinite(observation_variance) or observation_variance < 0:
            raise ValueError('without calibration, supply the other-patient observation variance')
        residual_variance = float(observation_variance)
    readings=y if reading_values is None else np.asarray(reading_values,float)
    if readings.shape != t.shape or not np.isfinite(readings[read]).all():
        raise ValueError('readings must be finite on scheduled support')
    for i in np.flatnonzero(read):
        observations[int(i)] = (float(readings[i] - h[i]), residual_variance)
    offset, variance, started = 0., 10000., not cal.any()
    slow = np.full(len(t), np.nan)
    for i in range(len(t)):
        variance += qa * (0. if i == 0 else t[i] - t[i - 1]) / 3600.
        if started:
            slow[i] = offset + u[i]
        if i in observations:
            value, noise = observations[i]
            denominator = variance + noise
            gain = variance / denominator if denominator > 0 else 0.
            offset += gain * (value - u[i] - offset)
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


def other_patient_reading_variance(calibration_variances):
    """No-initial-calibration R; each patient's own variance is excluded."""
    v=np.asarray(calibration_variances,float)
    if v.ndim != 1 or len(v)<2 or not np.isfinite(v).all() or (v<0).any():
        raise ValueError('finite calibration variances from at least two unique patients required')
    return np.array([np.median(np.delete(v,i))+25. for i in range(len(v))])


def assemble_cohort(records, models, qa=None):
    """Assemble chosen models and select qa by other patients' test MAE.

    Each record is one aligned patient. Neural inputs are unsmoothed block
    means. Unselected model streams remain network outputs. Return candidate
    predictions/errors too, so callers can repeat tuning inside bootstrap.
    """
    from .temporal import rolling_average
    names=list(records[0]['model_names']) if records else []
    if not records or not models or len(names)!=len(set(names)) or not set(models)<=set(names) or 'OffsetHR' in names:
        raise ValueError('specify valid models to assemble; OffsetHR is reserved')
    if qa is not None and (not np.isfinite(qa) or qa<0):
        raise ValueError('fixed qa must be finite and nonnegative')
    if any(list(r['model_names'])!=names for r in records):
        raise ValueError('identical ordered model coverage required across patients')
    candidates=np.asarray([.5,2.,8.,32.] if qa is None else [qa])
    if qa is None and len(records)<2:
        raise ValueError('other-patient qa selection needs at least two patients')
    no_initial=[not np.asarray(r['calibration_mask'],bool).any() for r in records]
    partitions=[record_runs(r) for r in records]
    if len({basis for _,basis in partitions})!=1:
        raise ValueError('split different run-gap protocols before cohort selection')
    if any(no_initial) and not all(no_initial):
        raise ValueError('split monitoring conditions before cohort selection')
    variances=other_patient_reading_variance([float(r['calibration_variance']) for r in records]) if all(no_initial) else [None]*len(records)
    result=[];predictions=[];errors=[]
    output_names=['OffsetHR']+names
    for r,variance,(runs,_) in zip(records,variances,partitions):
        t=np.asarray(r['time_s']);y=np.asarray(r['reference']);test=np.asarray(r['test_mask'],bool)
        neural=np.asarray(r['neural_predictions'])
        if neural.shape!=(len(t),len(names)) or not test.any():
            raise ValueError('aligned block-by-model neural predictions and nonempty test support required')
        ym=rolling_average(y,t,run_id=runs);h=r['hr_bpm']-rolling_average(r['hr_bpm'],t,run_id=runs)
        u=rolling_average(neural,t,run_id=runs)
        cube=np.empty((len(t),len(output_names),len(candidates)))
        for c,q in enumerate(candidates):
            for j,name in enumerate(output_names):
                if j and name not in models:
                    cube[:,j,c]=neural[:,j-1]
                else:
                    source=np.zeros(len(t)) if j==0 else u[:,j-1]
                    cube[:,j,c]=offset_hr(t,y,ym,source,h,r['calibration_mask'],r['reading_mask'],qa=q,
                        reading_values=r.get('reading_values'),observation_variance=variance)[1]
        if not np.isfinite(cube[test]).all():raise ValueError('test predictions must all be available')
        predictions.append(cube);errors.append(np.mean(abs(cube[test]-y[test,None,None]),axis=0))
    errors=np.asarray(errors)  # patient, model, candidate
    selected=np.column_stack([select_qa_other_patients(errors[:,j]) for j in range(len(output_names))]) if qa is None else np.zeros((len(records),len(output_names)),int)
    for i,r in enumerate(records):
        chosen=np.column_stack([predictions[i][:,j,selected[i,j]] for j in range(len(output_names))])
        result.append(dict(r,model_names=np.asarray(output_names),predictions=chosen,
            selected_qa=candidates[selected[i]],qa_selection='other_patient_test_mae' if qa is None else 'explicit_fixed'))
    return result,dict(qa_candidates=candidates,candidate_predictions=predictions,
                       candidate_errors=errors,selected_indices=selected)
