"""Align external training outputs without re-smoothing or guessing model scope."""
import numpy as np
from .temporal import rolling_average
from .runs import record_runs, validate_run_ids


def align_training_outputs(by_model, tag):
    """One patient's chosen condition from each model, at native 2min blocks."""
    if not by_model:
        raise ValueError('at least one explicitly named model required')
    names=list(by_model);first=by_model[names[0]]
    required=('Tb','Yb','raw_t','raw_bid','ub','availability_s',tag+'_positions',
              tag+'_test_mask',tag+'_calibration_mask','run_id','window_run_id','run_gap_basis')
    for d in by_model.values():
        for key in required:
            if key not in d or not np.array_equal(d[key],first[key]):
                raise ValueError('model support/condition differs: '+key)
        if tag+'_block' not in d or np.asarray(d[tag+'_block']).shape!=np.asarray(first['Tb']).shape:
            raise ValueError('unsmoothed block outputs required')
    t=np.asarray(first['Tb']);read=np.zeros(len(t),bool);read[first[tag+'_positions']]=True
    reading_values=np.asarray(first.get(tag+'_readings',np.full(len(t),np.nan)))
    for d in by_model.values():
        other=np.asarray(d.get(tag+'_readings',np.full(len(t),np.nan)))
        if not np.array_equal(reading_values[read],other[read]):
            raise ValueError('models must share the same realized readings')
    out=dict(time_s=t,reference=np.asarray(first['Yb']),test_mask=first[tag+'_test_mask'],
        calibration_mask=first[tag+'_calibration_mask'],reading_mask=read,reading_values=reading_values,
        run_id=first['run_id'],run_gap_basis=first['run_gap_basis'],
        model_names=np.asarray(names),neural_predictions=np.column_stack([by_model[n][tag+'_block'] for n in names]))
    runs,_=record_runs(out)
    window_runs=validate_run_ids(first['window_run_id'],len(first['raw_t']))
    block_index=np.searchsorted(first['ub'],first['raw_bid'])
    if np.any(block_index>=len(runs)) or not np.array_equal(window_runs,runs[block_index]):
        raise ValueError('window and block run IDs must describe the same partition')
    hrs=[np.asarray(d['HRb']) for d in by_model.values() if 'HRb' in d]
    if hrs:
        if any(not np.array_equal(hrs[0],h) for h in hrs[1:]):
            raise ValueError('model monitor HR support differs')
        hr=hrs[0]
        if hr.shape!=t.shape or not np.isfinite(hr).all():raise ValueError('invalid monitor block HR')
        out['hr_bpm']=hr
        cal=~np.asarray(first['te'],bool)
        h=hr-rolling_average(hr,t,run_id=runs);ym=rolling_average(out['reference'],t,run_id=runs)
        out['calibration_variance']=np.asarray(np.var((out['reference']-ym-h)[cal]))
    return out
