"""External preprocessed arrays and explicit patient splits; no bundled cohort."""
from pathlib import Path
import json
import hashlib
import numpy as np
from cardiossm_analysis.runs import partition_runs
from cardiossm_analysis.temporal import rolling_average

DIRECTORY = None
SPLITS = None
SUBS = []
SUPPLEMENTAL = []
CHECK_RESOURCE = lambda: None


def configure(directory, split_file, study=False):
    global DIRECTORY, SPLITS, SUBS, SUPPLEMENTAL
    DIRECTORY = Path(directory)
    manifest = json.loads(Path(split_file).read_text())
    SPLITS = manifest.get('folds', manifest)
    SUPPLEMENTAL = manifest.get('supplemental_subjects', []) if 'folds' in manifest else []
    if not isinstance(SPLITS, dict) or not SPLITS:
        raise ValueError("split JSON must map fold numbers to train/val/test subject lists")
    all_subjects = set()
    outer_tests = []
    for fold in SPLITS:
        tr, va, te = split(int(fold))
        all_subjects.update(tr + va + te); outer_tests.extend(te)
    if len(outer_tests) != len(set(outer_tests)):
        raise ValueError("an outer-test subject may occur in only one fold")
    SUBS = sorted(all_subjects)
    if study:
        if len(SPLITS) != 4 or len(outer_tests) != 32 or len(set(SUPPLEMENTAL)) != 3:
            raise ValueError('study manifest requires four folds, 32 outer-test patients and three supplemental recordings')
        cohort = set(outer_tests)
        if cohort & set(SUPPLEMENTAL):
            raise ValueError('supplemental recordings cannot be outer-test patients')
        for f in SPLITS:
            tr, va, te = split(int(f))
            if len(va) != 4 or set(tr+va+te) != cohort | set(SUPPLEMENTAL) or not set(SUPPLEMENTAL) <= set(tr):
                raise ValueError('each fold must cover the cohort, four validation patients and training-only supplemental recordings')
        # Identity/order and permutation algorithm are external, not guessed.


def split(fold):
    if SPLITS is None or str(fold) not in SPLITS:
        raise ValueError("configure an explicit fold manifest first")
    parts = [SPLITS[str(fold)][key] for key in ("train", "val", "test")]
    if any(not isinstance(v, list) or not v for v in parts):
        raise ValueError("each split must be a nonempty subject list")
    flat = [s for v in parts for s in v]
    if any(not isinstance(s, str) or Path(s).name != s or s in (".", "..") for s in flat):
        raise ValueError("subject names must be safe basenames")
    if len(flat) != len(set(flat)):
        raise ValueError("train, validation and test identities must be disjoint")
    return tuple(parts)


def load_all(require_features=True):
    if DIRECTORY is None:
        raise ValueError("configure an input directory first")
    result = {}
    for subject in SUBS:
        CHECK_RESOURCE()
        with np.load(DIRECTORY / (subject + ".npz"), allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        n = len(d["t"])
        if (d["sig"].shape != (n, 3, 1000) or d["map"].shape != (n,)
                or n < 1 or not np.isfinite(d["t"]).all() or np.any(np.diff(d["t"]) <= 0)):
            raise ValueError("invalid raw-window arrays for external subject")
        available = np.asarray(d['availability_s'], float)
        if (available.shape != (n,) or not np.isfinite(available).all()
                or np.any(np.diff(available) <= 0) or np.any(available < d['t'] + 10)):
            raise ValueError("availability_s must increase and follow completion of each 10s window")
        if not np.isfinite(d["sig"]).all() or not np.isfinite(d["map"]).all():
            raise ValueError("BCG and MAP windows must be finite")
        if "fid" not in d:
            if require_features:
                raise ValueError("BCGnet/CardioSSM require 42 aligned explicit features")
            d["fid"] = np.zeros((n, 3, 42), np.float32)
        if d["fid"].shape != (n, 3, 42):
            raise ValueError("features must have shape [windows,3,42]")
        result[subject] = d
    if SUPPLEMENTAL:
        count = sum(int(pretraining_mask(result[s], supplemental=True).sum()) for s in SUPPLEMENTAL)
        if count != 2339:
            raise ValueError('supplemental pretraining support must contain 2339 windows')
    return result


def pretraining_mask(d, supplemental=False):
    key = 'pretraining_mask' if supplemental else 'retained_mask'
    if key not in d:
        raise ValueError('explicit '+key+' required for pretraining support')
    mask = np.asarray(d[key])
    if mask.dtype != np.bool_ or mask.shape != d['t'].shape or not mask.any():
        raise ValueError('pretraining support must be a nonempty boolean window mask')
    return mask


def pretraining_data(recordings, subjects, run_gap_basis=None):
    """Main cohort uses retained support; supplemental windows are explicit."""
    result = {}
    for s in subjects:
        d = recordings[s]; keep = pretraining_mask(d, s in SUPPLEMENTAL)
        result[s] = {k: d[k][keep] for k in ('t', 'availability_s', 'sig', 'fid', 'map')}
        result[s].update(run_support(result[s]['t'],result[s]['availability_s'],run_gap_basis))
    return result


def run_support(t, available, run_gap_basis):
    """Build block support and one authoritative partition for both resolutions."""
    bid=((t-t[0])//120).astype(int)
    ub,inverse=np.unique(bid,return_inverse=True)
    groups=[np.flatnonzero(inverse==i) for i in range(len(ub))]
    tb=np.maximum(t[0]+(ub+1)*120.,np.array([np.max(available[g]) for g in groups]))
    wr,br=partition_runs(available,inverse,tb,run_gap_basis)
    return dict(bid=bid,ub=ub,groups=groups,Tb=tb,rid=br,window_run_id=wr,run_gap_basis=run_gap_basis)


def retained(d, subject, split_unit=None, run_gap_basis=None):
    """Consume external retained support; the internal split unit is explicit."""
    if "retained_mask" not in d:
        raise ValueError("outer-test input requires an explicit retained_mask")
    keep = np.asarray(d["retained_mask"], bool)
    if keep.shape != d["t"].shape or not keep.any():
        raise ValueError("invalid retained_mask")
    t, y, sig, fid = [d[k][keep] for k in ("t", "map", "sig", "fid")]
    y = y.astype(float)
    available = d['availability_s'][keep]
    support=run_support(t,available,run_gap_basis)
    bid,ub,groups,tb=[support[k] for k in ('bid','ub','groups','Tb')]
    if any(len(g) < 6 for g in groups):
        raise ValueError("the retained mask must contain only eligible >=6-window 2min blocks")
    yb = np.array([np.mean(y[g]) for g in groups])
    k = int(.3 * len(ub)); ci = np.flatnonzero(np.isin(bid, ub[:k]))
    if k < 1 or k >= len(ub):
        raise ValueError("insufficient retained calibration/train/validation/test support")
    if split_unit == 'blocks':
        train_blocks = int(k*.8)
        itr = np.flatnonzero(np.isin(bid, ub[:train_blocks]))
        iva = np.flatnonzero(np.isin(bid, ub[train_blocks:k]))
    elif split_unit == 'windows':
        ktr = int(len(ci)*.8); itr, iva = ci[:ktr], ci[ktr:]
    elif split_unit is None:
        itr = iva = np.array([], dtype=int)
    else:
        raise ValueError('split_unit must be blocks or windows')
    if split_unit is not None and (not len(itr) or not len(iva)):
        raise ValueError('insufficient internal train/validation support')
    result = dict(support,t=t, availability_s=available, y=y, sig=sig, fid=fid,
                  Yb=yb, k=k, itr=itr, iva=iva, calibration_windows=ci,
                  te=np.arange(len(ub)) >= k)
    if 'hr_bpm' in d:
        hr = np.asarray(d['hr_bpm'], float)
        if hr.shape != d['t'].shape or not np.isfinite(hr[keep]).all():
            raise ValueError('monitor HR must be finite and aligned with windows')
        result['HRb'] = np.array([np.mean(hr[keep][g]) for g in groups])
    if 'recording_start_s' in d:
        start=float(d['recording_start_s'])
        if not np.isfinite(start) or start>d['t'][0]:
            raise ValueError('recording_start_s must precede the first stored window')
        result['recording_start_s']=start
    for key, value in d.items():
        if key.startswith("reading_indices_"):
            result[key] = value
    return result


def schedule(subject, d, hours, setting):
    if setting not in ('A', 'B') or hours not in (4,8,12,24):
        raise ValueError('unsupported monitoring setting or interval')
    first = d['k'] if setting == 'A' else 0
    key = ('reading_indices_' if setting == 'A' else 'reading_indices_B_') + str(hours)
    if setting=='B' and 'recording_start_s' not in d:
        raise ValueError('B requires the original recording_start_s, not the first retained block time')
    start=d['Tb'][first] if setting=='A' else float(d['recording_start_s'])
    times = np.arange(start, d['Tb'][-1]+1e-6, hours*3600.)
    expected = np.unique(np.searchsorted(d['Tb'], times, side='left'))
    pos = np.asarray(d[key]) if key in d else expected
    if (pos.ndim != 1 or not len(pos) or not np.issubdtype(pos.dtype, np.integer)
            or pos[0] != first or np.any(np.diff(pos) <= 0) or pos[-1] >= len(d["Tb"])):
        raise ValueError("reading indices must start at first test block and increase within range")
    if not np.array_equal(pos, expected):
        raise ValueError("reading indices must map each scheduled time to the next retained block")
    return pos


def smooth(y, t, rid):
    return rolling_average(y,t,run_id=rid)


def nz(features, norm):
    return np.nan_to_num(np.clip((features - norm[0]) / norm[1], -10, 10)).astype(np.float32)


def noise(subject, hours, sd, n):
    if sd not in (0,5) or n < 0:
        raise ValueError('Methods specifies error-free or SD5 readings')
    # SHA seed is stable across processes and model names. The convention is
    # explicit; it is not claimed to recover the paper's unpublished noise draw.
    seed = int.from_bytes(hashlib.sha256(f'{subject}:{hours}:reading-noise:0'.encode()).digest()[:8], 'little')
    return np.random.default_rng(seed).normal(0, sd, n) if sd else np.zeros(n)
