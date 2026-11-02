"""External preprocessed arrays and explicit patient splits; no bundled cohort."""
from pathlib import Path
import json
import numpy as np

DIRECTORY = None
SPLITS = None
SUBS = []
CHECK_RESOURCE = lambda: None


def configure(directory, split_file):
    global DIRECTORY, SPLITS, SUBS
    DIRECTORY = Path(directory)
    SPLITS = json.loads(Path(split_file).read_text())
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
    return result


def retained(d, subject):
    """Use externally fixed retained windows; do not recompute quality selection.

    Bin grouping, initial first30% and internal first80% raw-window split match
    the original adaptation. Require explicit reading block indices per interval.
    """
    if "retained_mask" not in d:
        raise ValueError("outer-test input requires an explicit retained_mask")
    keep = np.asarray(d["retained_mask"], bool)
    if keep.shape != d["t"].shape or not keep.any():
        raise ValueError("invalid retained_mask")
    t, y, sig, fid = [d[k][keep] for k in ("t", "map", "sig", "fid")]
    y = y.astype(float)
    bid = ((t - t[0]) // 120).astype(int)
    ub = np.unique(bid); groups = [np.flatnonzero(bid == b) for b in ub]
    if any(len(g) < 6 for g in groups):
        raise ValueError("the retained mask must contain only eligible >=6-window 2min blocks")
    available = d['availability_s'][keep]
    tb = np.maximum(t[0] + (ub + 1)*120., np.array([np.max(available[g]) for g in groups]))
    if np.any(np.diff(tb) <= 0):
        raise ValueError("completed block availability must increase")
    yb = np.array([np.mean(y[g]) for g in groups])
    rid = np.r_[0, np.cumsum(np.diff(tb) > 1800)]
    k = int(.3 * len(ub)); ci = np.flatnonzero(np.isin(bid, ub[:k])); ktr = int(len(ci) * .8)
    if k < 1 or k >= len(ub) or ktr < 1 or ktr >= len(ci):
        raise ValueError("insufficient retained calibration/train/validation/test support")
    result = dict(t=t, availability_s=available, y=y, sig=sig, fid=fid, bid=bid, ub=ub, groups=groups,
                  Tb=tb, Yb=yb, rid=rid, k=k, itr=ci[:ktr], iva=ci[ktr:],
                  te=np.arange(len(ub)) >= k)
    for key, value in d.items():
        if key.startswith("reading_indices_"):
            result[key] = value
    return result


def schedule(subject, d, hours, setting):
    if setting != "A":
        raise ValueError("published training runner uses initial+periodic adaptation only")
    key = "reading_indices_" + str(hours)
    times = np.arange(d['Tb'][d['k']], d['Tb'][-1]+1e-6, hours*3600.)
    expected = np.unique(np.searchsorted(d['Tb'], times, side='left'))
    pos = np.asarray(d[key]) if key in d else expected
    if (pos.ndim != 1 or not len(pos) or not np.issubdtype(pos.dtype, np.integer)
            or pos[0] != d["k"] or np.any(np.diff(pos) <= 0) or pos[-1] >= len(d["Tb"])):
        raise ValueError("reading indices must start at first test block and increase within range")
    if not np.array_equal(pos, expected):
        raise ValueError("reading indices must map each scheduled time to the next retained block")
    return pos


def smooth(y, t, rid):
    return np.array([np.mean(y[(rid == rid[i]) & (t > t[i]-3600) & (t <= t[i])]) for i in range(len(t))])


def nz(features, norm):
    return np.nan_to_num(np.clip((features - norm[0]) / norm[1], -10, 10)).astype(np.float32)


def noise(subject, hours, sd, n):
    if sd != 0:
        raise ValueError("this runner fixes error-free calibration readings")
    return np.zeros(n)
