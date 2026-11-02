"""One explicit run partition shared by window states and block analyses."""
import numpy as np

RUN_GAP_BASES = ('window_completion', 'block_completion')


def validate_run_ids(run_id, count):
    ids = np.asarray(run_id)
    if (ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids[1:] < ids[:-1])):
        raise ValueError('run_id must be aligned, nonnegative, monotone integer labels')
    return ids


def run_slices(time, run_id=None, gap_s=1800.):
    """Explicit IDs take precedence; timestamp fallback is for standalone APIs."""
    t = np.asarray(time)
    if run_id is None:
        cuts = np.flatnonzero(np.diff(t) > gap_s) + 1
    else:
        ids = validate_run_ids(run_id, len(t))
        cuts = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    edges = np.r_[0, cuts, len(t)]
    return zip(edges[:-1], edges[1:])


def partition_runs(window_completion, window_to_block, block_completion, basis):
    """Choose the gap clock once, then project its IDs to both resolutions.

    A block cannot straddle multiple window-defined runs. Reject such support
    rather than silently averaging signals from separate runs into one block.
    """
    if basis not in RUN_GAP_BASES:
        raise ValueError('specify run gap basis: window_completion or block_completion')
    wt, bt = [np.asarray(x, float) for x in (window_completion, block_completion)]
    ix = np.asarray(window_to_block)
    if (wt.ndim != 1 or bt.ndim != 1 or not len(wt) or not len(bt)
            or not np.isfinite(wt).all() or not np.isfinite(bt).all()
            or np.any(np.diff(wt) <= 0) or np.any(np.diff(bt) <= 0)
            or ix.shape != wt.shape or not np.issubdtype(ix.dtype, np.integer)
            or not np.array_equal(np.unique(ix), np.arange(len(bt))) or np.any(np.diff(ix) < 0)):
        raise ValueError('invalid chronological window/block support')
    if basis == 'block_completion':
        block_run = np.r_[0, np.cumsum(np.diff(bt) > 1800)]
        return block_run[ix], block_run
    window_run = np.r_[0, np.cumsum(np.diff(wt) > 1800)]
    starts = np.r_[0, np.flatnonzero(np.diff(ix)) + 1]
    ends = np.r_[starts[1:] - 1, len(ix) - 1]
    if np.any(window_run[starts] != window_run[ends]):
        raise ValueError('a 2min block straddles window-defined runs; fix input support explicitly')
    return window_run, window_run[starts]


def record_runs(record):
    """Analysis records must carry the runner's partition, never infer a new one."""
    if 'run_id' not in record or 'run_gap_basis' not in record:
        raise ValueError('run_id and run_gap_basis required; regenerate aligned inputs with shared run metadata')
    basis = str(np.asarray(record['run_gap_basis']).item())
    if basis not in RUN_GAP_BASES:
        raise ValueError('unsupported run_gap_basis')
    return validate_run_ids(record['run_id'], len(record['time_s'])), basis
