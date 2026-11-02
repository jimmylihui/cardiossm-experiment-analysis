"""Every chronological patient/axis window in the shared, explicit run partition."""
import numpy as np
from cardiossm_analysis.runs import validate_run_ids, RUN_GAP_BASES


class Pool:
    def __init__(self, data, subjects):
        self.axes = 3
        self.offset, self.count, self.report = {}, {}, {}
        xs, fs, ys, times, patients, axes, streams, keys, runs = [], [], [], [], [], [], [], [], []
        base = stream_offset = 0
        for patient, subject in enumerate(subjects):
            d = data[subject]; n = len(d['t'])
            self.offset[subject] = base; self.count[subject] = n
            xs.append(d['sig'].reshape(-1, 1, 1000).astype(np.float16))
            fs.append(d['fid'].reshape(-1, 42).astype(np.float32))
            ys.append(np.repeat(d.get('map', d.get('y')), 3))
            times.append(np.repeat(d['availability_s'], 3)); patients.append(np.full(n*3, patient)); axes.append(np.tile(np.arange(3), n))
            order = np.argsort(d['t'], kind='stable')
            t = np.asarray(d['availability_s'])[order]
            if n < 1 or np.any(np.diff(t) <= 0):
                raise ValueError('window completion times must increase within each patient')
            if d.get('run_gap_basis') not in RUN_GAP_BASES or 'window_run_id' not in d:
                raise ValueError('history requires the shared window_run_id and explicit run_gap_basis')
            run=validate_run_ids(np.asarray(d['window_run_id'])[order],n)
            original_run=np.empty(n,dtype=run.dtype);original_run[order]=run
            runs.append(np.repeat(original_run,3))
            cuts = np.r_[0, np.flatnonzero(np.diff(run)) + 1]
            lower = np.repeat(cuts, np.diff(np.r_[cuts, n]))
            key = np.empty((n*3, 2), np.int64); max_tokens = 0
            max_tokens = int(np.max(np.arange(n) - lower + 1))
            for axis in range(3):
                ix = base + order*3 + axis; streams.append(ix)
                key[order*3 + axis] = np.c_[lower + stream_offset, np.arange(n) + stream_offset]
                stream_offset += n
            keys.append(key)
            self.report[subject] = dict(raw_windows=n, axis_samples=n*3, max_history_tokens=max_tokens,
                history_policy='all_causal_within_run',run_gap_basis=d['run_gap_basis'], gap_seconds=1800, time_cap=None, token_cap=None)
            base += n*3
        self.X = np.concatenate(xs); self.F = np.concatenate(fs); self.Y = np.concatenate(ys).astype(np.float32)
        self.T = np.concatenate(times); self.pid = np.concatenate(patients); self.axis = np.concatenate(axes)
        self.run_id=np.concatenate(runs)
        self.stream = np.concatenate(streams); self.key = np.concatenate(keys)
        assert np.array_equal(np.sort(self.stream), np.arange(base))
        queries = np.unique(np.r_[np.random.default_rng(0).integers(base, size=min(base, 500)), 0, base-1])
        for q in queries:
            indices = self.stream[self.key[q, 0]:self.key[q, 1]+1]
            assert indices[-1] == q and np.all(self.T[indices] <= self.T[q])
            assert np.all(self.pid[indices] == self.pid[q]) and np.all(self.axis[indices] == self.axis[q])
            assert np.all(self.run_id[indices]==self.run_id[q])
            # Completeness, not merely no-future/no-cross-axis checks.
            same = np.flatnonzero((self.pid == self.pid[q]) & (self.axis == self.axis[q]))
            same = same[np.argsort(self.T[same], kind='stable')]
            end = int(np.flatnonzero(same == q)[0])
            gaps = np.flatnonzero(np.diff(self.run_id[same[:end+1]]))
            start = int(gaps[-1]+1) if len(gaps) else 0
            assert np.array_equal(indices, same[start:end+1])
        self.context_audit = dict(status='PASS', no_future=True, only_gap_resets=True,
            shared_run_ids=True,complete_history=True, oracle_queries=len(queries), max_tokens=int(np.max(self.key[:, 1]-self.key[:, 0]+1)), subjects=list(subjects))

    def ids(self, subject, windows):
        return (self.offset[subject] + np.asarray(windows)[:, None]*3 + np.arange(3)).reshape(-1)

    def contexts(self, ids, ignore=False):
        ids = np.asarray(ids, dtype=np.int64)
        if ignore:
            return ids[:, None], np.ones((len(ids), 1), bool)
        key = self.key[ids]; lengths = key[:, 1]-key[:, 0]+1
        grid = np.arange(int(lengths.max()))[None]
        positions = np.minimum(key[:, 0, None]+grid, key[:, 1, None])
        return self.stream[positions], grid < lengths[:, None]
