"""Full same-patient/axis/phase history since the last >30min gap."""
import numpy as np


class Pool:
    def __init__(self, data, subjects):
        self.axes = 3
        self.offset, self.count, self.report = {}, {}, {}
        xs, fs, ys, times, patients, axes, streams, keys = [], [], [], [], [], [], [], []
        base = stream_offset = 0
        for patient, subject in enumerate(subjects):
            d = data[subject]; n = len(d['t'])
            self.offset[subject] = base; self.count[subject] = n
            xs.append(d['sig'].reshape(-1, 1, 1000).astype(np.float16))
            fs.append(d['fid'].reshape(-1, 42).astype(np.float32))
            ys.append(np.repeat(d.get('map', d.get('y')), 3))
            times.append(np.repeat(d['availability_s'], 3)); patients.append(np.full(n*3, patient)); axes.append(np.tile(np.arange(3), n))
            order = np.argsort(d['t'], kind='stable'); tt = d['t'][order]
            bins = np.floor((tt - tt[0]) / 120).astype(np.int64)
            starts = np.r_[0, np.flatnonzero(np.diff(bins)) + 1]
            phase = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
            key = np.empty((n*3, 2), np.int64); max_tokens = 0
            for rank in np.unique(phase):
                wi = order[phase == rank]; t = d['availability_s'][wi]
                cuts = np.r_[0, np.flatnonzero(np.diff(t) > 1800) + 1]
                lower = np.repeat(cuts, np.diff(np.r_[cuts, len(wi)]))
                max_tokens = max(max_tokens, int(np.max(np.arange(len(wi)) - lower + 1)))
                for axis in range(3):
                    ix = base + wi*3 + axis; streams.append(ix)
                    key[wi*3 + axis] = np.c_[lower + stream_offset, np.arange(len(wi)) + stream_offset]
                    stream_offset += len(ix)
            keys.append(key)
            self.report[subject] = dict(raw_windows=n, axis_samples=n*3, max_history_tokens=max_tokens,
                history_policy='all_causal_within_run', gap_seconds=1800, time_cap=None, token_cap=None)
            base += n*3
        self.X = np.concatenate(xs); self.F = np.concatenate(fs); self.Y = np.concatenate(ys).astype(np.float32)
        self.T = np.concatenate(times); self.pid = np.concatenate(patients); self.axis = np.concatenate(axes)
        self.stream = np.concatenate(streams); self.key = np.concatenate(keys)
        assert np.array_equal(np.sort(self.stream), np.arange(base))
        queries = np.unique(np.r_[np.random.default_rng(0).integers(base, size=min(base, 500)), 0, base-1])
        for q in queries:
            indices = self.stream[self.key[q, 0]:self.key[q, 1]+1]
            assert indices[-1] == q and np.all(self.T[indices] <= self.T[q])
            assert np.all(self.pid[indices] == self.pid[q]) and np.all(self.axis[indices] == self.axis[q])
            assert np.all(np.diff(self.T[indices]) <= 1800)
        self.context_audit = dict(status='PASS', no_future=True, only_gap_resets=True,
            oracle_queries=len(queries), max_tokens=int(np.max(self.key[:, 1]-self.key[:, 0]+1)), subjects=list(subjects))

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
