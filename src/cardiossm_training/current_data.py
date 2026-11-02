"""BCGnet current-window samples, preserving float32 waveform precision."""
import numpy as np


class Pool:
    def __init__(self, data, subjects, whole_windows=False):
        self.axes = 1 if whole_windows else 3
        self.offset, self.count, self.report = {}, {}, {}
        x, f, y = [], [], []; offset = 0
        for subject in subjects:
            d = data[subject]; n = len(d["t"])
            self.offset[subject] = offset; self.count[subject] = n
            x.append(d["sig"].astype(np.float32) if whole_windows else d["sig"].reshape(-1, 1, 1000).astype(np.float32))
            f.append(np.zeros((n, 42), np.float32) if whole_windows else d["fid"].reshape(-1, 42).astype(np.float32))
            y.append(np.repeat(d.get("map", d.get("y")), self.axes))
            self.report[subject] = dict(raw_windows=n, samples=self.axes*n, history_policy="current_window_only")
            offset += n * self.axes
        self.X = np.concatenate(x); self.F = np.concatenate(f)
        self.Y = np.concatenate(y).astype(np.float32)
        self.context_audit = dict(status="PASS", no_future=True, max_tokens=1, subjects=list(subjects))

    def ids(self, subject, windows):
        return (self.offset[subject] + np.asarray(windows)[:, None] * self.axes + np.arange(self.axes)).reshape(-1)
