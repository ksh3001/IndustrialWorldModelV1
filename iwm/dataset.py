"""Training data for the learned motion model.

Sources (all become the same thing: per-object tracks of raw floor positions over time)
  - simulation: blind-corner runs (iwm/sim.py) and whole-warehouse shifts (iwm/warehouse.py)
  - real video: track logs written by run_camera.py --log-tracks or the UI Real video page
    (logs/tracks/*.jsonl, one detection per line: {"t", "id", "type", "x", "y"})

Each track is Kalman-smoothed exactly as the live world model does, resampled to 10 Hz, split at
gaps / ID switches, then cut into windows: history (input) -> real future positions (target).
The future positions come straight from the data, so no labelling is needed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .learned import DT, Windows, cv_path, features, rot
from .schema import FORKLIFT
from .worldmodel import KalmanCV

Track = dict  # {"type": str, "t": array, "x": array, "y": array}


# ------------------------------------------------------------------------ sources
def read_track_log(path: str | Path) -> dict[str, Track]:
    """Read a JSONL track log; ids are prefixed with the file name so clips don't collide."""
    rows = defaultdict(list)
    stem = Path(path).stem
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows[f"{stem}/{d['id']}"].append((d["t"], d["x"], d["y"], d["type"]))
    out = {}
    for tid, r in rows.items():
        r.sort()
        a = np.array([x[:3] for x in r], float)
        out[tid] = {"type": r[0][3], "t": a[:, 0], "x": a[:, 1], "y": a[:, 2]}
    return out


def sim_tracks(cfg: dict, warehouse_runs: int = 12, corner_runs: int = 300, seconds: float = 120.0,
               seed0: int = 100_000) -> dict[str, Track]:
    """Noisy simulated detections. Seeds start at 100,000 so they never overlap iwm.evaluate (0..999)."""
    from .sim import Simulation, WhatIf
    from .warehouse import Warehouse

    rows = defaultdict(list)
    rng = np.random.default_rng(seed0)
    for i in range(warehouse_runs):
        w = WhatIf(mirror=bool(rng.random() < 0.3), speed_limit=bool(rng.random() < 0.3))
        wh = Warehouse(cfg, seed=seed0 + i, whatif=w)
        while wh.t < seconds:
            for o in wh.step():
                rows[(f"wh{i}/{o.id}", o.type)].append((o.t, o.x, o.y))
    for i in range(corner_runs):
        s = Simulation(cfg, "random", seed0 + 10_000 + i)
        while s.t < s.duration:
            for o in s.step():
                rows[(f"bc{i}/{o.id}", o.type)].append((o.t, o.x, o.y))
    out = {}
    for (tid, typ), r in rows.items():
        a = np.array(r, float)
        out[tid] = {"type": typ, "t": a[:, 0], "x": a[:, 1], "y": a[:, 2]}
    return out


# ------------------------------------------------------------------------- windows
def _segments(tr: Track, max_gap: float = 0.6, max_speed: float = 8.0):
    """Split a track where detections stop for a while or jump impossibly far (an ID switch)."""
    t, x, y = tr["t"], tr["x"], tr["y"]
    if len(t) < 2:
        return
    gap = np.diff(t)
    jump = np.hypot(np.diff(x), np.diff(y)) / np.maximum(gap, 1e-3)
    cuts = np.where((gap > max_gap) | (jump > max_speed))[0] + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if b - a >= 5:
            yield t[a:b], x[a:b], y[a:b]


def smooth_and_resample(t, x, y, meas_noise: float = 0.1):
    """Kalman-smooth (online, like the live system) and resample to a DT grid."""
    k = KalmanCV(x[0], y[0], t[0], meas_noise)
    states = [k.s.copy()]
    for ti, xi, yi in zip(t[1:], x[1:], y[1:]):
        states.append(k.step(xi, yi, ti).copy())
    s = np.array(states)
    grid = np.arange(t[0], t[-1] + 1e-9, DT)
    sm = np.stack([np.interp(grid, t, s[:, i]) for i in range(4)], axis=1)  # x, y, vx, vy
    raw = np.stack([np.interp(grid, t, x), np.interp(grid, t, y)], axis=1)
    return grid, sm, raw


def build_windows(tracks: dict[str, Track], history_s: float = 2.0, horizon_s: float = 5.0,
                  stride_s: float = 0.5, min_speed: float = 0.0) -> Windows:
    """Cut every track into (history -> future residual) training windows."""
    H, S, step = int(round(history_s / DT)), int(round(horizon_s / DT)), max(1, int(round(stride_s / DT)))
    X, Y, FK, G = [], [], [], []
    for gi, (tid, tr) in enumerate(tracks.items()):
        fk = tr["type"] == FORKLIFT
        for t, x, y in _segments(tr):
            _, sm, raw = smooth_and_resample(t, x, y)
            for i in range(H - 1, len(sm) - S, step):
                hist = sm[i - H + 1:i + 1, :2]
                vx, vy = sm[i, 2], sm[i, 3]
                future = raw[i + 1:i + 1 + S]
                if np.hypot(vx, vy) < min_speed and np.hypot(*(future[-1] - hist[-1])) < 0.2:
                    continue  # skip objects standing still the whole time
                f, th = features(hist, vx, vy, fk)
                res = (future - cv_path(hist[-1], vx, vy, S)) @ rot(th).T
                X.append(f)
                Y.append(res.ravel())
                FK.append(fk)
                G.append(gi)
    if not X:
        n_in, n_out = 2 * H + 2, 2 * S
        return Windows(np.zeros((0, n_in)), np.zeros((0, n_out)), np.zeros(0, bool), np.zeros(0, int))
    return Windows(np.array(X), np.array(Y), np.array(FK), np.array(G))


def split(w: Windows, val_frac: float = 0.2, seed: int = 0) -> tuple[Windows, Windows]:
    """Split by source track (not by window), so the test never sees a track it trained on."""
    groups = np.unique(w.group)
    rng = np.random.default_rng(seed)
    val_groups = set(rng.choice(groups, max(1, int(len(groups) * val_frac)), replace=False)) if len(groups) > 1 else set()
    m = np.array([g in val_groups for g in w.group])
    pick = lambda mask: Windows(w.X[mask], w.Y[mask], w.is_forklift[mask], w.group[mask])  # noqa: E731
    return pick(~m), pick(m)
