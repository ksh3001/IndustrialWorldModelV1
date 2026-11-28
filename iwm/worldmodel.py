"""World model: Kalman state estimation + Monte Carlo futures.

1. Each track is smoothed with a constant-velocity Kalman filter, which also
   estimates its velocity from noisy positions.
2. For every worker-forklift pair, we roll `rollouts` possible futures forward
   `horizon_s` seconds, with noise on speed and heading and a chance that the
   walker stops.
3. p_collision = share of futures where the pair comes within `contact_m`;
   ttc_s = median time of first contact among those futures.
4. explain() re-runs the futures with one factor changed (a counterfactual)
   and reports which change lowers risk most: the "why".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .schema import FORKLIFT, WORKER, Track


class KalmanCV:
    """Constant-velocity Kalman filter, state = [x, y, vx, vy]."""

    def __init__(self, x: float, y: float, t: float, meas_noise=0.1, accel_noise=1.5):
        self.s = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([meas_noise**2, meas_noise**2, 4.0, 4.0])
        self.r, self.q, self.t = meas_noise, accel_noise, t

    def step(self, x: float, y: float, t: float) -> np.ndarray:
        dt = max(t - self.t, 1e-3)
        self.t = t
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + G @ G.T * self.q**2
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = Hm @ self.P @ Hm.T + np.eye(2) * self.r**2
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.s = self.s + K @ (np.array([x, y]) - Hm @ self.s)
        self.P = (np.eye(4) - K @ Hm) @ self.P
        return self.s


@dataclass
class PairRisk:
    forklift: str
    worker: str
    p_collision: float
    ttc_s: float
    distance_m: float
    paths: dict = field(default_factory=dict)  # id -> (steps, 2) mean predicted path


class WorldModel:
    def __init__(self, cfg: dict):
        c = cfg["world_model"]
        self.c = c
        self.dt = 0.1
        self.steps = int(round(c["horizon_s"] / self.dt))
        self.times = np.arange(1, self.steps + 1) * self.dt
        self.rng = np.random.default_rng(c.get("seed"))
        self.filters: dict[str, KalmanCV] = {}
        self.types: dict[str, str] = {}
        self.last_seen: dict[str, float] = {}
        self.tracks: dict[str, Track] = {}
        # Hybrid world model: part of the futures come from the learned motion model (iwm/learned.py)
        self.mode = c.get("mode", "kalman")
        self.learned_share = float(c.get("learned_share", 0.5)) if self.mode != "kalman" else 0.0
        if self.mode == "learned":
            self.learned_share = 1.0
        self.learned = None
        if self.learned_share > 0:
            from .learned import load_cached

            self.learned = load_cached(c.get("learned_model", "models/motion.npz"))
            if self.learned is None:
                self.learned_share = 0.0  # no trained model yet: Kalman only
        self.history: dict[str, deque] = {}
        self.stats = {"learned": 0, "kalman": 0}  # how many tracks used each, for the UI

    # --------------------------------------------------------------- tracking
    def update(self, t: float, detections) -> list[Track]:
        """Feed raw detections (Track or dict with id/type/x/y); returns smoothed tracks."""
        for d in detections:
            d = d.to_dict() if isinstance(d, Track) else d
            k = self.filters.get(d["id"])
            if k is None:
                k = self.filters[d["id"]] = KalmanCV(d["x"], d["y"], t, self.c.get("meas_noise", 0.1))
                s = k.s
            else:
                s = k.step(d["x"], d["y"], t)
            self.types[d["id"]] = d["type"]
            self.last_seen[d["id"]] = t
            self.tracks[d["id"]] = Track(t, d["id"], d["type"], *map(float, s))
            if self.learned is not None:
                self.history.setdefault(d["id"], deque(maxlen=200)).append((t, float(s[0]), float(s[1])))
        for tid in [i for i, ts in self.last_seen.items() if t - ts > 1.0]:  # forget lost tracks
            for store in (self.filters, self.types, self.last_seen, self.tracks, self.history):
                store.pop(tid, None)
        return list(self.tracks.values())

    # ---------------------------------------------------------------- futures
    def _history_grid(self, tid: str) -> np.ndarray | None:
        """Last `history_s` seconds of smoothed positions on a 0.1 s grid, or None if too short."""
        h = self.history.get(tid)
        need = self.learned.history_s
        if not h or h[-1][0] - h[0][0] < need - 0.05:
            return None
        a = np.array(h)
        grid = a[-1, 0] - np.arange(self.learned.H - 1, -1, -1) * 0.1
        return np.stack([np.interp(grid, a[:, 0], a[:, 1]), np.interp(grid, a[:, 0], a[:, 2])], axis=1)

    def _futures(self, tr: Track, speed_scale: float = 1.0, frozen: bool = False) -> np.ndarray:
        """Sample possible future positions, shape (rollouts, steps, 2).

        Hybrid mode: `learned_share` of the futures come from the learned motion model, the rest
        from the Kalman rules below. Tracks with less than `history_s` of history use Kalman only.
        """
        n_total = self.c["rollouts"]
        n_learned = int(round(n_total * self.learned_share)) if not frozen and speed_scale == 1.0 else 0
        hist = self._history_grid(tr.id) if n_learned else None
        if hist is None:
            n_learned = 0
        k = self._kalman_futures(tr, n_total - n_learned, speed_scale, frozen)
        if not n_learned:
            self.stats["kalman"] += 1
            return k
        self.stats["learned"] += 1
        lf = self.learned.sample(hist, tr.vx, tr.vy, tr.type == FORKLIFT, n_learned, self.steps, self.rng)
        return np.concatenate([k, lf], axis=0) if len(k) else lf

    def _kalman_futures(self, tr: Track, n: int, speed_scale: float = 1.0, frozen: bool = False) -> np.ndarray:
        """The original rule-based futures: constant velocity plus noise and walker stops."""
        c = self.c
        if n <= 0:
            return np.zeros((0, self.steps, 2))
        if frozen:
            return np.broadcast_to([tr.x, tr.y], (n, self.steps, 2)).copy()
        speed = np.hypot(tr.vx, tr.vy) * speed_scale
        heading = np.arctan2(tr.vy, tr.vx)
        hn = c["heading_noise_worker"] if tr.type == WORKER else c["heading_noise_forklift"]
        sp = np.clip(speed * (1 + self.rng.normal(0, c["speed_noise"], n)), 0, None)
        hd = heading + self.rng.normal(0, hn, n)
        v = np.stack([sp * np.cos(hd), sp * np.sin(hd)], axis=1)  # (n, 2)
        t_eff = np.broadcast_to(self.times, (n, self.steps)).copy()
        if tr.type == WORKER:  # some futures: the walker stops at a random moment
            stops = self.rng.random(n) < c["walker_stop_prob"]
            stop_t = np.where(stops, self.rng.uniform(0, c["horizon_s"], n), np.inf)
            t_eff = np.minimum(t_eff, stop_t[:, None])
        return np.array([tr.x, tr.y]) + v[:, None, :] * t_eff[..., None]

    def _collide(self, f_fut: np.ndarray, w_fut: np.ndarray) -> tuple[float, float]:
        dist = np.linalg.norm(f_fut - w_fut, axis=2)  # (n, steps)
        hit = dist < self.c["contact_m"]
        any_hit = hit.any(axis=1)
        p = float(any_hit.mean())
        if not any_hit.any():
            return p, float("inf")
        first = hit[any_hit].argmax(axis=1)
        return p, float(np.median(self.times[first]))

    def risk(self, tracks: list[Track] | None = None) -> list[PairRisk]:
        tracks = tracks if tracks is not None else list(self.tracks.values())
        forklifts = [t for t in tracks if t.type == FORKLIFT]
        workers = [t for t in tracks if t.type == WORKER]
        out = []
        futures = {t.id: self._futures(t) for t in forklifts + workers} if forklifts and workers else {}
        reach = lambda tr: np.hypot(tr.vx, tr.vy) * (1 + 3 * self.c["speed_noise"])  # noqa: E731
        for f in forklifts:
            for w in workers:
                dist_now = float(np.hypot(f.x - w.x, f.y - w.y))
                if dist_now > (reach(f) + reach(w)) * self.c["horizon_s"] + self.c["contact_m"]:
                    out.append(PairRisk(f.id, w.id, 0.0, float("inf"), dist_now))  # too far apart to meet
                    continue
                if dist_now < self.c["contact_m"]:
                    p, ttc = 1.0, 0.0
                else:
                    p, ttc = self._collide(futures[f.id], futures[w.id])
                out.append(PairRisk(f.id, w.id, p, ttc, dist_now,
                                    {f.id: futures[f.id].mean(0), w.id: futures[w.id].mean(0)}))
        return out

    # ---------------------------------------------------------------- explain
    def explain(self, pair: PairRisk) -> list[tuple[str, float]]:
        """Counterfactuals: which single change lowers this pair's risk most?"""
        f, w = self.tracks.get(pair.forklift), self.tracks.get(pair.worker)
        if f is None or w is None:
            return []
        base = self._collide(self._futures(f), self._futures(w))[0]
        options = {
            f"forklift slows to 1 m/s (now {np.hypot(f.vx, f.vy):.1f} m/s)":
                (self._futures(f, speed_scale=min(1.0, 1.0 / max(np.hypot(f.vx, f.vy), 1e-3))), self._futures(w)),
            "worker stops now": (self._futures(f), self._futures(w, frozen=True)),
            "forklift stops now": (self._futures(f, frozen=True), self._futures(w)),
        }
        drops = [(name, base - self._collide(ff, wf)[0]) for name, (ff, wf) in options.items()]
        return sorted(drops, key=lambda d: -d[1])


def detection_only(tracks: list[Track], threshold_m: float) -> list[tuple[str, str]]:
    """Baseline 'today's video AI': alert only when a pair is already close."""
    fl = [t for t in tracks if t.type == FORKLIFT]
    wk = [t for t in tracks if t.type == WORKER]
    return [(f.id, w.id) for f in fl for w in wk if np.hypot(f.x - w.x, f.y - w.y) < threshold_m]
