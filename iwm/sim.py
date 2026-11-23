"""2D top-down warehouse simulation with a blind corner.

It produces the same Track records as the camera pipeline, plus ground truth
(the exact time of any worker-forklift contact). We use it because real
near-misses are too rare and too dangerous to stage.

Layout: a horizontal forklift aisle and a vertical walkway meet at a corner
hidden by racking. The worker and the driver only see each other when both are
within `visibility_m` of the corner (the "blind corner"). After a human
reaction time the worker stops while the forklift is moving, and the driver
brakes (2.5 m/s^2); once the forklift has stopped, the pedestrian crosses first
and the forklift waits until the walkway is clear. What-if fixes:
  - mirror:      convex mirror, both see each other from 5 m
  - speed_limit: the forklift slows to 1 m/s near the corner
  - workers_react_to_alerts: our alert reaches the worker's phone and the
    driver's beeper, so both react before they can see each other

Run:  python -m iwm.sim --scenario blind_corner --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from . import load_config
from .schema import FORKLIFT, WORKER, Track

SCENARIOS = ["blind_corner", "crossing", "parallel", "random"]


@dataclass
class Agent:
    id: str
    type: str
    start: np.ndarray
    end: np.ndarray
    speed: float
    t0: float  # time the agent starts moving
    pause_at: float = -1.0  # optional random pause (walkers)
    pause_len: float = 0.0
    pos: np.ndarray = field(default=None)
    crosses: bool = False  # walker's path goes through the corner
    done: bool = False

    def __post_init__(self):
        self.pos = self.start.astype(float).copy()
        self.cur_speed = self.speed
        d = self.end - self.start
        self.dir = d / np.linalg.norm(d)


@dataclass
class WhatIf:
    mirror: bool = False
    speed_limit: bool = False
    workers_react_to_alerts: bool = False


class Simulation:
    def __init__(self, cfg: dict, scenario: str = "random", seed: int | None = None,
                 whatif: WhatIf | None = None):
        self.c = cfg["sim"]
        self.contact_m = cfg["world_model"]["contact_m"]
        self.rng = np.random.default_rng(seed)
        self.whatif = whatif or WhatIf()
        self.dt = self.c["dt"]
        self.corner = np.array(self.c["corner"], float)
        self.t = 0.0
        self.contact_t: float | None = None
        self.min_dist = np.inf
        self._alerted: dict[str, float] = {}  # worker id -> time alert received
        self._seen: dict[str, float] = {}  # agent id -> time it first saw the other
        if scenario == "random":
            scenario = self.rng.choice(["crossing", "crossing", "crossing", "parallel"])
        self.scenario = scenario
        self.agents = self._build(scenario)
        self.duration = max(a.t0 + np.linalg.norm(a.end - a.start) / a.speed for a in self.agents) + 8.0

    # ------------------------------------------------------------------ setup
    def _build(self, scenario: str) -> list[Agent]:
        c, r = self.c, self.rng
        W, H = c["width_m"], c["height_m"]
        cx, cy = self.corner
        vf = r.uniform(*c["forklift_speed"])
        vw = r.uniform(*c["worker_speed"])

        if scenario == "blind_corner":  # the fixed demo case: arrive together
            vf, vw, offset = 2.5, 1.3, 0.3
            f_start, f_end = np.array([0.0, cy]), np.array([W, cy])
            w_start, w_end = np.array([cx, 0.0]), np.array([cx, H])
        else:
            flip_f, flip_w = r.random() < 0.5, r.random() < 0.5
            f_start, f_end = np.array([0.0, cy]), np.array([W, cy])
            if flip_f:
                f_start, f_end = f_end, f_start
            if scenario == "crossing":
                w_start, w_end = np.array([cx, 0.0]), np.array([cx, H])
            else:  # parallel walkway 2.2-3.5 m from the forklift aisle
                y = cy + r.uniform(2.2, 3.5) * (1 if r.random() < 0.5 else -1)
                w_start, w_end = np.array([0.0, y]), np.array([W, y])
            if flip_w:
                w_start, w_end = w_end, w_start
            offset = r.uniform(-3.0, 3.0)  # worker arrival minus forklift arrival (s)

        # Choose start times so both reach the corner `offset` seconds apart.
        f_arrive = np.linalg.norm(self.corner - f_start) / vf
        w_travel = np.linalg.norm(self.corner - w_start) / vw if scenario != "parallel" else 0.0
        w_t0 = f_arrive + offset - w_travel
        f_t0 = 0.0
        if w_t0 < 0:
            f_t0, w_t0 = -w_t0, 0.0

        worker = Agent("wk_01", WORKER, w_start, w_end, vw, w_t0, crosses=scenario != "parallel")
        if scenario != "blind_corner" and r.random() < 0.25:  # walkers sometimes pause
            worker.pause_at = w_t0 + r.uniform(0.5, 4.0)
            worker.pause_len = r.uniform(0.5, 2.0)
        forklift = Agent("fl_01", FORKLIFT, f_start, f_end, vf, f_t0)
        return [forklift, worker]

    # ------------------------------------------------------------------- step
    def notify_alert(self, worker_id: str, tier: str) -> None:
        """Phone buzz + forklift beep (used only when workers_react_to_alerts)."""
        if self.whatif.workers_react_to_alerts and tier in ("warning", "critical"):
            for a in self.agents:
                self._alerted.setdefault(a.id, self.t)

    def _forklift_ahead(self, f: Agent) -> bool:
        """True while the forklift has not yet passed the corner by 3 m."""
        return float(np.dot(self.corner - f.pos, f.dir)) > -3.0

    def step(self) -> list[Track]:
        c, dt = self.c, self.dt
        f = next(a for a in self.agents if a.type == FORKLIFT)
        w = next(a for a in self.agents if a.type == WORKER)
        vis = 5.0 if self.whatif.mirror else c["visibility_m"]
        near = lambda a: np.linalg.norm(a.pos - self.corner) < vis  # noqa: E731
        walker_cleared = w.done or float(np.dot(w.pos - self.corner, w.dir)) > 2.0
        if w.crosses and near(w) and near(f) and self._forklift_ahead(f) and not walker_cleared:
            self._seen.setdefault(w.id, self.t)
            self._seen.setdefault(f.id, self.t)

        def triggered(a):
            return self.t >= min(self._seen.get(a.id, np.inf), self._alerted.get(a.id, np.inf)) + c["reaction_s"]

        # Forklift: brake once the driver reacts, until the walker is clear.
        if not f.done and self.t >= f.t0:
            target = f.speed
            if self.whatif.speed_limit and np.linalg.norm(f.pos - self.corner) < 5.0:
                target = min(target, 1.0)
            if triggered(f) and not walker_cleared and self._forklift_ahead(f):
                f.cur_speed = max(0.0, f.cur_speed - 2.5 * dt)
            else:
                f.cur_speed = min(target, f.cur_speed + 1.5 * dt) if f.cur_speed < target else target
            self._move(f, f.cur_speed)

        # Worker: stops while a forklift it knows about is still moving towards the corner.
        if not w.done and self.t >= w.t0:
            waiting = triggered(w) and self._forklift_ahead(f) and f.cur_speed > 0.2 and not walker_cleared
            pausing = w.pause_at <= self.t < w.pause_at + w.pause_len
            if not (waiting or pausing):
                self._move(w, w.speed)

        # Ground truth: contact = closer than contact_m.
        d = float(np.linalg.norm(w.pos - f.pos))
        if self.t >= max(w.t0, f.t0):
            self.min_dist = min(self.min_dist, d)
            if d < self.contact_m and self.contact_t is None:
                self.contact_t = self.t

        obs = []
        for a in self.agents:
            noisy = a.pos + self.rng.normal(0, c["position_noise_m"], 2)
            obs.append(Track(round(self.t, 2), a.id, a.type, float(noisy[0]), float(noisy[1])))
        self.t += self.dt
        return obs

    def _move(self, a: Agent, speed: float) -> None:
        a.pos = a.pos + a.dir * speed * self.dt
        if np.dot(a.end - a.pos, a.dir) <= 0:
            a.pos, a.done = a.end.copy(), True

    def run(self) -> list[list[Track]]:
        frames = []
        while self.t < self.duration:
            frames.append(self.step())
        return frames


def _plot(frames, sim, path="runs/sim.png"):
    import os

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for aid, colour in (("fl_01", "tab:orange"), ("wk_01", "tab:blue")):
        xs = [t.x for fr in frames for t in fr if t.id == aid]
        ys = [t.y for fr in frames for t in fr if t.id == aid]
        ax.plot(xs, ys, ".", ms=2, color=colour, label=aid)
    ax.add_patch(plt.Rectangle((sim.corner[0] - 3, sim.corner[1] + 0.8), 2.2, 2.2, color="grey", alpha=0.4))
    ax.set(xlim=(0, sim.c["width_m"]), ylim=(0, sim.c["height_m"]), aspect="equal",
           title=f"{sim.scenario}: contact at {sim.contact_t}")
    ax.legend()
    fig.savefig(path, dpi=120)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="blind_corner", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    s = Simulation(load_config(), args.scenario, args.seed)
    fr = s.run()
    print(f"{s.scenario}: {len(fr)} frames, contact_t={None if s.contact_t is None else round(s.contact_t, 1)}, min distance={s.min_dist:.2f} m")
    if args.plot:
        _plot(fr, s)
