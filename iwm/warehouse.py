"""Whole-warehouse simulation: many forklifts and workers, nine blind intersections.

Layout (40 x 24 m):  three horizontal forklift aisles (rows A, B, C) and three
vertical pedestrian walkways (columns 1, 2, 3). Every crossing is hidden by
racking, so each of the nine intersections A1..C3 is a blind corner.

    A1 --- A2 --- A3      forklifts drive back and forth along the aisles,
    |      |      |       pausing at the ends to load;
    B1 --- B2 --- B3      workers walk up and down the walkways,
    |      |      |       pausing to pick;
    C1 --- C2 --- C3      they meet at the intersections.

Same rules as the single-corner simulation (iwm/sim.py): people only see each
other within `visibility_m` of the corner; after a reaction time the worker
stops while the forklift is moving and the driver brakes; once the forklift
has stopped, the pedestrian crosses first. What-ifs: mirror, speed limit, and
alerts that reach the worker and driver before they can see each other.

WarehouseSession runs the simulation + world model + alert engine frame by frame
and keeps per-zone status for the overview display (demo_warehouse.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .alerts import AlertEngine
from .schema import FORKLIFT, TIERS, WORKER, Alert, Track
from .sim import WhatIf
from .worldmodel import WorldModel

WIDTH, HEIGHT = 40.0, 24.0
AISLES_Y = {"A": 19.0, "B": 12.0, "C": 5.0}  # forklift aisles (3 m wide)
WALKS_X = {"1": 8.0, "2": 20.0, "3": 32.0}  # pedestrian walkways (2.5 m wide)
ZONES = {f"{r}{c}": (x, y) for r, y in AISLES_Y.items() for c, x in WALKS_X.items()}


@dataclass
class Mover:
    id: str
    type: str
    fixed: float  # y of the aisle (forklift) or x of the walkway (worker)
    pos: float  # position along the line
    direction: int  # +1 / -1
    speed: float
    lo: float
    hi: float
    cur_speed: float = 0.0
    pause_until: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.pos, self.fixed]) if self.type == FORKLIFT else np.array([self.fixed, self.pos])


@dataclass
class PairState:
    seen: float = np.inf
    alerted: float = np.inf
    in_contact: bool = False


@dataclass
class Contact:
    t: float
    forklift: str
    worker: str
    zone: str
    xy: tuple


class Warehouse:
    def __init__(self, cfg: dict, seed: int = 3, forklifts: int = 4, workers: int = 6, whatif: WhatIf | None = None):
        self.c = cfg["sim"]
        self.contact_m = cfg["world_model"]["contact_m"]
        self.dt = self.c["dt"]
        self.whatif = whatif or WhatIf()
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        r = self.rng
        rows, cols = list(AISLES_Y.values()), list(WALKS_X.values())
        # forklifts sharing an aisle drive in their own lane (two-way traffic, 1.2 m apart)
        lane = lambda i: 0.0 if forklifts <= 3 else (0.6 if (i // 3) % 2 == 0 else -0.6)  # noqa: E731
        self.forklifts = [Mover(f"fl_{i + 1}", FORKLIFT, rows[i % 3] + lane(i), r.uniform(3, WIDTH - 3),
                                int(r.choice([-1, 1])), r.uniform(*self.c["forklift_speed"]), 1.5, WIDTH - 1.5)
                          for i in range(forklifts)]
        for f in self.forklifts:
            f.cur_speed = f.speed
        self.workers = [Mover(f"wk_{i + 1}", WORKER, cols[i % 3], r.uniform(2, HEIGHT - 2), int(r.choice([-1, 1])),
                              r.uniform(*self.c["worker_speed"]), 1.0, HEIGHT - 1.0) for i in range(workers)]
        self.pairs = {(f.id, w.id): PairState() for f in self.forklifts for w in self.workers}
        self.contacts: list[Contact] = []
        self.encounters = 0
        self._near: set = set()

    # ------------------------------------------------------------- geometry
    @staticmethod
    def aisle_y(f: Mover) -> float:
        return min(AISLES_Y.values(), key=lambda y: abs(y - f.fixed))

    @staticmethod
    def zone_of(f: Mover, w: Mover) -> str:
        row = min(AISLES_Y, key=lambda k: abs(AISLES_Y[k] - f.fixed))
        col = next(k for k, x in WALKS_X.items() if x == w.fixed)
        return f"{row}{col}"

    @staticmethod
    def _ahead(m: Mover, along: float, margin: float) -> bool:
        """True while `m` has not passed coordinate `along` by more than `margin` metres."""
        return (along - m.pos) * m.direction > -margin

    def notify_alert(self, forklift_id: str, worker_id: str, tier: str) -> None:
        if self.whatif.workers_react_to_alerts and tier in ("warning", "critical"):
            st = self.pairs.get((forklift_id, worker_id))
            if st and not np.isfinite(st.alerted):
                st.alerted = self.t

    # ------------------------------------------------------------------ step
    def step(self) -> list[Track]:
        c, dt, t = self.c, self.dt, self.t
        vis = 5.0 if self.whatif.mirror else c["visibility_m"]
        brake = {f.id: False for f in self.forklifts}
        wait = {w.id: False for w in self.workers}

        for f in self.forklifts:
            for w in self.workers:
                st = self.pairs[(f.id, w.id)]
                ix, iy = w.fixed, self.aisle_y(f)  # their intersection
                f_ahead = self._ahead(f, ix, 3.0)
                w_cleared = (w.pos - iy) * w.direction > 2.0
                active = f_ahead and not w_cleared
                if not active:  # encounter over: forget it
                    st.seen, st.alerted = np.inf, np.inf
                    continue
                f_d, w_d = abs(f.pos - ix), abs(w.pos - iy)
                if f_d < 8 and w_d < 6 and (f.id, w.id) not in self._near:
                    self._near.add((f.id, w.id))
                    self.encounters += 1
                elif f_d > 10 or w_d > 8:
                    self._near.discard((f.id, w.id))
                if f_d < vis and w_d < vis:  # blind corner: only see each other when both are close
                    st.seen = min(st.seen, t)
                if t >= min(st.seen, st.alerted) + c["reaction_s"]:
                    if f_d < 10:
                        brake[f.id] = True
                    if w_d < 6 and (iy - w.pos) * w.direction > 0 and f.cur_speed > 0.2:
                        wait[w.id] = True

        for f in self.forklifts:
            if t < f.pause_until:
                f.cur_speed = 0.0
                continue
            target = f.speed
            if self.whatif.speed_limit and min(abs(f.pos - x) for x in WALKS_X.values()) < 5.0:
                target = min(target, 1.0)
            if brake[f.id]:
                f.cur_speed = max(0.0, f.cur_speed - 2.5 * dt)
            else:
                f.cur_speed = target if f.cur_speed >= target else min(target, f.cur_speed + 1.5 * dt)
            self._move(f, f.cur_speed, pause=(1.0, 3.0))

        for w in self.workers:
            if t < w.pause_until or wait[w.id]:
                w.cur_speed = 0.0
                continue
            w.cur_speed = w.speed
            self._move(w, w.speed, pause=(1.0, 2.5))
            # occasionally stop to pick from a rack, never inside an aisle
            if self.rng.random() < 0.004 and min(abs(w.pos - y) for y in AISLES_Y.values()) > 4:
                w.pause_until = t + self.rng.uniform(1.0, 3.0)

        for f in self.forklifts:  # ground truth contacts
            for w in self.workers:
                st = self.pairs[(f.id, w.id)]
                d = float(np.linalg.norm(f.xy - w.xy))
                if d < self.contact_m and f.cur_speed > 0.3 and not st.in_contact:  # a stopped forklift isn't a hit
                    st.in_contact = True
                    self.contacts.append(Contact(round(t, 1), f.id, w.id, self.zone_of(f, w), tuple(w.xy)))
                elif d > 4.0:
                    st.in_contact = False

        noise = c["position_noise_m"]
        obs = [Track(round(t, 2), m.id, m.type, *(m.xy + self.rng.normal(0, noise, 2)))
               for m in self.forklifts + self.workers]
        self.t += dt
        return obs

    def _move(self, m: Mover, speed: float, pause: tuple) -> None:
        m.pos += m.direction * speed * self.dt
        if m.pos >= m.hi or m.pos <= m.lo:  # end of the line: pause, then turn around
            m.pos = min(max(m.pos, m.lo), m.hi)
            m.direction *= -1
            m.pause_until = self.t + self.rng.uniform(*pause)


# --------------------------------------------------------------------- session
@dataclass
class ZoneStatus:
    zone: str
    tier: str = "none"
    p: float = 0.0
    ttc: float = float("inf")
    pair: tuple | None = None


@dataclass
class WFrame:
    t: float
    tracks: list
    risks: list
    zones: dict
    new_alerts: list[Alert] = field(default_factory=list)
    new_contacts: list[Contact] = field(default_factory=list)


class WarehouseSession:
    def __init__(self, cfg: dict, seed: int = 3, whatif: WhatIf | None = None, forklifts: int = 4, workers: int = 6):
        self.cfg = cfg
        self.wh = Warehouse(cfg, seed, forklifts, workers, whatif)
        self.wm, self.engine = WorldModel(cfg), AlertEngine(cfg)
        self.alerts: list[Alert] = []
        self.alert_zone: dict[int, str] = {}
        self.reasons: dict[tuple, list] = {}
        self.frame: WFrame | None = None

    def pair_zone(self, forklift_id: str, worker_id: str) -> str:
        f = next(m for m in self.wh.forklifts if m.id == forklift_id)
        w = next(m for m in self.wh.workers if m.id == worker_id)
        return self.wh.zone_of(f, w)

    def step(self) -> WFrame:
        t = self.wh.t
        n_contacts = len(self.wh.contacts)
        obs = self.wh.step()
        tracks = self.wm.update(t, obs)
        risks = self.wm.risk(tracks)
        new = self.engine.update(t, risks)
        for a in new:
            pair = next(r for r in risks if [r.forklift, r.worker] == a.pair)
            reasons = self.wm.explain(pair)
            a.cause = reasons[0][0] if reasons else ""
            self.reasons[tuple(a.pair)] = reasons
            self.wh.notify_alert(a.pair[0], a.pair[1], a.tier)
            self.alert_zone[id(a)] = self.pair_zone(*a.pair)
            self.alerts.append(a)
        zones = {z: ZoneStatus(z) for z in ZONES}
        for r in risks:
            tier = self.engine.tier_of(r.forklift, r.worker)
            z = zones[self.pair_zone(r.forklift, r.worker)]
            key = (TIERS.index(tier), r.p_collision)
            if key > (TIERS.index(z.tier), z.p):
                z.tier, z.p, z.ttc, z.pair = tier, r.p_collision, r.ttc_s, (r.forklift, r.worker)
        self.frame = WFrame(t, tracks, risks, zones, new, self.wh.contacts[n_contacts:])
        return self.frame

    def hottest(self) -> ZoneStatus | None:
        """The zone with the highest current tier (then risk), or None if all are clear."""
        if not self.frame:
            return None
        z = max(self.frame.zones.values(), key=lambda z: (TIERS.index(z.tier), z.p))
        return z if z.tier != "none" else None

    def stats(self) -> dict:
        tiers = [a.tier for a in self.alerts]
        return {"t": round(self.wh.t, 1), "encounters": self.wh.encounters, "contacts": len(self.wh.contacts),
                "advisory": tiers.count("advisory"), "warning": tiers.count("warning"),
                "critical": tiers.count("critical")}
