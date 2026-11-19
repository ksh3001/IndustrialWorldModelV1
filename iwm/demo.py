"""One demo session: simulation + world model + alerts, one frame at a time.

Shared by the pygame live demo (demo_sim.py) and the Streamlit dashboard
(app.py), so both show exactly the same behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .alerts import AlertEngine
from .schema import Alert
from .sim import Simulation, WhatIf
from .worldmodel import PairRisk, WorldModel, detection_only


@dataclass
class Frame:
    t: float
    tracks: list
    risks: list[PairRisk]
    tier: str
    new_alerts: list[Alert]
    detection_only: bool
    contact: bool
    reasons: list = field(default_factory=list)


class DemoSession:
    def __init__(self, cfg: dict, scenario: str = "blind_corner", seed: int = 1, whatif: WhatIf | None = None):
        self.cfg = cfg
        self.sim = Simulation(cfg, scenario, seed, whatif)
        self.wm = WorldModel(cfg)
        self.engine = AlertEngine(cfg)
        self.history: list[Frame] = []
        self.alerts: list[Alert] = []
        self.first_detection: float | None = None
        self.last_reasons: list = []

    @property
    def done(self) -> bool:
        return self.sim.t >= self.sim.duration

    def step(self) -> Frame:
        t = self.sim.t
        obs = self.sim.step()
        tracks = self.wm.update(t, obs)
        risks = self.wm.risk(tracks)
        new = self.engine.update(t, risks)
        reasons = []
        for a in new:
            pair = next(r for r in risks if [r.forklift, r.worker] == a.pair)
            reasons = self.wm.explain(pair)
            self.last_reasons = reasons or self.last_reasons
            a.cause = reasons[0][0] if reasons else ""
            self.sim.notify_alert(a.pair[1], a.tier)
            self.alerts.append(a)
        tier = max((self.engine.tier_of(r.forklift, r.worker) for r in risks),
                   key=["none", "advisory", "warning", "critical"].index, default="none")
        det = bool(detection_only(tracks, self.cfg["alerts"]["detection_only_m"]))
        if det and self.first_detection is None:
            self.first_detection = t
        fr = Frame(t, tracks, risks, tier, new, det, self.sim.contact_t is not None, reasons)
        self.history.append(fr)
        return fr

    def run(self) -> list[Frame]:
        while not self.done:
            self.step()
        return self.history

    def summary(self) -> dict:
        contact = self.sim.contact_t
        first_alert = self.alerts[0].t if self.alerts else None
        first_warn = next((a.t for a in self.alerts if a.tier in ("warning", "critical")), None)
        lead = lambda x: round(contact - x, 1) if contact is not None and x is not None and x < contact else None  # noqa: E731
        return {
            "scenario": self.sim.scenario,
            "contact_s": None if contact is None else round(contact, 1),
            "closest_m": round(self.sim.min_dist, 2),
            "first_alert_s": first_alert,
            "first_warning_s": first_warn,
            "lead_first_alert_s": lead(first_alert),
            "lead_detection_only_s": lead(self.first_detection),
        }
