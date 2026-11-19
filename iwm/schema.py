"""The two records every module exchanges (see README: Data format).

Track:  one per object per frame, in floor metres and seconds.
Alert:  one per worker-forklift pair whose risk tier goes up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

WORKER = "worker"
FORKLIFT = "forklift"
TIERS = ["none", "advisory", "warning", "critical"]


@dataclass
class Track:
    t: float
    id: str
    type: str  # "worker" | "forklift"
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    t: float
    pair: list[str]  # [forklift_id, worker_id]
    tier: str  # "advisory" | "warning" | "critical"
    p_collision: float
    ttc_s: float
    cause: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["p_collision"] = round(self.p_collision, 3)
        d["ttc_s"] = round(self.ttc_s, 2)
        return d
