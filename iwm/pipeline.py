"""Real-video pipeline: detections -> world model -> alerts -> annotated frame.

Shared by run_camera.py (desktop window) and the UI's Real video page, so both
behave exactly the same. Perception is separate (iwm/perception.py), which keeps
this file testable without YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .alerts import AlertEngine
from .schema import TIERS, Alert
from .worldmodel import WorldModel

COLOURS = {"none": (94, 197, 34), "advisory": (8, 179, 234), "warning": (12, 88, 234), "critical": (38, 38, 220)}  # BGR


@dataclass
class StepInfo:
    t: float
    tier: str
    p_collision: float
    ttc_s: float
    distance_m: float | None
    workers: int
    vehicles: int
    new_alerts: list[Alert] = field(default_factory=list)


class VideoPipeline:
    def __init__(self, cfg: dict, mapper, publisher=None):
        self.cfg, self.mapper, self.pub = cfg, mapper, publisher
        self.wm, self.engine = WorldModel(cfg), AlertEngine(cfg)
        self.alerts: list[Alert] = []
        self.peak = "none"

    def step(self, frame: np.ndarray, t: float, dets: list[dict]) -> tuple[np.ndarray, StepInfo]:
        tracks = self.wm.update(t, dets)
        risks = self.wm.risk(tracks)
        new = self.engine.update(t, risks)
        for a in new:
            pair = next((r for r in risks if [r.forklift, r.worker] == a.pair), None)
            reasons = self.wm.explain(pair) if pair else []
            a.cause = reasons[0][0] if reasons else ""
            self.alerts.append(a)
            if self.pub:
                self.pub.send(a)
        tier = max((self.engine.tier_of(r.forklift, r.worker) for r in risks), key=TIERS.index, default="none")
        self.peak = max(self.peak, tier, key=TIERS.index)
        top = max(risks, key=lambda r: r.p_collision, default=None)
        info = StepInfo(t, tier, top.p_collision if top else 0.0, top.ttc_s if top else float("inf"),
                        top.distance_m if top else None,
                        sum(d["type"] == "worker" for d in dets), sum(d["type"] == "forklift" for d in dets), new)
        return self.draw(frame.copy(), tracks, risks, dets, info), info

    def draw(self, frame, tracks, risks, dets, info: StepInfo) -> np.ndarray:
        h, w = frame.shape[:2]
        for r in risks:  # predicted mean paths, floor -> pixels
            colour = COLOURS[self.engine.tier_of(r.forklift, r.worker)]
            for path in r.paths.values():
                pts = np.array([self.mapper.to_pixel(x, y) for x, y in path[::5]], np.int32)
                pts = np.clip(pts, [-w, -h], [2 * w, 2 * h])
                cv2.polylines(frame, [pts], False, colour, 3, cv2.LINE_AA)
        speeds = {tr.id: float(np.hypot(tr.vx, tr.vy)) for tr in tracks}
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            colour = (235, 99, 37) if d["type"] == "worker" else (12, 88, 234)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f"{'Worker' if d['type'] == 'worker' else 'Vehicle'} {d['id']} {speeds.get(d['id'], 0):.1f} m/s"
            cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        bar = max(40, h // 18)
        cv2.rectangle(frame, (0, 0), (w, bar), COLOURS[info.tier], -1)
        text = "CLEAR" if info.tier == "none" else info.tier.upper()
        if info.p_collision > 0:
            ttc = "-" if info.ttc_s == float("inf") else f"{info.ttc_s:.1f}s"
            text += f"   risk {info.p_collision:.0%}   contact in {ttc}"
        cv2.putText(frame, text, (14, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, bar / 50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"t={info.t:.1f}s", (w - 120, int(bar * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame
