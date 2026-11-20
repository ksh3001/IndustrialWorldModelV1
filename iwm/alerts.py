"""Alert engine: risk -> tiered alerts with hysteresis -> MQTT (phone buzz).

Tiers (config.yaml):  advisory  p > 0.30 and ttc < 5 s
                      warning   p > 0.60 and ttc < 3 s
                      critical  p > 0.85 and ttc < 2 s

Hysteresis keeps alerts from flickering: a higher tier must hold for
`rise_frames` frames before we escalate, and a tier is only dropped after
`hold_s` seconds below its threshold. Only escalations are published.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

from .schema import TIERS, Alert
from .worldmodel import PairRisk


@dataclass
class _PairState:
    tier: int = 0
    pending: int = 0
    pending_tier: int = 0
    below_since: float | None = None


class AlertEngine:
    def __init__(self, cfg: dict):
        a = cfg["alerts"]
        self.rules = [(i, *a[name]) for i, name in enumerate(TIERS) if name != "none"]
        self.rise_frames, self.hold_s = a["rise_frames"], a["hold_s"]
        self.state: dict[tuple[str, str], _PairState] = {}

    def candidate(self, p: float, ttc: float) -> int:
        tier = 0
        for idx, p_min, ttc_max in self.rules:
            if p >= p_min and ttc <= ttc_max:
                tier = idx
        return tier

    def update(self, t: float, risks: list[PairRisk]) -> list[Alert]:
        alerts = []
        for r in risks:
            st = self.state.setdefault((r.forklift, r.worker), _PairState())
            cand = self.candidate(r.p_collision, r.ttc_s)
            if cand > st.tier:
                st.below_since = None
                st.pending += 1
                # escalate to the lowest tier seen during the rise, so one spike can't jump to critical
                st.pending_tier = cand if st.pending == 1 else min(st.pending_tier, cand)
                if st.pending >= self.rise_frames:
                    st.tier, st.pending = st.pending_tier, 0
                    alerts.append(Alert(round(t, 2), [r.forklift, r.worker], TIERS[st.tier],
                                        r.p_collision, r.ttc_s))
            elif cand < st.tier:
                st.pending = 0
                st.below_since = t if st.below_since is None else st.below_since
                if t - st.below_since >= self.hold_s:
                    st.tier, st.below_since = cand, None
            else:
                st.pending, st.below_since = 0, None
        return alerts

    def tier_of(self, forklift: str, worker: str) -> str:
        st = self.state.get((forklift, worker))
        return TIERS[st.tier] if st else "none"


class Publisher:
    """Sends alerts to MQTT (phone + dashboard); falls back to the console."""

    def __init__(self, cfg: dict, log_path: str = "logs/alerts.jsonl", beep: bool = True):
        m = cfg["mqtt"]
        self.topic, self.beep, self.client = m["topic"], beep, None
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        try:
            import paho.mqtt.client as mqtt

            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.connect(m["host"], m["port"], keepalive=30)
            self.client.loop_start()
        except Exception as e:  # no broker or no paho: still log + print
            print(f"[alerts] MQTT unavailable ({e}); printing alerts instead", file=sys.stderr)
            self.client = None

    def send(self, alert: Alert) -> None:
        msg = json.dumps(alert.to_dict())
        self.log.write(msg + "\n")
        self.log.flush()
        if self.client:
            self.client.publish(self.topic, msg, qos=0)
        if self.beep and alert.tier in ("warning", "critical"):
            sys.stdout.write("\a")  # laptop beep = the forklift's horn in the demo
        print(f"[ALERT] {alert.tier.upper():8s} {alert.pair} p={alert.p_collision:.2f} ttc={alert.ttc_s:.1f}s")


if __name__ == "__main__":
    # Test the phone buzz without running a demo:  python -m iwm.alerts warning
    import time as _time

    from . import load_config

    tier = sys.argv[1] if len(sys.argv) > 1 else "warning"
    pub = Publisher(load_config(), log_path="logs/test_alerts.jsonl", beep=False)
    if pub.client is None:
        raise SystemExit("MQTT broker not reachable: start Mosquitto with  mosquitto -c mosquitto.conf -v")
    pub.send(Alert(0.0, ["fl_test", "wk_test"], tier, 0.75, 2.5, "test alert"))
    _time.sleep(1.0)  # let the message go out before exiting
    print("Sent. The phone should vibrate now.")
