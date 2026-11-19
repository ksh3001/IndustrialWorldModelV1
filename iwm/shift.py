"""Shift reporting: simulate a shift of forklift-worker encounters and collect incidents.

Each scenario run is one encounter at the blind corner. Every run that raised
an alert or ended in contact becomes an incident record with its own brief.
The shift summary rolls them up for the site manager.
"""

from __future__ import annotations

import csv
import io
from collections import Counter

import numpy as np

from .demo import DemoSession
from .report import site_fix, template_brief
from .schema import TIERS
from .sim import WhatIf


def simulate_shift(cfg: dict, encounters: int = 60, seed: int = 100, whatif: WhatIf | None = None,
                   shift_minutes: int = 480) -> list[dict]:
    """Run `encounters` scenarios spread across a shift; return incident records."""
    rng = np.random.default_rng(seed)
    clock = np.sort(rng.uniform(0, shift_minutes * 60, encounters))
    incidents = []
    for i, start in enumerate(clock):
        s = DemoSession(cfg, "random", seed + i, whatif)
        s.run()
        if not s.alerts and s.sim.contact_t is None:
            continue
        sm = s.summary()
        peak = max((a.tier for a in s.alerts), key=TIERS.index, default="none")
        top = max(s.alerts, key=lambda a: a.p_collision) if s.alerts else None
        # where the pair were closest = the hotspot location
        closest = min(s.history, key=lambda f: min((r.distance_m for r in f.risks), default=99))
        pos = [(t.x, t.y) for t in closest.tracks]
        hx, hy = (float(np.mean([p[0] for p in pos])), float(np.mean([p[1] for p in pos]))) if pos else (0.0, 0.0)
        hh, mm = divmod(int(start // 60), 60)
        incidents.append({
            "id": f"INC-{i + 1:03d}",
            "time": f"{6 + hh:02d}:{mm:02d}",
            "scenario": s.sim.scenario,
            "peak_tier": peak,
            "max_p": round(top.p_collision, 2) if top else None,
            "contact": sm["contact_s"] is not None,
            "closest_m": sm["closest_m"],
            "warned_s_early": sm["lead_first_alert_s"],
            "detection_only_s_early": sm["lead_detection_only_s"],
            "suggested_fix": site_fix(s.last_reasons),
            "x": round(hx, 1), "y": round(hy, 1),
            "brief": template_brief(sm, s.alerts[0].to_dict() if s.alerts else None, s.last_reasons),
        })
    return incidents


def shift_summary(incidents: list[dict], encounters: int) -> dict:
    tiers = Counter(i["peak_tier"] for i in incidents)
    leads = [i["warned_s_early"] for i in incidents if i["warned_s_early"] is not None]
    fixes = Counter(i["suggested_fix"] for i in incidents)
    return {
        "encounters": encounters,
        "incidents": len(incidents),
        "contacts": sum(i["contact"] for i in incidents),
        "critical": tiers.get("critical", 0),
        "warning": tiers.get("warning", 0),
        "advisory": tiers.get("advisory", 0),
        "avg_warning_s": round(float(np.mean(leads)), 1) if leads else None,
        "top_fix": fixes.most_common(1)[0][0] if fixes else "-",
    }


def shift_report_markdown(summary: dict, incidents: list[dict], site: str = "Warehouse A · blind corner") -> str:
    lines = [
        f"# Shift safety report — {site}",
        "",
        f"- Encounters observed: **{summary['encounters']}**",
        f"- Incidents (alert or contact): **{summary['incidents']}** "
        f"(critical {summary['critical']}, warning {summary['warning']}, advisory {summary['advisory']})",
        f"- Contacts: **{summary['contacts']}**",
        f"- Average warning before contact: **{summary['avg_warning_s']} s**",
        f"- Recommended fix: **{summary['top_fix']}**",
        "",
        "## Incidents",
        "",
        "| ID | Time | Peak tier | Max risk | Contact | Closest (m) | Warned (s early) | Suggested fix |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in incidents:
        lines.append(f"| {i['id']} | {i['time']} | {i['peak_tier']} | {i['max_p']} | {'yes' if i['contact'] else 'no'} "
                     f"| {i['closest_m']} | {i['warned_s_early'] or '-'} | {i['suggested_fix']} |")
    return "\n".join(lines) + "\n"


def incidents_csv(incidents: list[dict]) -> str:
    buf = io.StringIO()
    cols = [k for k in incidents[0] if k != "brief"] if incidents else []
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(incidents)
    return buf.getvalue()
