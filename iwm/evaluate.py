"""Evaluate the world model against detection-only on simulated scenarios.

    python -m iwm.evaluate --runs 1000

Reports, for both approaches:
  - near-misses caught: share of true contacts that got a Warning first
  - lead time: seconds between the first Warning (and first alert of any
    tier, including Advisory) and the true contact
  - false alarms: Warnings in runs with no contact, per simulated hour
Then the what-if results: contact rate with no fix, a convex mirror, a speed
limit, and with workers reacting to our alerts.
Writes results/results.json and results/results.png.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import load_config
from .alerts import AlertEngine
from .sim import Simulation, WhatIf
from .worldmodel import WorldModel, detection_only


def run_once(cfg, seed, scenario="random", whatif=None, use_model=True):
    sim = Simulation(cfg, scenario, seed, whatif)
    wm, engine = WorldModel(cfg), AlertEngine(cfg)
    first_warn = first_alert = first_det = None
    warn_events = det_events = 0
    det_active = False
    while sim.t < sim.duration:
        t = sim.t
        obs = sim.step()
        if not use_model:
            continue
        tracks = wm.update(t, obs)
        for a in engine.update(t, wm.risk(tracks)):
            first_alert = t if first_alert is None else first_alert
            if a.tier in ("warning", "critical"):
                sim.notify_alert(a.pair[1], a.tier)
                first_warn = t if first_warn is None else first_warn
                warn_events += 1
        close = bool(detection_only(tracks, cfg["alerts"]["detection_only_m"]))
        if close and not det_active:
            det_events += 1
            first_det = t if first_det is None else first_det
        det_active = close
    return dict(contact=sim.contact_t, first_warn=first_warn, first_alert=first_alert, first_det=first_det,
                warn_events=warn_events, det_events=det_events, duration=sim.duration)


def summarise(rows, key_first, key_events):
    contacts = [r for r in rows if r["contact"] is not None]
    caught = [r for r in contacts if r[key_first] is not None and r[key_first] < r["contact"]]
    leads = [r["contact"] - r[key_first] for r in caught]
    safe = [r for r in rows if r["contact"] is None]
    hours = sum(r["duration"] for r in safe) / 3600
    return dict(contacts=len(contacts), caught_pct=100 * len(caught) / max(len(contacts), 1),
                lead_mean_s=float(np.mean(leads)) if leads else 0.0,
                lead_median_s=float(np.median(leads)) if leads else 0.0,
                false_alarms_per_hour=sum(r[key_events] for r in safe) / max(hours, 1e-9))


def contact_rate(cfg, runs, whatif, use_model=False):
    hits = sum(run_once(cfg, s, "crossing", whatif, use_model)["contact"] is not None for s in range(runs))
    return 100 * hits / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    args = ap.parse_args()
    cfg = load_config()

    rows = [run_once(cfg, s) for s in range(args.runs)]
    model = summarise(rows, "first_warn", "warn_events")
    first = summarise(rows, "first_alert", "warn_events")
    model["lead_first_alert_mean_s"] = first["lead_mean_s"]  # includes Advisory buzzes
    baseline = summarise(rows, "first_det", "det_events")

    n = max(args.runs // 2, 50)
    whatifs = {
        "No fix": contact_rate(cfg, n, WhatIf()),
        "Convex mirror": contact_rate(cfg, n, WhatIf(mirror=True)),
        "Speed limit 1 m/s": contact_rate(cfg, n, WhatIf(speed_limit=True)),
        "World-model alerts": contact_rate(cfg, n, WhatIf(workers_react_to_alerts=True), use_model=True),
    }
    results = {"runs": args.runs, "world_model": model, "detection_only": baseline,
               "crossing_contact_rate_pct": whatifs}

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    _chart(results)


def _chart(res, path="results/results.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["Detection-only\n(< 2 m now)", "World model\nfirst Warning", "World model\nfirst alert"]
    leads = [res["detection_only"]["lead_mean_s"], res["world_model"]["lead_mean_s"],
             res["world_model"]["lead_first_alert_mean_s"]]
    bars = a1.bar(names, leads, color=["#94a3b8", "#818cf8", "#4f46e5"])
    a1.bar_label(bars, fmt="%.1f s")
    a1.set_title("Average warning before contact")
    a1.set_ylabel("seconds")
    a1.spines[["top", "right"]].set_visible(False)

    w = res["crossing_contact_rate_pct"]
    bars = a2.barh(list(w), list(w.values()), color=["#94a3b8", "#0ea5e9", "#0ea5e9", "#4f46e5"])
    a2.bar_label(bars, fmt="%.0f%%")
    a2.invert_yaxis()
    a2.set_title("Crossing scenarios that end in contact")
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
