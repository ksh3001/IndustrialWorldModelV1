"""Does the trained model help? Kalman vs hybrid vs learned, on prediction accuracy and on safety.

    python evaluate_motion.py                                  # simulation test tracks + 300 safety runs
    python evaluate_motion.py --real "logs/tracks/*.jsonl"     # also score on real video tracks
    python evaluate_motion.py --runs 1000 --shares 0 0.3 0.5 0.7 1

1. Prediction accuracy: on tracks the model never trained on (new seeds / your real clips), how many
   metres off is the predicted position after 1, 3 and 5 s? Kalman = straight line, learned = the
   trained model's most likely path.
2. Safety: the same blind-corner evaluation as iwm.evaluate (warning time, collisions caught, false
   alarms per hour), run once per learned share (0 = Kalman only, 0.5 = hybrid 50/50, 1 = learned only).

Writes results/motion_results.json and results/motion_results.png (shown on the UI Results page).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os

import numpy as np

from iwm import load_config
from iwm.dataset import build_windows, read_track_log, sim_tracks
from iwm.evaluate import run_once, summarise
from iwm.learned import LearnedMotion


def accuracy(model: LearnedMotion, w) -> dict:
    """Mean position error (m) at 1, 3, 5 s and averaged over the horizon, Kalman vs learned."""
    if not len(w.X):
        return {}
    S = model.S
    true = w.Y.reshape(-1, S, 2)  # residual vs straight line = Kalman's error vector
    pred = model.predict_residual(w.X).reshape(-1, S, 2)
    e_k, e_l = np.linalg.norm(true, axis=2), np.linalg.norm(true - pred, axis=2)
    out = {"windows": int(len(w.X))}
    for name, e in (("kalman", e_k), ("learned", e_l)):
        out[name] = {f"{s}s": round(float(e[:, int(s / 0.1) - 1].mean()), 3) for s in (1, 3, 5) if s / 0.1 <= S}
        out[name]["average"] = round(float(e.mean()), 3)
    for typ, mask in (("worker", ~w.is_forklift), ("forklift", w.is_forklift)):
        if mask.any():
            out[f"{typ}_average"] = {"kalman": round(float(e_k[mask].mean()), 3),
                                     "learned": round(float(e_l[mask].mean()), 3)}
    return out


def safety(cfg: dict, share: float, runs: int) -> dict:
    c = copy.deepcopy(cfg)
    c["world_model"]["mode"] = "hybrid" if share > 0 else "kalman"
    c["world_model"]["learned_share"] = share
    rows = [run_once(c, s) for s in range(runs)]
    m = summarise(rows, "first_warn", "warn_events")
    first = summarise(rows, "first_alert", "warn_events")
    m["lead_first_alert_mean_s"] = first["lead_mean_s"]
    return m


def chart(res: dict, path: str = "results/motion_results.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = res["accuracy"].get("real") or res["accuracy"]["simulation"]
    src = "real video" if res["accuracy"].get("real") else "simulation"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    hs = [k for k in ("1s", "3s", "5s") if k in acc["kalman"]]
    x = np.arange(len(hs))
    for i, (name, colour) in enumerate((("kalman", "#94a3b8"), ("learned", "#4f46e5"))):
        bars = a1.bar(x + (i - 0.5) * 0.38, [acc[name][h] for h in hs], 0.38, color=colour,
                      label="Kalman (straight line)" if name == "kalman" else "Trained model")
        a1.bar_label(bars, fmt="%.2f", fontsize=8)
    a1.set_xticks(x, [f"after {h}" for h in hs])
    a1.set_ylabel("position error (m)")
    a1.set_title(f"Prediction error on unseen {src} tracks (lower is better)")
    a1.legend(frameon=False, fontsize=9)
    a1.spines[["top", "right"]].set_visible(False)

    rows = res["safety"]
    labels = ["Kalman only" if r["learned_share"] == 0 else "Learned only" if r["learned_share"] == 1
              else f"Hybrid {r['learned_share']:.0%}" for r in rows]
    colours = ["#94a3b8" if r["learned_share"] == 0 else "#4f46e5" if r["learned_share"] == 0.5 else "#a5b4fc"
               for r in rows]
    bars = a2.bar(labels, [r["false_alarms_per_hour"] for r in rows], color=colours)
    a2.bar_label(bars, labels=[f"{r['false_alarms_per_hour']:.1f}/h\nwarns {r['lead_first_alert_mean_s']:.1f} s early"
                               for r in rows], fontsize=8)
    a2.set_ylabel("false alarms per hour")
    a2.set_ylim(0, max(r["false_alarms_per_hour"] for r in rows) * 1.35)
    a2.set_title("False alarms by mix (same warning time; blind-corner simulation)")
    a2.tick_params(axis="x", labelsize=8)
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: world_model.learned_model in config.yaml")
    ap.add_argument("--real", default=None, help='glob of real track logs to score on')
    ap.add_argument("--runs", type=int, default=300, help="safety runs per share")
    ap.add_argument("--shares", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7, 1.0])
    args = ap.parse_args()
    cfg = load_config()
    if args.model:
        cfg["world_model"]["learned_model"] = args.model
    model = LearnedMotion.load(cfg["world_model"]["learned_model"])

    print("1/2 Prediction accuracy on unseen tracks...")
    test = build_windows(sim_tracks(cfg, warehouse_runs=4, corner_runs=100, seed0=200_000),
                         model.history_s, model.horizon_s)
    acc = {"simulation": accuracy(model, test)}
    if args.real:
        tracks = {}
        for f in sorted(glob.glob(args.real)):
            tracks.update(read_track_log(f))
        acc["real"] = accuracy(model, build_windows(tracks, model.history_s, model.horizon_s, stride_s=0.2))
    print(json.dumps(acc, indent=2))

    print(f"2/2 Safety, {args.runs} blind-corner runs per share...")
    rows = []
    for share in args.shares:
        m = safety(cfg, share, args.runs)
        m["learned_share"] = share
        rows.append(m)
        print(f"  share {share:.1f}: warning {m['lead_first_alert_mean_s']:.2f} s before contact, "
              f"caught {m['caught_pct']:.0f}%, false alarms {m['false_alarms_per_hour']:.1f}/h")
    res = {"model": cfg["world_model"]["learned_model"], "accuracy": acc, "safety": rows, "runs": args.runs}
    os.makedirs("results", exist_ok=True)
    with open("results/motion_results.json", "w") as f:
        json.dump(res, f, indent=2)
    chart(res)


if __name__ == "__main__":
    main()
