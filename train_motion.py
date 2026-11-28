"""Train the learned motion model (the trained half of the hybrid world model).

    python train_motion.py                                   # pretrain on simulation -> models/motion.npz
    python train_motion.py --real "logs/tracks/*.jsonl"      # pretrain on simulation, then fine-tune on real tracks
    python train_motion.py --real "logs/tracks/*.jsonl" --history 1 --horizon 3   # short real clips

Real tracks come from:  python run_camera.py --source clip.mp4 --approx 15 --log-tracks
(or the UI Real video page, "Save tracks for training"). They land in logs/tracks/.

Then compare against Kalman:  python evaluate_motion.py
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from iwm import load_config
from iwm.dataset import build_windows, read_track_log, sim_tracks, split
from iwm.learned import LearnedMotion


def describe(name, w):
    print(f"{name}: {len(w.X):,} windows ({int(w.is_forklift.sum()):,} forklift, "
          f"{int((~w.is_forklift).sum()):,} worker) from {len(np.unique(w.group)):,} tracks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help='glob of real track logs, e.g. "logs/tracks/*.jsonl"')
    ap.add_argument("--no-sim", action="store_true", help="train on real tracks only (no pretraining)")
    ap.add_argument("--warehouse-runs", type=int, default=12, help="simulated 2-minute warehouse shifts")
    ap.add_argument("--corner-runs", type=int, default=300, help="simulated blind-corner encounters")
    ap.add_argument("--history", type=float, default=2.0, help="seconds of history the model sees")
    ap.add_argument("--horizon", type=float, default=5.0, help="seconds the model predicts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--out", default="models/motion.npz")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    model = None
    report = {"history_s": args.history, "horizon_s": args.horizon}

    if not args.no_sim:
        print("Generating simulated tracks...")
        sim = build_windows(sim_tracks(cfg, args.warehouse_runs, args.corner_runs), args.history, args.horizon)
        tr, va = split(sim, 0.15)
        describe("Simulation train", tr)
        describe("Simulation val  ", va)
        print("Pretraining on simulation...")
        model = LearnedMotion(args.history, args.horizon)
        model.fit(tr, va, epochs=args.epochs)
        report["sim_windows"] = int(len(sim.X))

    if args.real:
        files = sorted(glob.glob(args.real))
        tracks = {}
        for f in files:
            tracks.update(read_track_log(f))
        real = build_windows(tracks, args.history, args.horizon, stride_s=0.2)
        print(f"Real track logs: {len(files)} files, {len(tracks)} tracks")
        describe("Real", real)
        if len(real.X) < 20:
            print("Not enough real windows to fine-tune (need tracks at least "
                  f"{args.history + args.horizon:.0f} s long). Try --history 1 --horizon 3.")
        else:
            tr, va = split(real, 0.2)
            print("Fine-tuning on real tracks..." if model else "Training on real tracks...")
            new = LearnedMotion(args.history, args.horizon)
            new.fit(tr, va, epochs=args.finetune_epochs if model else args.epochs,
                    lr=3e-4 if model else 1e-3, init=model)
            model = new
            report["real_windows"] = int(len(real.X))
            report["real_files"] = files

    if model is None:
        raise SystemExit("Nothing to train on: drop --no-sim or give --real track logs.")
    model.save(args.out)
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"Saved {args.out} in {report['seconds']} s. Next: python evaluate_motion.py")


if __name__ == "__main__":
    main()
