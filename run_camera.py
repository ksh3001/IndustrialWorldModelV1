"""Live demo on real video: camera -> tracks -> world model -> alerts -> phone.

    mosquitto -c mosquitto.conf          # in another terminal
    python run_camera.py --source data/demo.mp4
    python run_camera.py --source 0      # webcam / phone camera via USB or IP cam app
    python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4   # record for the UI

Shows the video with boxes, IDs, speeds, predicted paths and the alert tier.
Press q to quit. Alerts are also written to logs/alerts.jsonl for the dashboard.
"""

import argparse

import cv2
import numpy as np

from iwm import load_config
from iwm.alerts import AlertEngine, Publisher
from iwm.perception import Perception
from iwm.worldmodel import WorldModel

COLOURS = {"none": (120, 200, 120), "advisory": (0, 200, 255), "warning": (0, 120, 255), "critical": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--save", default=None, help="save the annotated video (e.g. data/camera_demo.mp4)")
    ap.add_argument("--headless", action="store_true", help="no window (just record / alert)")
    args = ap.parse_args()

    cfg = load_config()
    perception = Perception(cfg, args.source)
    wm, engine, pub = WorldModel(cfg), AlertEngine(cfg), Publisher(cfg)
    homo = perception.homography
    recorder = None
    if args.save:
        from iwm.video import VideoRecorder

        recorder = VideoRecorder(args.save, fps=perception.fps or 15)

    for frame, t, dets in perception.stream():
        tracks = wm.update(t, dets)
        risks = wm.risk(tracks)
        for alert in engine.update(t, risks):
            reasons = wm.explain(next(r for r in risks if [r.forklift, r.worker] == alert.pair))
            alert.cause = reasons[0][0] if reasons else ""
            pub.send(alert)

        # --- overlay -------------------------------------------------------
        worst = "none"
        for r in risks:
            tier = engine.tier_of(r.forklift, r.worker)
            if ["none", "advisory", "warning", "critical"].index(tier) > ["none", "advisory", "warning", "critical"].index(worst):
                worst = tier
            for path in r.paths.values():  # predicted mean paths, floor -> pixels
                pts = np.array([homo.to_pixel(x, y) for x, y in path[::5]], np.int32)
                cv2.polylines(frame, [pts], False, COLOURS[tier], 2)
        speeds = {tr.id: np.hypot(tr.vx, tr.vy) for tr in tracks}
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{d['id']} {speeds.get(d['id'], 0):.1f} m/s", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        top = max(risks, key=lambda r: r.p_collision, default=None)
        banner = f"{worst.upper()}" + (f"  p={top.p_collision:.2f}  ttc={top.ttc_s:.1f}s" if top else "")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), COLOURS[worst], -1)
        cv2.putText(frame, banner, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if recorder:
            recorder.write(frame)
        if not args.headless:
            cv2.imshow("Industrial World Model", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if recorder:
        recorder.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
