"""Live demo on real video: camera -> tracks -> world model -> alerts -> phone.

    python run_camera.py --source data/demo.mp4                       # calibrated (config.yaml homography)
    python run_camera.py --source data/samples/clip.mp4 --approx 15   # uncalibrated: frame ~15 m wide
    python run_camera.py --source 0                                   # webcam / phone camera
    python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4   # record for the UI

Shows the video with boxes, speeds, predicted paths and the alert tier. Press q to quit.
Alerts go to the phone (ntfy app if NTFY_TOPIC is set; phone.html if Mosquitto runs) and to
logs/alerts.jsonl for the UI. The UI's "Real video" page does the same in the browser.
"""

import argparse

import cv2

from iwm import load_config
from iwm.alerts import Publisher
from iwm.perception import ApproxScale, Perception, video_info
from iwm.pipeline import VideoPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--approx", type=float, default=None, help="no calibration: frame width in metres")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (faster on CPU)")
    ap.add_argument("--save", default=None, help="save the annotated video (e.g. data/camera_demo.mp4)")
    ap.add_argument("--headless", action="store_true", help="no window (just record / alert)")
    args = ap.parse_args()

    cfg = load_config()
    source = args.source or cfg["camera"]["source"]
    mapper = None
    if args.approx:
        w, h, _ = video_info(int(source) if str(source).isdigit() else source)
        mapper = ApproxScale(w, h, args.approx)
    perception = Perception(cfg, source, mapper=mapper, stride=args.stride)
    pipe = VideoPipeline(cfg, perception.homography, Publisher(cfg))
    recorder = None
    if args.save:
        from iwm.video import VideoRecorder

        recorder = VideoRecorder(args.save, fps=(perception.fps or 15) / args.stride)

    for frame, t, dets in perception.stream():
        annotated, _ = pipe.step(frame, t, dets)
        if recorder:
            recorder.write(annotated)
        if not args.headless:
            cv2.imshow("Industrial World Model", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if recorder:
        recorder.close()
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"Done: {len(pipe.alerts)} alerts, peak tier {pipe.peak}.")


if __name__ == "__main__":
    main()
