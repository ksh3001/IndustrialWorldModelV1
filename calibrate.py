"""Click the 4 taped floor markers to calibrate the camera.

    python calibrate.py --source data/demo.mp4

Click the markers in the SAME order as `floor_points` in config.yaml,
then press any key. Paste the printed `image_points` into config.yaml.
"""

import argparse

import cv2

from iwm import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="video file, webcam index or RTSP URL")
    args = ap.parse_args()
    cfg = load_config()
    src = args.source or cfg["camera"]["source"]
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {src}")

    points = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(frame, str(len(points)), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("calibrate", frame)

    print("Click the 4 floor markers in order:", cfg["camera"]["floor_points"])
    cv2.imshow("calibrate", frame)
    cv2.setMouseCallback("calibrate", on_click)
    while len(points) < 4:
        if cv2.waitKey(50) == 27:
            break
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print(f"\nimage_points: {points}")


if __name__ == "__main__":
    main()
