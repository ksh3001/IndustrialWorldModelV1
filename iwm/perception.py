"""Perception on real video: YOLOv11 + ByteTrack + pixel-to-floor mapping.

Turns camera frames into floor-plane detections (metres), the same shape
the simulation produces, so the world model never knows the difference.

    from iwm.perception import Perception
    for frame, t, dets in Perception(cfg, "data/demo.mp4").stream():
        ...  # dets: [{"id": "wk_3", "type": "worker", "x": 2.1, "y": 3.4, "bbox": [...]}]

Two ways to map pixels to metres:
  - Homography:  calibrated from 4 taped floor markers (calibrate.py). Accurate.
  - ApproxScale: no calibration; assumes the frame is roughly top-down and
                 `width_m` metres wide. Good enough for a demo on a downloaded clip.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .schema import FORKLIFT, WORKER


class Homography:
    """Pixel <-> floor-metre mapping from 4 taped floor markers."""

    def __init__(self, image_points, floor_points):
        src = np.array(image_points, dtype=np.float32)
        dst = np.array(floor_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        self.H_inv = np.linalg.inv(self.H)

    def to_floor(self, u: float, v: float) -> tuple[float, float]:
        p = cv2.perspectiveTransform(np.array([[[u, v]]], np.float32), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def to_pixel(self, x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.H_inv)[0, 0]
        return int(p[0]), int(p[1])


class ApproxScale:
    """Uncalibrated fallback: a uniform metres-per-pixel scale (frame is `width_m` wide)."""

    def __init__(self, frame_width_px: int, frame_height_px: int, width_m: float = 15.0):
        self.s = width_m / float(frame_width_px)  # metres per pixel
        self.h = frame_height_px

    def to_floor(self, u: float, v: float) -> tuple[float, float]:
        return float(u * self.s), float((self.h - v) * self.s)

    def to_pixel(self, x: float, y: float) -> tuple[int, int]:
        return int(x / self.s), int(self.h - y / self.s)


def video_info(source) -> tuple[int, int, float | None]:
    """(width, height, fps) of a video file or stream; fps is None for live streams."""
    cap = cv2.VideoCapture(source)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    live = isinstance(source, int) or str(source).startswith(("rtsp", "http"))
    return w, h, (None if live or not fps else float(fps))


class Perception:
    def __init__(self, cfg: dict, source=None, mapper=None, stride: int = 1, conf: float = 0.3):
        from ultralytics import YOLO  # imported here so the rest of the repo runs without it

        cam = cfg["camera"]
        self.source = cam["source"] if source is None else source
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        self.model = YOLO(cam["model"])
        self.homography = mapper or Homography(cam["image_points"], cam["floor_points"])
        self.stride, self.conf = max(1, int(stride)), conf
        self.kind = {}
        for cid, name in self.model.names.items():  # class id -> name
            if name in cam["worker_classes"]:
                self.kind[cid] = WORKER
            elif name in cam["forklift_classes"]:
                self.kind[cid] = FORKLIFT
        self.width, self.height, self.fps = video_info(self.source)

    def stream(self):
        """Yield (frame, t_seconds, detections) for every processed frame."""
        t_start = time.time()
        results = self.model.track(self.source, stream=True, persist=True, tracker="bytetrack.yaml",
                                   classes=list(self.kind), conf=self.conf, vid_stride=self.stride, verbose=False)
        for i, r in enumerate(results):
            t = i * self.stride / self.fps if self.fps else time.time() - t_start
            dets = []
            if r.boxes is not None and r.boxes.id is not None:
                for box, cid, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int),
                                         r.boxes.id.cpu().numpy().astype(int)):
                    kind = self.kind.get(int(cid))
                    if kind is None:
                        continue
                    x1, y1, x2, y2 = box
                    # Bottom-centre of the box = where the object touches the floor.
                    fx, fy = self.homography.to_floor((x1 + x2) / 2, y2)
                    prefix = "wk" if kind == WORKER else "fl"
                    dets.append({"id": f"{prefix}_{tid}", "type": kind, "x": fx, "y": fy,
                                 "bbox": [int(x1), int(y1), int(x2), int(y2)]})
            yield r.orig_img, t, dets
