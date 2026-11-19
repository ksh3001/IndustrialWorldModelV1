"""Perception on real video: YOLOv11 + ByteTrack + floor homography.

Turns camera frames into floor-plane detections (metres), the same shape
the simulation produces, so the world model never knows the difference.

    from iwm.perception import Perception
    for frame, t, dets in Perception(cfg).stream():
        ...  # dets: [{"id": "wk_3", "type": "worker", "x": 2.1, "y": 3.4, "bbox": [...]}]
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


class Perception:
    def __init__(self, cfg: dict, source=None):
        from ultralytics import YOLO  # imported here so the rest of the repo runs without it

        cam = cfg["camera"]
        self.source = cam["source"] if source is None else source
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        self.model = YOLO(cam["model"])
        self.homography = Homography(cam["image_points"], cam["floor_points"])
        names = self.model.names  # class id -> name
        self.kind = {}
        for cid, name in names.items():
            if name in cam["worker_classes"]:
                self.kind[cid] = WORKER
            elif name in cam["forklift_classes"]:
                self.kind[cid] = FORKLIFT
        self.fps = self._fps()

    def _fps(self) -> float | None:
        if isinstance(self.source, str) and not self.source.startswith(("rtsp", "http")):
            cap = cv2.VideoCapture(self.source)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps or None
        return None  # live stream: use wall-clock time

    def stream(self):
        """Yield (frame, t_seconds, detections) for every frame."""
        t_start = time.time()
        results = self.model.track(self.source, stream=True, persist=True, tracker="bytetrack.yaml",
                                   classes=list(self.kind), verbose=False)
        for i, r in enumerate(results):
            t = i / self.fps if self.fps else time.time() - t_start
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
