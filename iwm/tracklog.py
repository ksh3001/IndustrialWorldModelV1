"""Save tracked detections from real video, for training the motion model (train_motion.py).

One JSON line per detection: {"t", "id", "type", "x", "y"} in seconds and floor metres.
Files go to logs/tracks/<clip>.jsonl; train_motion.py --real "logs/tracks/*.jsonl" reads them.
"""

from __future__ import annotations

import json
from pathlib import Path


class TrackLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")
        self.count = 0

    def write(self, t: float, dets: list[dict]) -> None:
        for d in dets:
            self.f.write(json.dumps({"t": round(float(t), 3), "id": d["id"], "type": d["type"],
                                     "x": round(float(d["x"]), 3), "y": round(float(d["y"]), 3)}) + "\n")
            self.count += 1

    def close(self) -> None:
        self.f.close()
        print(f"[tracks] saved {self.count} detections to {self.path}")


def default_path(source: str | Path) -> Path:
    return Path("logs/tracks") / f"{Path(str(source)).stem}.jsonl"
