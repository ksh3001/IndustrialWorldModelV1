"""Write demo recordings that play in a browser (and so in the Streamlit UI).

OpenCV records with the 'mp4v' codec, which works everywhere (no extra DLLs).
Browsers need H.264, so on close we convert with ffmpeg: the one bundled by
the `imageio-ffmpeg` pip package, or a system ffmpeg if installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class VideoRecorder:
    def __init__(self, path: str, fps: float = 15):
        self.path, self.fps, self.writer = path, fps, None
        self.raw = path.replace(".mp4", "_raw.mp4")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, frame_bgr) -> None:
        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            self.writer = cv2.VideoWriter(self.raw, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        if os.path.exists(self.path):
            os.remove(self.path)
        ffmpeg = _ffmpeg()
        if ffmpeg:
            print("[video] converting to H.264 for the browser...")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", self.raw, "-vcodec", "libx264",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                           check=False)
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            os.remove(self.raw)
            print(f"[video] saved {self.path}")
        else:
            os.replace(self.raw, self.path)
            print(f"[video] saved {self.path} (mp4v; run 'pip install imageio-ffmpeg' so it plays in the browser)")
