"""Write demo recordings that play in a browser (and so in the Streamlit UI).

Browsers need H.264 ('avc1'). Some OpenCV builds can't encode it, so we fall
back to 'mp4v' and print the one-line ffmpeg command that converts the file.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2


class VideoRecorder:
    def __init__(self, path: str, fps: float = 15):
        self.path, self.fps, self.writer, self.codec = path, fps, None, None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, frame_bgr) -> None:
        if self.writer is None:
            h, w = frame_bgr.shape[:2]
            for codec in ("avc1", "mp4v"):
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), self.fps, (w, h))
                if self.writer.isOpened():
                    self.codec = codec
                    break
        self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"[video] saved {self.path} ({self.codec})")
            if self.codec != "avc1" and shutil.which("ffmpeg"):  # make it browser-playable
                tmp = self.path.replace(".mp4", "_tmp.mp4")
                os.replace(self.path, tmp)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vcodec", "libx264",
                                "-preset", "veryfast", "-pix_fmt", "yuv420p", self.path], check=False)
                if os.path.exists(self.path):
                    os.remove(tmp)
                    print(f"[video] converted to H.264 for the browser: {self.path}")
                    return
                os.replace(tmp, self.path)
            if self.codec != "avc1":
                print(f"[video] if the browser won't play it: ffmpeg -y -i {self.path} -vcodec libx264 "
                      f"-pix_fmt yuv420p {self.path.replace('.mp4', '_h264.mp4')}")
