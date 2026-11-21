import numpy as np

from iwm import load_config
from iwm.perception import ApproxScale
from iwm.pipeline import VideoPipeline

CFG = load_config()


class FakePublisher:
    def __init__(self):
        self.sent = []

    def send(self, alert, wait=False):
        self.sent.append(alert)


def test_approx_scale_round_trip():
    m = ApproxScale(1920, 1080, 15.0)
    x, y = m.to_floor(960, 540)
    assert abs(x - 7.5) < 1e-6 and m.to_pixel(x, y) == (960, 540)


def test_pipeline_alerts_and_draws_on_a_crossing():
    """A worker and a vehicle heading for the same point: alerts fire, go to the publisher, frames are drawn."""
    mapper = ApproxScale(1280, 720, 20.0)
    pub = FakePublisher()
    pipe = VideoPipeline(CFG, mapper, pub)
    frame = np.zeros((720, 1280, 3), np.uint8)
    tiers = []
    for i in range(40):
        t = i * 0.1
        fx, fy = 4.0 + 2.5 * t, 6.0          # vehicle moving right at 2.5 m/s
        wx, wy = 10.0, 1.5 + 1.2 * t         # worker moving up at 1.2 m/s, both reach (10, 6) at ~2.4-3.7 s
        dets = []
        for kind, tid, (x, y) in (("forklift", "fl_1", (fx, fy)), ("worker", "wk_1", (wx, wy))):
            u, v = mapper.to_pixel(x, y)
            dets.append({"id": tid, "type": kind, "x": x, "y": y, "bbox": [u - 20, v - 60, u + 20, v]})
        out, info = pipe.step(frame, t, dets)
        tiers.append(info.tier)
        assert out.shape == frame.shape
    assert pub.sent and pub.sent[0].tier in ("advisory", "warning", "critical")
    assert "warning" in tiers or "critical" in tiers
    assert pipe.peak in ("warning", "critical")
