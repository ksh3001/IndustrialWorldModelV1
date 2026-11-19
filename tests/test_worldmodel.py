import numpy as np

from iwm import load_config
from iwm.schema import Track
from iwm.worldmodel import KalmanCV, WorldModel, detection_only

CFG = load_config()


def test_kalman_estimates_velocity():
    rng = np.random.default_rng(0)
    k = KalmanCV(0, 0, 0)
    for i in range(1, 40):
        t = i * 0.1
        s = k.step(2.0 * t + rng.normal(0, 0.08), 1.0 + rng.normal(0, 0.08), t)
    assert abs(s[2] - 2.0) < 0.3 and abs(s[3]) < 0.3


def test_head_on_crossing_is_high_risk_with_correct_ttc():
    wm = WorldModel(CFG)
    # Forklift 6 m left of the corner at 2 m/s, worker 3 m below at 1 m/s: both arrive in ~3 s.
    tracks = [Track(0, "fl", "forklift", 4.0, 6.0, 2.0, 0.0), Track(0, "wk", "worker", 10.0, 3.0, 0.0, 1.0)]
    r = wm.risk(tracks)[0]
    assert r.p_collision > 0.6
    assert 2.0 < r.ttc_s < 3.2


def test_parallel_paths_are_low_risk():
    wm = WorldModel(CFG)
    tracks = [Track(0, "fl", "forklift", 2.0, 6.0, 2.5, 0.0), Track(0, "wk", "worker", 4.0, 9.0, 1.3, 0.0)]
    assert wm.risk(tracks)[0].p_collision < 0.1


def test_explain_ranks_a_fix():
    wm = WorldModel(CFG)
    wm.update(0, [Track(0, "fl", "forklift", 4.0, 6.0), Track(0, "wk", "worker", 10.0, 3.0)])
    for i in range(1, 11):
        t = i * 0.1
        wm.update(t, [Track(t, "fl", "forklift", 4.0 + 2 * t, 6.0), Track(t, "wk", "worker", 10.0, 3.0 + t)])
    pair = wm.risk()[0]
    reasons = wm.explain(pair)
    assert reasons and reasons[0][1] > 0.3


def test_detection_only_fires_only_when_close():
    far = [Track(0, "fl", "forklift", 0, 0), Track(0, "wk", "worker", 5, 0)]
    near = [Track(0, "fl", "forklift", 0, 0), Track(0, "wk", "worker", 1.5, 0)]
    assert detection_only(far, 2.0) == [] and detection_only(near, 2.0) == [("fl", "wk")]
