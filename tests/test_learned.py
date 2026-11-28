import copy

import numpy as np

from iwm import load_config
from iwm.dataset import build_windows, sim_tracks, split
from iwm.learned import LearnedMotion
from iwm.schema import Track
from iwm.worldmodel import WorldModel

CFG = load_config()


def small_model(tmp_path):
    w = build_windows(sim_tracks(CFG, warehouse_runs=1, corner_runs=20, seconds=40), 2.0, 5.0)
    tr, va = split(w, 0.2)
    m = LearnedMotion(2.0, 5.0, hidden=32)
    m.fit(tr, va, epochs=3, verbose=False)
    path = tmp_path / "m.npz"
    m.save(path)
    return w, path


def test_windows_have_the_right_shape():
    w = build_windows(sim_tracks(CFG, warehouse_runs=1, corner_runs=5, seconds=30), 2.0, 5.0)
    assert len(w.X) > 100 and w.X.shape[1] == 42 and w.Y.shape[1] == 100


def test_save_load_and_sample(tmp_path):
    w, path = small_model(tmp_path)
    m = LearnedMotion.load(path)
    hist = np.stack([np.linspace(0, 2.5, 20), np.full(20, 6.0)], axis=1)  # moving +x at ~1.3 m/s
    fut = m.sample(hist, 1.3, 0.0, False, 25, 50, np.random.default_rng(0))
    assert fut.shape == (25, 50, 2) and np.isfinite(fut).all()
    assert fut[:, -1, 0].mean() > hist[-1, 0]  # still heading the same way


def test_hybrid_world_model_mixes_futures(tmp_path):
    _, path = small_model(tmp_path)
    c = copy.deepcopy(CFG)
    c["world_model"].update(mode="hybrid", learned_share=0.5, learned_model=str(path))
    wm = WorldModel(c)
    for i in range(30):  # 3 s of history: enough for the learned model
        t = i * 0.1
        wm.update(t, [Track(t, "fl", "forklift", 4 + 2 * t, 6.0), Track(t, "wk", "worker", 10.0, 1.0 + t)])
    r = wm.risk()[0]
    assert wm.stats["learned"] > 0 and 0 <= r.p_collision <= 1


def test_missing_model_falls_back_to_kalman():
    c = copy.deepcopy(CFG)
    c["world_model"].update(mode="hybrid", learned_model="models/does_not_exist.npz")
    wm = WorldModel(c)
    assert wm.learned_share == 0.0
