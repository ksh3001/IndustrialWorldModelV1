"""Learned motion model: the trained half of the hybrid world model.

What it learns
  Given the last `history_s` seconds of a track (Kalman-smoothed positions) it predicts how the
  next `horizon_s` seconds DIFFER from Kalman's straight-line extrapolation: slowing before a
  corner, stopping, curving, turning at an aisle end. Learning only this correction (a "residual")
  needs far less data than learning motion from scratch, and an untrained model falls back to
  plain Kalman.

How it samples futures
  mean correction  = small neural network (NumPy MLP, trained with Adam, no PyTorch needed)
  spread           = learned from the training errors (low-rank Gaussian, one per object type)
  future           = Kalman straight line + mean correction + sampled spread

Everything is done in the object's own heading frame (x = direction of travel), so a model trained
on one camera or layout transfers to another. Saved as a single .npz file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DT = 0.1


# --------------------------------------------------------------------------- features
def heading_of(hist: np.ndarray, vx: float, vy: float) -> float:
    """Direction of travel: Kalman velocity if moving, else history displacement, else 0."""
    if np.hypot(vx, vy) > 0.2:
        return float(np.arctan2(vy, vx))
    d = hist[-1] - hist[0]
    return float(np.arctan2(d[1], d[0])) if np.hypot(*d) > 0.3 else 0.0


def rot(theta: float) -> np.ndarray:
    """World -> local rotation matrix (local x = heading)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def features(hist: np.ndarray, vx: float, vy: float, is_forklift: bool) -> tuple[np.ndarray, float]:
    """hist: (H, 2) smoothed positions at DT spacing, last row = now. Returns (feature vector, heading)."""
    th = heading_of(hist, vx, vy)
    rel = (hist - hist[-1]) @ rot(th).T
    return np.concatenate([rel.ravel(), [np.hypot(vx, vy), float(is_forklift)]]), th


def cv_path(cur: np.ndarray, vx: float, vy: float, steps: int) -> np.ndarray:
    t = np.arange(1, steps + 1)[:, None] * DT
    return cur + np.array([vx, vy]) * t


# ------------------------------------------------------------------------------- MLP
class MLP:
    """Two hidden layers, tanh, linear output. Trained with Adam on mean squared error."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 128, seed: int = 0):
        r = np.random.default_rng(seed)
        self.p = {
            "W1": r.normal(0, np.sqrt(1 / n_in), (n_in, hidden)), "b1": np.zeros(hidden),
            "W2": r.normal(0, np.sqrt(1 / hidden), (hidden, hidden)), "b2": np.zeros(hidden),
            "W3": r.normal(0, np.sqrt(1 / hidden) * 0.1, (hidden, n_out)), "b3": np.zeros(n_out),
        }

    def forward(self, x: np.ndarray, keep: bool = False):
        h1 = np.tanh(x @ self.p["W1"] + self.p["b1"])
        h2 = np.tanh(h1 @ self.p["W2"] + self.p["b2"])
        y = h2 @ self.p["W3"] + self.p["b3"]
        return (y, (x, h1, h2)) if keep else y

    def fit(self, X, Y, Xv=None, Yv=None, epochs=40, batch=256, lr=1e-3, seed=0, verbose=True) -> list:
        r = np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in self.p.items()}
        v_ = {k: np.zeros_like(v) for k, v in self.p.items()}
        step, history, best, best_p = 0, [], np.inf, None
        for ep in range(epochs):
            idx = r.permutation(len(X))
            for i in range(0, len(X), batch):
                b = idx[i:i + batch]
                y, (x, h1, h2) = self.forward(X[b], keep=True)
                g = 2 * (y - Y[b]) / y.size
                grads = {"W3": h2.T @ g, "b3": g.sum(0)}
                d2 = (g @ self.p["W3"].T) * (1 - h2 ** 2)
                grads.update({"W2": h1.T @ d2, "b2": d2.sum(0)})
                d1 = (d2 @ self.p["W2"].T) * (1 - h1 ** 2)
                grads.update({"W1": x.T @ d1, "b1": d1.sum(0)})
                step += 1
                for k in self.p:  # Adam
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v_[k] = 0.999 * v_[k] + 0.001 * grads[k] ** 2
                    mh, vh = m[k] / (1 - 0.9 ** step), v_[k] / (1 - 0.999 ** step)
                    self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
            tr = float(np.mean((self.forward(X) - Y) ** 2))
            va = float(np.mean((self.forward(Xv) - Yv) ** 2)) if Xv is not None and len(Xv) else tr
            history.append((tr, va))
            if va < best:  # keep the best epoch on the validation set
                best, best_p = va, {k: v.copy() for k, v in self.p.items()}
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  epoch {ep + 1:3d}/{epochs}  train {tr:.4f}  val {va:.4f}")
        if best_p:
            self.p = best_p
        return history


# ------------------------------------------------------------------------ the model
@dataclass
class Windows:
    X: np.ndarray  # (N, F) features
    Y: np.ndarray  # (N, S*2) residuals in the local frame
    is_forklift: np.ndarray  # (N,) bool
    group: np.ndarray  # (N,) source id, for train/test splits


class LearnedMotion:
    def __init__(self, history_s: float = 2.0, horizon_s: float = 5.0, hidden: int = 128, rank: int = 8):
        self.H = int(round(history_s / DT))
        self.S = int(round(horizon_s / DT))
        self.history_s, self.horizon_s, self.hidden, self.rank = history_s, horizon_s, hidden, rank
        self.net: MLP | None = None
        self.x_mu = self.x_sd = self.y_sd = None
        self.noise: dict[bool, np.ndarray] = {}  # is_forklift -> (rank, S*2) spread basis
        self.info: dict = {}

    # ----------------------------------------------------------------- training
    def fit(self, train: Windows, val: Windows | None = None, epochs: int = 40, lr: float = 1e-3,
            init: "LearnedMotion | None" = None, verbose: bool = True) -> list:
        """Train (or fine-tune, with `init`) on windows from iwm.dataset."""
        if init is not None:  # fine-tune: keep the pretrained network and its scaling
            self.net, self.x_mu, self.x_sd, self.y_sd = init.net, init.x_mu, init.x_sd, init.y_sd
        else:
            self.x_mu, self.x_sd = train.X.mean(0), train.X.std(0) + 1e-6
            self.y_sd = train.Y.std() + 1e-6
            self.net = MLP(train.X.shape[1], train.Y.shape[1], self.hidden)
        Xt, Yt = (train.X - self.x_mu) / self.x_sd, train.Y / self.y_sd
        Xv = Yv = None
        if val is not None and len(val.X):
            Xv, Yv = (val.X - self.x_mu) / self.x_sd, val.Y / self.y_sd
        hist = self.net.fit(Xt, Yt, Xv, Yv, epochs=epochs, lr=lr, verbose=verbose)
        ref = val if val is not None and len(val.X) else train
        res = ref.Y - self.predict_residual(ref.X)  # what the network still gets wrong = the spread
        for fk in (False, True):
            r = res[ref.is_forklift == fk]
            if len(r) < 20:
                r = res
            _, s, vt = np.linalg.svd(r - r.mean(0), full_matrices=False)
            k = min(self.rank, len(s))
            self.noise[fk] = (s[:k, None] * vt[:k]) / np.sqrt(len(r))
        self.info.update(train_windows=int(len(train.X)), val_windows=int(0 if val is None else len(val.X)))
        return hist

    # ---------------------------------------------------------------- inference
    def predict_residual(self, X: np.ndarray) -> np.ndarray:
        return self.net.forward((X - self.x_mu) / self.x_sd) * self.y_sd

    def mean_path(self, hist: np.ndarray, vx: float, vy: float, is_forklift: bool, steps: int) -> np.ndarray:
        return self.sample(hist, vx, vy, is_forklift, 0, steps, None)[0]

    def sample(self, hist, vx, vy, is_forklift, n, steps, rng) -> np.ndarray:
        """Futures (n, steps, 2) in world coordinates; n=0 returns the mean path as (1, steps, 2)."""
        f, th = features(hist, vx, vy, is_forklift)
        res = self.predict_residual(f[None])[0]  # (S*2,)
        if n:
            z = rng.normal(0, 1, (n, self.noise[bool(is_forklift)].shape[0]))
            res = res + z @ self.noise[bool(is_forklift)]
        res = np.atleast_2d(res).reshape(-1, self.S, 2) @ rot(th)  # local -> world
        out = cv_path(hist[-1], vx, vy, self.S) + res
        if steps > self.S:  # model horizon shorter than the world model's: keep the last velocity
            v_end = (out[:, -1] - out[:, -2]) / DT
            extra = np.arange(1, steps - self.S + 1)[None, :, None] * DT * v_end[:, None, :]
            out = np.concatenate([out, out[:, -1:] + extra], axis=1)
        return out[:, :steps]

    # -------------------------------------------------------------------- files
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"net_{k}": v for k, v in self.net.p.items()}
        arrays.update(x_mu=self.x_mu, x_sd=self.x_sd, y_sd=np.array(self.y_sd),
                      noise_worker=self.noise[False], noise_forklift=self.noise[True],
                      meta=np.array([self.history_s, self.horizon_s, self.hidden, self.rank]))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "LearnedMotion":
        z = np.load(path)
        hs, hz, hid, rk = z["meta"]
        m = cls(float(hs), float(hz), int(hid), int(rk))
        m.net = MLP(1, 1, m.hidden)
        m.net.p = {k[4:]: z[k] for k in z.files if k.startswith("net_")}
        m.x_mu, m.x_sd, m.y_sd = z["x_mu"], z["x_sd"], float(z["y_sd"])
        m.noise = {False: z["noise_worker"], True: z["noise_forklift"]}
        return m


@lru_cache(maxsize=4)
def load_cached(path: str) -> LearnedMotion | None:
    """Load once per process; None if the file is missing (the world model then uses Kalman only)."""
    p = Path(path)
    if not p.is_absolute():
        from . import ROOT

        p = ROOT / p
    return LearnedMotion.load(p) if p.exists() else None
