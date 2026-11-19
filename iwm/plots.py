"""Matplotlib figures for the dashboard (kept here so they can be tested headless)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIER_COLOUR = {"advisory": "#ca8a04", "warning": "#ea580c", "critical": "#dc2626"}


def floor_plot(session, frame_idx: int | None = None):
    """Top-down view: aisles, trajectories so far, and predicted paths at one moment."""
    cfg = session.cfg["sim"]
    W, H = cfg["width_m"], cfg["height_m"]
    cx, cy = cfg["corner"]
    hist = session.history
    fr = hist[frame_idx if frame_idx is not None else -1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.add_patch(plt.Rectangle((0, 0), W, H, color="#94a3b8"))
    ax.add_patch(plt.Rectangle((0, cy - 1.5), W, 3, color="#e2e8f0"))
    ax.add_patch(plt.Rectangle((cx - 1, 0), 2, H, color="#e2e8f0"))
    upto = hist[: hist.index(fr) + 1]
    for tid, colour in (("fl_01", "#ea580c"), ("wk_01", "#2563eb")):
        xs = [t.x for f in upto for t in f.tracks if t.id == tid]
        ys = [t.y for f in upto for t in f.tracks if t.id == tid]
        ax.plot(xs, ys, color=colour, lw=1.5, alpha=0.6)
        if xs:
            ax.plot(xs[-1], ys[-1], "s" if tid.startswith("fl") else "o", color=colour, ms=12,
                    label="Forklift" if tid.startswith("fl") else "Worker")
    for r in fr.risks:
        for path in r.paths.values():
            ax.plot(path[:, 0], path[:, 1], "--", color=TIER_COLOUR.get(fr.tier, "#16a34a"), lw=2)
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal", xticks=[], yticks=[],
           title=f"t = {fr.t:.1f} s · {fr.tier.upper() if fr.tier != 'none' else 'CLEAR'}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def risk_timeline(session):
    """Collision probability over time, alerts, detection-only moment and true contact."""
    hist = session.history
    ts = [f.t for f in hist]
    ps = [max((r.p_collision for r in f.risks), default=0) for f in hist]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ts, ps, color="#4f46e5", lw=2, label="World-model collision risk")
    for i, a in enumerate(session.alerts):
        ax.axvline(a.t, color=TIER_COLOUR[a.tier], lw=1.5, ls=":")
        ax.text(a.t, 1.12 + 0.1 * (i % 3), a.tier, color=TIER_COLOUR[a.tier], fontsize=8, ha="center")
    if session.first_detection is not None:
        ax.axvline(session.first_detection, color="#64748b", lw=1.5, ls="--")
        ax.text(session.first_detection, 0.5, " detection-only", color="#64748b", fontsize=8)
    if session.sim.contact_t is not None:
        ax.axvline(session.sim.contact_t, color="#dc2626", lw=2)
        ax.text(session.sim.contact_t, 0.85, " contact", color="#dc2626", fontsize=9)
    marks = [a.t for a in session.alerts] + [session.sim.contact_t or 0, session.first_detection or 0]
    ax.set(ylim=(0, 1.4), xlim=(0, min(ts[-1], max(marks) + 4)), xlabel="seconds", ylabel="P(collision)",
           yticks=[0, 0.25, 0.5, 0.75, 1.0])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def hotspot_plot(cfg: dict, incidents: list[dict]):
    """Where incidents happened on the floor, coloured by peak tier."""
    s = cfg["sim"]
    W, H = s["width_m"], s["height_m"]
    cx, cy = s["corner"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.add_patch(plt.Rectangle((0, 0), W, H, color="#94a3b8"))
    ax.add_patch(plt.Rectangle((0, cy - 1.5), W, 3, color="#e2e8f0"))
    ax.add_patch(plt.Rectangle((cx - 1, 0), 2, H, color="#e2e8f0"))
    for tier in ("advisory", "warning", "critical"):
        pts = [(i["x"], i["y"]) for i in incidents if i["peak_tier"] == tier]
        if pts:
            ax.scatter(*zip(*pts), s=90, alpha=0.7, color=TIER_COLOUR[tier], edgecolor="white", label=f"{tier} ({len(pts)})")
    contacts = [(i["x"], i["y"]) for i in incidents if i["contact"]]
    if contacts:
        ax.scatter(*zip(*contacts), s=30, marker="x", color="#0f172a", label=f"contact ({len(contacts)})")
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal", xticks=[], yticks=[], title="Incident hotspots")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
