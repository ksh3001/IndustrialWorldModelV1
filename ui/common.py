"""Shared UI helpers: styling, headers, badges, status and cached runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from iwm import load_config
from iwm.alerts import NtfyNotifier, Publisher
from iwm.demo import DemoSession
from iwm.shift import simulate_shift
from iwm.sim import WhatIf

CFG = load_config()
RESULTS = Path("results/results.json")
CAMERA_LOG = Path("logs/alerts.jsonl")
TIER_ICON = {"none": "🟢", "advisory": "🟡", "warning": "🟠", "critical": "🔴"}

WHATIF_OPTIONS = {
    "Alerts reach worker & driver": "react",
    "Convex mirror": "mirror",
    "Speed limit 1 m/s at corner": "limit",
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1240px;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e5e7eb; border-radius: 14px;
  padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04);}
[data-testid="stMetricLabel"] p {font-size: .85rem; color: #64748b;}
[data-testid="stMetricValue"] {font-weight: 700;}
.iwm-kicker {font-size: .78rem; font-weight: 700; letter-spacing: .08em; color: #4f46e5; text-transform: uppercase;}
.iwm-sub {color: #475569; font-size: 1.02rem; margin: -.4rem 0 1rem 0;}
.iwm-badge {display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .85rem;
  color: #fff; letter-spacing: .03em;}
.iwm-none {background: #16a34a;} .iwm-advisory {background: #ca8a04;} .iwm-warning {background: #ea580c;}
.iwm-critical {background: #dc2626;}
.iwm-step {border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px; height: 100%; background: #fff;}
.iwm-step b {display: block; margin-bottom: 4px;}
.iwm-step span {color: #64748b; font-size: .9rem;}
.iwm-flow {font-size: .88rem; line-height: 1.9; color: #334155;}
.iwm-flow .on {font-weight: 700; color: #4f46e5;}
.iwm-status {font-size: .85rem; line-height: 1.8; color: #334155;}
</style>
"""


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="iwm-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="iwm-sub">{subtitle}</div>', unsafe_allow_html=True)


def badge(tier: str) -> str:
    label = "CLEAR" if tier == "none" else tier.upper()
    return f'<span class="iwm-badge iwm-{tier}">{label}</span>'


def next_link(page, label: str) -> None:
    st.divider()
    st.page_link(page, label=f"Next: {label}", icon=":material/arrow_forward:")


def whatif_picker(key: str) -> tuple[bool, bool, bool]:
    picked = st.pills("What-if fixes", list(WHATIF_OPTIONS), selection_mode="multi", key=key,
                      help="Switch fixes on and the simulation re-runs instantly.") or []
    flags = {WHATIF_OPTIONS[p] for p in picked}
    return "react" in flags, "mirror" in flags, "limit" in flags


def load_results() -> dict | None:
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None


def azure_ready() -> bool:
    return all(os.getenv(k) for k in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"))


def sidebar_extras(current: str, flow: list[str]) -> None:
    with st.sidebar:
        st.divider()
        st.caption("PRESENTATION FLOW")
        items = "".join(
            f'<div class="{"on" if name == current else ""}">{i}. {name}</div>' for i, name in enumerate(flow, 1))
        st.markdown(f'<div class="iwm-flow">{items}</div>', unsafe_allow_html=True)
        st.divider()
        st.caption("SYSTEM STATUS")
        rows = [
            ("World model", True, "ready"),
            ("Evaluation results", RESULTS.exists(), "loaded" if RESULTS.exists() else "run iwm.evaluate"),
            ("Azure OpenAI", azure_ready(), "connected" if azure_ready() else "template briefs"),
            ("Trained model", model_ready(), (f"hybrid {CFG['world_model'].get('learned_share', 0.5):.0%}"
                                              if CFG["world_model"].get("mode") != "kalman" else "loaded, off")
             if model_ready() else "run train_motion.py"),
            ("Phone (ntfy)", phone_ready(), f"topic {NtfyNotifier(CFG).topic}" if phone_ready() else "set NTFY_TOPIC in .env"),
            ("Camera log", CAMERA_LOG.exists(), "receiving" if CAMERA_LOG.exists() else "no video yet"),
        ]
        html = "".join(f'<div>{"🟢" if ok else "⚪"} {name}: <span style="color:#64748b">{note}</span></div>'
                       for name, ok, note in rows)
        st.markdown(f'<div class="iwm-status">{html}</div>', unsafe_allow_html=True)


def phone_ready() -> bool:
    return NtfyNotifier(CFG).enabled


@st.cache_resource(show_spinner=False)
def get_publisher() -> Publisher:
    """One publisher for the UI: sends to the ntfy app (and to MQTT / phone.html if a broker is running)."""
    return Publisher(CFG, log_path="logs/ui_alerts.jsonl", beep=False)


@st.cache_resource(show_spinner=False)
def get_camera_publisher() -> Publisher:
    """Publisher for real-video alerts: phone (ntfy / phone.html) + logs/alerts.jsonl for the Camera alerts page."""
    return Publisher(CFG, log_path=str(CAMERA_LOG), beep=False)


MODES = {"Hybrid (trained + Kalman)": "hybrid", "Kalman only": "kalman"}


def cfg_for(mode: str) -> dict:
    """A copy of the config with the world-model mode switched (kalman / hybrid)."""
    import copy

    c = copy.deepcopy(CFG)
    c["world_model"]["mode"] = mode
    return c


def model_ready() -> bool:
    from iwm.learned import load_cached

    return load_cached(CFG["world_model"].get("learned_model", "models/motion.npz")) is not None


def mode_picker(key: str) -> str:
    """World-model switch shown on the simulation pages; returns 'hybrid' or 'kalman'."""
    if not model_ready():
        st.caption("World model: Kalman only (no trained model yet: run `python train_motion.py`).")
        return "kalman"
    share = CFG["world_model"].get("learned_share", 0.5)
    default = "Hybrid (trained + Kalman)" if CFG["world_model"].get("mode", "kalman") != "kalman" else "Kalman only"
    label = st.segmented_control("World model", list(MODES), default=default, key=key,
                                 help=f"Hybrid: {share:.0%} of the 50 futures come from the trained motion model, "
                                      "the rest from Kalman rules.") or default
    return MODES[label]


@st.cache_resource(show_spinner="Rolling 50 futures per frame...")
def run_session(scenario: str, seed: int, react: bool, mirror: bool, limit: bool, mode: str = "kalman") -> DemoSession:
    s = DemoSession(cfg_for(mode), scenario, int(seed),
                    WhatIf(mirror=mirror, speed_limit=limit, workers_react_to_alerts=react))
    s.run()
    return s


@st.cache_resource(show_spinner="Simulating a shift...")
def run_shift(encounters: int, seed: int, react: bool, mirror: bool, limit: bool) -> list[dict]:
    return simulate_shift(CFG, int(encounters), int(seed),
                          WhatIf(mirror=mirror, speed_limit=limit, workers_react_to_alerts=react))
