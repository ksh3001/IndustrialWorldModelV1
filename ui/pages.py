"""The six pages of the demo UI. app.py wires them into the sidebar navigation."""

from __future__ import annotations

import json
import time

import streamlit as st

from iwm.plots import floor_plot, hotspot_plot, risk_timeline
from iwm.report import executive_summary, incident_brief
from iwm.shift import incidents_csv, shift_report_markdown, shift_summary

from .common import (CAMERA_LOG, CFG, TIER_ICON, badge, header, load_results, next_link, run_session, run_shift,
                     whatif_picker)

NAV: dict = {}  # filled by app.py: name -> st.Page, used for "Next" links

SCENARIOS = {"Blind corner (demo)": "blind_corner", "Random crossing": "crossing", "Parallel paths": "parallel"}
ALERT_COLUMNS = {
    "t": st.column_config.NumberColumn("Time (s)", format="%.1f"),
    "tier": st.column_config.TextColumn("Tier"),
    "p_collision": st.column_config.ProgressColumn("Collision risk", min_value=0, max_value=1, format="%.2f"),
    "ttc_s": st.column_config.NumberColumn("Time to contact (s)", format="%.1f"),
    "cause": st.column_config.TextColumn("Top fix"),
}


# ----------------------------------------------------------------------------- 1
def overview():
    header("Industrial World Model", "See the collision before it happens",
           "Cameras today record accidents. Our world model predicts them, and buzzes the worker and the "
           "driver seconds before their paths cross.")
    res = load_results()
    if res:
        wm, base, rate = res["world_model"], res["detection_only"], res["crossing_contact_rate_pct"]
        c = st.columns(4)
        c[0].metric("Warning before contact", f"{wm['lead_first_alert_mean_s']:.1f} s",
                    f"{wm['lead_first_alert_mean_s'] - base['lead_mean_s']:+.1f} s vs today's camera AI")
        c[1].metric("Collisions flagged early", f"{wm['caught_pct']:.0f}%")
        c[2].metric("False alarms per hour", f"{wm['false_alarms_per_hour']:.0f}",
                    f"{wm['false_alarms_per_hour'] - base['false_alarms_per_hour']:+.0f} vs detection-only",
                    delta_color="inverse")
        c[3].metric("Crossings ending in contact", f"{rate['World-model alerts']:.0f}%",
                    f"{rate['World-model alerts'] - rate['No fix']:+.0f} pts with alerts", delta_color="inverse")
        st.caption(f"Measured on {res['runs']:,} simulated blind-corner encounters.")

    st.subheader("How it works")
    steps = [("1 · See", "YOLOv11 + ByteTrack track workers and forklifts on existing CCTV."),
             ("2 · Predict", "50 possible futures per pair, 5 s ahead: collision risk and time to contact."),
             ("3 · Alert", "Advisory → Warning → Critical, sent to the worker's phone and the forklift."),
             ("4 · Improve", "Explains why, tests fixes, and writes incident and shift reports.")]
    for col, (t, d) in zip(st.columns(4), steps):
        col.markdown(f'<div class="iwm-step"><b>{t}</b><span>{d}</span></div>', unsafe_allow_html=True)
    with st.expander("System architecture", expanded=True):
        st.image("docs/architecture.png")
    st.info("Privacy-first: positions only, no faces stored. Perception runs on real video; prediction is "
            "validated in simulation because real near-misses are too rare and too dangerous to stage.",
            icon=":material/lock:")
    next_link(NAV["live"], "watch it predict a collision")


# ----------------------------------------------------------------------------- 2
def live_demo():
    header("Step 2 · Live demo", "Watch it predict a collision",
           "A forklift and a worker approach a blind corner. Drag the time slider or press Play, then switch on "
           "a fix and watch the outcome change.")
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        label = c1.segmented_control("Scenario", list(SCENARIOS), default="Blind corner (demo)") or "Blind corner (demo)"
        seed = c2.number_input("Seed", 0, 10_000, 1, help="Change for a different random encounter.")
        react, mirror, limit = whatif_picker("live_fix")
    s = run_session(SCENARIOS[label], seed, react, mirror, limit)
    sm = s.summary()

    if sm["contact_s"] is not None:
        st.error(f"**Collision at {sm['contact_s']} s.** The world model warned "
                 f"**{sm['lead_first_alert_s'] or '-'} s** earlier; a detection-only camera fired only "
                 f"{sm['lead_detection_only_s'] or '-'} s before.", icon=":material/warning:")
    else:
        st.success(f"**No collision.** Closest approach {sm['closest_m']} m."
                   + (" Alerts reached the worker and driver in time." if react else ""),
                   icon=":material/check_circle:")

    c = st.columns(4)
    c[0].metric("Collision", "none" if sm["contact_s"] is None else f"at {sm['contact_s']} s")
    c[1].metric("Closest approach", f"{sm['closest_m']} m")
    c[2].metric("World model warned", "-" if sm["lead_first_alert_s"] is None else f"{sm['lead_first_alert_s']} s early")
    c[3].metric("Detection-only warned",
                "-" if sm["lead_detection_only_s"] is None else f"{sm['lead_detection_only_s']} s early")

    hist, dt = s.history, CFG["sim"]["dt"]
    t_end = round(min(hist[-1].t, (sm["contact_s"] or hist[-1].t) + 2), 1)
    left, right = st.columns([3, 2], gap="large")
    with left:
        b1, b2 = st.columns([1, 4])
        play = b1.button("Play", icon=":material/play_arrow:", type="primary")
        t_sel = b2.slider("Time", 0.0, t_end, min(1.5, t_end), step=0.1, format="%.1f s", label_visibility="collapsed")
        view = st.empty()
    frames = list(range(0, int(t_end / dt), 2)) if play else [int(round(t_sel / dt))]
    state = right.empty()
    for i in frames:
        i = min(i, len(hist) - 1)
        fr = hist[i]
        view.pyplot(floor_plot(s, i), clear_figure=True)
        top = max(fr.risks, key=lambda r: r.p_collision, default=None)
        with state.container(border=True):
            st.markdown(f"**At {fr.t:.1f} s** &nbsp; {badge(fr.tier)}", unsafe_allow_html=True)
            if top:
                k = st.columns(2)
                k[0].metric("Collision risk", f"{top.p_collision:.0%}")
                k[1].metric("Time to contact", "-" if top.ttc_s == float("inf") else f"{top.ttc_s:.1f} s")
            st.caption("Detection-only camera: " + ("🔴 firing" if fr.detection_only else "silent"))
        if play:
            time.sleep(0.04)

    with right:
        st.markdown("**Alerts sent**")
        if s.alerts:
            st.dataframe([a.to_dict() | {"tier": f"{TIER_ICON[a.tier]} {a.tier}"} for a in s.alerts],
                         column_order=list(ALERT_COLUMNS), column_config=ALERT_COLUMNS, hide_index=True)
        else:
            st.caption("No alerts in this scenario.")
        if s.last_reasons:
            st.markdown("**Why: what would have lowered the risk**")
            for name, drop in s.last_reasons:
                st.progress(max(0.0, min(1.0, drop)), text=f"{name}: −{drop:.0%}")

    st.subheader("Risk over time")
    st.pyplot(risk_timeline(s), clear_figure=True)
    with st.expander("Incident brief (Azure OpenAI)", icon=":material/description:"):
        if st.button("Generate brief"):
            first = s.alerts[0].to_dict() if s.alerts else None
            st.code(incident_brief(sm, first, s.last_reasons), language=None, wrap_lines=True)
    next_link(NAV["videos"], "see it on real video")


# ----------------------------------------------------------------------------- 3
def demo_videos():
    from pathlib import Path

    header("Step 3 · Demo videos", "Real video and simulation, side by side",
           "Left: tracking on real CCTV-style footage. Right: the pygame simulation, where thousands of "
           "near-misses can be tested safely.")
    left, right = st.columns(2, gap="large")
    for col, path, title, cmd in (
        (left, Path("data/camera_demo.mp4"), "Real video · tracking + predicted paths",
         "python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4"),
        (right, Path("data/sim_demo.mp4"), "Simulation · blind corner (pygame)",
         "python demo_sim.py --mqtt --record data/sim_demo.mp4"),
    ):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            if path.exists():
                st.video(str(path))
            else:
                up = st.file_uploader(f"Upload {path.name}", type=["mp4", "webm", "mov"], key=path.name)
                if up:
                    st.video(up)
                else:
                    st.caption("Not recorded yet. Record it with:")
                    st.code(cmd, language="bash")
    next_link(NAV["results"], "the evidence from 1,000 runs")


# ----------------------------------------------------------------------------- 4
def results():
    header("Step 4 · Results", "The evidence: 1,000 simulated encounters",
           "World model vs a detection-only camera that alerts when a pair is already under 2 m apart.")
    res = load_results()
    if not res:
        st.warning("No results yet. Run `python -m iwm.evaluate --runs 1000`.")
        return
    wm, base, rate = res["world_model"], res["detection_only"], res["crossing_contact_rate_pct"]
    c = st.columns(3)
    c[0].metric("Warning before contact", f"{wm['lead_first_alert_mean_s']:.1f} s", f"vs {base['lead_mean_s']:.1f} s",
                delta_color="off")
    c[1].metric("Collisions flagged early", f"{wm['caught_pct']:.0f}%")
    c[2].metric("False alarms per hour", f"{wm['false_alarms_per_hour']:.0f}", f"vs {base['false_alarms_per_hour']:.0f}",
                delta_color="off")
    st.image("results/results.png")
    with st.container(border=True):
        st.markdown("**What it means**")
        st.markdown(f"- Workers get about **{wm['lead_first_alert_mean_s'] / max(base['lead_mean_s'], 0.1):.0f}× more "
                    f"time** to react than with detection-only.\n"
                    f"- When alerts reach the worker and driver, crossings that end in contact fall from "
                    f"**{rate['No fix']:.0f}%** to **{rate['World-model alerts']:.0f}%**.\n"
                    f"- Fewer false alarms means workers keep trusting the buzz.")
    with st.expander("Full numbers"):
        st.json(res)
    next_link(NAV["reports"], "the shift report for site managers")


# ----------------------------------------------------------------------------- 5
def reports():
    header("Step 5 · Shift reports", "What a site manager sees after a shift",
           "Every alert becomes an incident with a brief, a hotspot and a suggested fix.")
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        encounters = c1.slider("Forklift-worker encounters in the shift", 20, 200, 60, step=10)
        seed = c2.number_input("Shift seed", 0, 10_000, 100)
        react, mirror, limit = whatif_picker("rep_fix")
    incidents = run_shift(encounters, seed, react, mirror, limit)
    sm = shift_summary(incidents, encounters)

    c = st.columns(5)
    c[0].metric("Encounters", sm["encounters"])
    c[1].metric("Incidents", sm["incidents"])
    c[2].metric("Critical", sm["critical"])
    c[3].metric("Collisions", sm["contacts"])
    c[4].metric("Avg warning", "-" if sm["avg_warning_s"] is None else f"{sm['avg_warning_s']} s")

    t1, t2, t3 = st.tabs([":material/summarize: Summary", ":material/list: Incidents", ":material/download: Export"])
    with t1:
        left, right = st.columns([3, 2], gap="large")
        left.pyplot(hotspot_plot(CFG, incidents), clear_figure=True)
        with right:
            st.markdown("**Executive summary**")
            st.markdown(executive_summary(sm))
            st.success(f"**Recommended fix:** {sm['top_fix']}", icon=":material/build:")
    with t2:
        rows = [{k: v for k, v in i.items() if k not in ("brief", "x", "y")} | {"peak_tier": f"{TIER_ICON[i['peak_tier']]} {i['peak_tier']}"}
                for i in incidents]
        st.dataframe(rows, hide_index=True, column_config={
            "max_p": st.column_config.ProgressColumn("Max risk", min_value=0, max_value=1, format="%.2f"),
            "contact": st.column_config.CheckboxColumn("Collision"),
            "warned_s_early": st.column_config.NumberColumn("Warned (s early)", format="%.1f"),
            "detection_only_s_early": st.column_config.NumberColumn("Detection-only (s early)", format="%.1f"),
        })
        if incidents:
            pick = st.selectbox("Open an incident brief",
                                [f"{i['id']} · {i['time']} · {i['peak_tier']}" for i in incidents])
            st.code(next(i for i in incidents if pick.startswith(i["id"]))["brief"], language=None, wrap_lines=True)
    with t3:
        md = shift_report_markdown(sm, incidents)
        d1, d2 = st.columns(2)
        d1.download_button("Shift report (.md)", md, "shift_report.md", "text/markdown",
                           icon=":material/description:", use_container_width=True)
        d2.download_button("Incidents (.csv)", incidents_csv(incidents), "incidents.csv", "text/csv",
                           icon=":material/table:", use_container_width=True)
        with st.expander("Preview the report"):
            st.markdown(md)
    next_link(NAV["camera"], "alerts from the real camera")


# ----------------------------------------------------------------------------- 6
def camera():
    header("Live source", "Camera alerts", "Alerts raised by `run_camera.py` on real video appear here.")
    if CAMERA_LOG.exists():
        rows = [json.loads(line) for line in CAMERA_LOG.read_text().splitlines() if line.strip()]
        st.metric("Alerts logged", len(rows))
        st.dataframe([r | {"tier": f"{TIER_ICON.get(r['tier'], '')} {r['tier']}"} for r in rows[::-1]],
                     column_order=list(ALERT_COLUMNS) + ["pair"], column_config=ALERT_COLUMNS, hide_index=True)
    else:
        st.info("No camera alerts yet.", icon=":material/videocam_off:")
    with st.container(border=True):
        st.markdown("**Connect the camera and the phone**")
        st.markdown("1. Start the broker: `mosquitto -c mosquitto.conf -v`\n"
                    "2. Serve the phone page: `python -m http.server 8000`\n"
                    "3. On the phone, open `http://<laptop-ip>:8000/phone.html?broker=<laptop-ip>` and tap Start\n"
                    "4. Run `python run_camera.py --source data/demo.mp4`")
