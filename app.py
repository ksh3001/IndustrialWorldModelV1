"""Industrial World Model — demo UI (Streamlit).

    streamlit run app.py

Pages (sidebar):
  Overview        the problem, how it works, headline results
  Demo videos     pygame simulation + annotated real video, side by side
  Live scenario   play a simulated encounter, toggle what-if fixes, see risk + alerts + brief
  Results         1,000-run evaluation vs detection-only
  Reports         shift safety report: KPIs, hotspots, incident list, briefs, downloads
  Camera          alerts logged from real video (run_camera.py)
"""

import json
import time
from pathlib import Path

import streamlit as st

from iwm import load_config
from iwm.demo import DemoSession
from iwm.plots import floor_plot, hotspot_plot, risk_timeline
from iwm.report import executive_summary, incident_brief
from iwm.shift import incidents_csv, shift_report_markdown, shift_summary, simulate_shift
from iwm.sim import WhatIf

st.set_page_config(page_title="Industrial World Model", layout="wide")
cfg = load_config()
RESULTS = Path("results/results.json")

st.sidebar.title("Industrial World Model")
st.sidebar.caption("Team Future Frames · Industry & Energy")
page = st.sidebar.radio("Page", ["Overview", "Demo videos", "Live scenario", "Results", "Reports", "Camera"])


@st.cache_resource(show_spinner="Rolling futures...")
def run_session(scenario, seed, react, mirror, limit):
    s = DemoSession(cfg, scenario, int(seed), WhatIf(mirror=mirror, speed_limit=limit, workers_react_to_alerts=react))
    s.run()
    return s


@st.cache_resource(show_spinner="Simulating a shift...")
def run_shift(encounters, seed, react, mirror, limit):
    return simulate_shift(cfg, int(encounters), int(seed),
                          WhatIf(mirror=mirror, speed_limit=limit, workers_react_to_alerts=react))


def whatif_controls(key):
    c1, c2, c3 = st.columns(3)
    return (c1.toggle("Worker & driver react to alerts", key=f"{key}r"),
            c2.toggle("Convex mirror", key=f"{key}m"),
            c3.toggle("1 m/s speed limit near corner", key=f"{key}s"))


# --------------------------------------------------------------------- Overview
if page == "Overview":
    st.title("See the collision before it happens")
    st.write("Industrial sites are full of cameras, but footage is reviewed only after an accident, and today's "
             "video AI only flags danger when it is already happening. **Industrial World Model** tracks every "
             "worker and forklift, rolls the scene forward into 50 possible futures, and buzzes the worker's "
             "phone and the driver's beeper **seconds before** their paths cross.")
    if RESULTS.exists():
        r = json.loads(RESULTS.read_text())
        rate = r["crossing_contact_rate_pct"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Warning before contact", f"{r['world_model']['lead_first_alert_mean_s']:.1f} s",
                  f"vs {r['detection_only']['lead_mean_s']:.1f} s today", delta_color="off")
        c2.metric("Contacts flagged early", f"{r['world_model']['caught_pct']:.0f}%")
        c3.metric("False alarms", f"{r['world_model']['false_alarms_per_hour']:.0f}/h",
                  f"vs {r['detection_only']['false_alarms_per_hour']:.0f}/h detection-only", delta_color="off")
        c4.metric("Crossings ending in contact", f"{rate['World-model alerts']:.0f}%",
                  f"from {rate['No fix']:.0f}% with no alerts", delta_color="off")
        st.caption(f"From {r['runs']} simulated blind-corner encounters.")
    st.subheader("How it works")
    st.image("docs/architecture.png", use_container_width=True)
    st.markdown("""
1. **See** — YOLOv11 + ByteTrack find workers and forklifts in CCTV; a floor homography turns pixels into metres.
2. **Predict** — a Kalman filter estimates each speed and heading; 50 Monte Carlo futures per pair give
   collision probability and time to contact.
3. **Alert** — Advisory / Warning / Critical tiers with hysteresis; MQTT to the worker's phone and the forklift.
4. **Improve** — counterfactuals explain *why*, what-if fixes are tested in simulation, and Azure OpenAI writes
   incident briefs and shift reports.

*Privacy-first: positions only, no faces stored. Perception runs on real video; prediction is validated in
simulation because real near-misses are too rare and too dangerous to stage.*""")

# ------------------------------------------------------------------ Demo videos
elif page == "Demo videos":
    st.title("Demo: real video and simulation")
    st.write("Left: perception on real CCTV-style footage. Right: the world model in the warehouse simulation, "
             "where we can safely test thousands of near-misses.")
    left, right = st.columns(2)
    for col, path, title, cmd in (
        (left, Path("data/camera_demo.mp4"), "Real video: tracking + predicted paths",
         "python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4"),
        (right, Path("data/sim_demo.mp4"), "Simulation: blind corner (pygame)",
         "python demo_sim.py --mqtt --record data/sim_demo.mp4"),
    ):
        with col:
            st.subheader(title)
            if path.exists():
                st.video(str(path))
            else:
                up = st.file_uploader(f"Upload {path.name}", type=["mp4", "webm", "mov"], key=path.name)
                if up:
                    st.video(up)
                else:
                    st.info(f"Record it with:\n\n`{cmd}`")
    st.caption("Videos are read from the data/ folder. If one won't play in the browser, convert it to H.264 "
               "with the ffmpeg command printed when recording.")

# ---------------------------------------------------------------- Live scenario
elif page == "Live scenario":
    st.title("Live scenario")
    c1, c2 = st.columns([2, 1])
    scenario = c1.selectbox("Scenario", ["blind_corner", "crossing", "parallel"],
                            help="blind_corner = the fixed demo case")
    seed = c2.number_input("Seed", 0, 10_000, 1)
    react, mirror, limit = whatif_controls("live")
    session = run_session(scenario, seed, react, mirror, limit)
    sm = session.summary()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contact", "none" if sm["contact_s"] is None else f"at {sm['contact_s']} s")
    m2.metric("Closest approach", f"{sm['closest_m']} m")
    m3.metric("World model warned", "-" if sm["lead_first_alert_s"] is None else f"{sm['lead_first_alert_s']} s early")
    m4.metric("Detection-only warned",
              "-" if sm["lead_detection_only_s"] is None else f"{sm['lead_detection_only_s']} s early")

    n = len(session.history)
    play = st.button("Play")
    idx = st.slider("Replay", 0, n - 1, min(15, n - 1), format="frame %d")
    left, right = st.columns([3, 2])
    view = left.empty()
    if play:
        last = min(n - 1, int(((sm["contact_s"] or session.history[-1].t) + 2) / cfg["sim"]["dt"]))
        for i in range(0, last, 2):
            view.pyplot(floor_plot(session, i), clear_figure=True)
            time.sleep(0.05)
    else:
        view.pyplot(floor_plot(session, idx), clear_figure=True)
    with right:
        st.subheader("Alerts")
        if session.alerts:
            st.dataframe([a.to_dict() | {"pair": " / ".join(a.pair)} for a in session.alerts],
                         hide_index=True, use_container_width=True)
        else:
            st.write("No alerts.")
        if session.last_reasons:
            st.subheader("Why — counterfactuals")
            for name, drop in session.last_reasons:
                st.write(f"- {name}: risk −{drop:.0%}")
    st.pyplot(risk_timeline(session), clear_figure=True)
    st.subheader("Incident brief")
    if st.button("Generate brief (Azure OpenAI)"):
        first = session.alerts[0].to_dict() if session.alerts else None
        st.text(incident_brief(sm, first, session.last_reasons))

# ---------------------------------------------------------------------- Results
elif page == "Results":
    st.title("Results: world model vs detection-only")
    if RESULTS.exists():
        res = json.loads(RESULTS.read_text())
        wm_, base = res["world_model"], res["detection_only"]
        st.table({
            "Metric": ["Contacts flagged before they happen", "Average warning (first alert)",
                       "Average warning (Warning tier)", "False alarms per simulated hour"],
            "Detection-only (< 2 m now)": [f"{base['caught_pct']:.0f}% (last moment)", f"{base['lead_mean_s']:.1f} s",
                                           f"{base['lead_mean_s']:.1f} s", f"{base['false_alarms_per_hour']:.0f}"],
            "World model": [f"{wm_['caught_pct']:.0f}%", f"{wm_['lead_first_alert_mean_s']:.1f} s",
                            f"{wm_['lead_mean_s']:.1f} s", f"{wm_['false_alarms_per_hour']:.0f}"],
        })
        st.image("results/results.png", use_container_width=True)
        st.caption(f"{res['runs']} simulated runs ({wm_['contacts']} with contact). "
                   "Re-run: python -m iwm.evaluate --runs 1000")
    else:
        st.info("Run `python -m iwm.evaluate --runs 1000` first.")

# ---------------------------------------------------------------------- Reports
elif page == "Reports":
    st.title("Shift safety report")
    c1, c2 = st.columns(2)
    encounters = c1.slider("Forklift-worker encounters in the shift", 20, 200, 60, step=10)
    seed = c2.number_input("Shift seed", 0, 10_000, 100)
    react, mirror, limit = whatif_controls("rep")
    incidents = run_shift(encounters, seed, react, mirror, limit)
    summary = shift_summary(incidents, encounters)

    k = st.columns(5)
    k[0].metric("Encounters", summary["encounters"])
    k[1].metric("Incidents", summary["incidents"])
    k[2].metric("Critical", summary["critical"])
    k[3].metric("Contacts", summary["contacts"])
    k[4].metric("Avg warning", "-" if summary["avg_warning_s"] is None else f"{summary['avg_warning_s']} s")

    st.subheader("Executive summary")
    st.markdown(executive_summary(summary))

    left, right = st.columns([3, 2])
    left.pyplot(hotspot_plot(cfg, incidents), clear_figure=True)
    with right:
        st.subheader("Recommended fix")
        st.success(summary["top_fix"])
        st.caption("Toggle the what-if fixes above to see how the shift would change.")

    st.subheader("Incidents")
    st.dataframe([{k: v for k, v in i.items() if k not in ("brief", "x", "y")} for i in incidents],
                 hide_index=True, use_container_width=True)
    if incidents:
        pick = st.selectbox("Incident brief", [f"{i['id']} · {i['time']} · {i['peak_tier']}" for i in incidents])
        st.text(next(i for i in incidents if pick.startswith(i["id"]))["brief"])

    md = shift_report_markdown(summary, incidents)
    d1, d2 = st.columns(2)
    d1.download_button("Download shift report (.md)", md, "shift_report.md", "text/markdown")
    d2.download_button("Download incidents (.csv)", incidents_csv(incidents), "incidents.csv", "text/csv")

# ----------------------------------------------------------------------- Camera
elif page == "Camera":
    st.title("Camera alerts (real video)")
    log = Path("logs/alerts.jsonl")
    if log.exists():
        rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        st.metric("Alerts logged", len(rows))
        st.dataframe(rows[::-1], hide_index=True, use_container_width=True)
    else:
        st.info("Run `python run_camera.py --source data/demo.mp4` to log alerts from real video.")
    st.markdown("Phone wristband: `python -m http.server 8000`, then open "
                "`http://<laptop-ip>:8000/phone.html?broker=<laptop-ip>` on the phone.")
