"""The six pages of the demo UI. app.py wires them into the sidebar navigation."""

from __future__ import annotations

import json
import time

import cv2

import streamlit as st

from iwm.plots import floor_plot, hotspot_plot, risk_timeline
from iwm.report import executive_summary, incident_brief
from iwm.shift import incidents_csv, shift_report_markdown, shift_summary

from .common import (CAMERA_LOG, CFG, TIER_ICON, badge, get_camera_publisher, get_publisher, header, load_results,
                     next_link, phone_ready, run_session, run_shift, whatif_picker)
from iwm.schema import Alert

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
        p1, p2 = st.columns([3, 1])
        buzz = p1.toggle("Buzz phone on Play", value=phone_ready(), disabled=not phone_ready(),
                         help="Sends each alert to the ntfy app at the moment it happens. "
                              "Set NTFY_TOPIC in .env and restart the UI to enable.")
        if p2.button("Send test buzz", icon=":material/vibration:", disabled=not phone_ready()):
            get_publisher().send(Alert(0.0, ["fl_test", "wk_test"], "warning", 0.75, 2.5, "test from the UI"), wait=True)
            if get_publisher().ntfy.last_ok:
                st.toast("Test alert sent to the phone", icon=":material/check_circle:")
            else:
                st.toast("Send failed: check internet (try the phone's hotspot)", icon=":material/error:")
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
    prev_t = -1.0
    for i in frames:
        t_frame = time.time()
        i = min(i, len(hist) - 1)
        fr = hist[i]
        if play and buzz:  # send the alerts that happen in this step, at the moment they happen
            for a in s.alerts:
                if prev_t < a.t <= fr.t:
                    get_publisher().send(a)
                    st.toast(f"{TIER_ICON[a.tier]} {a.tier.upper()} sent to phone at {a.t:.1f} s")
            prev_t = fr.t
        view.pyplot(floor_plot(s, i), clear_figure=True)
        top = max(fr.risks, key=lambda r: r.p_collision, default=None)
        with state.container(border=True):
            st.markdown(f"**At {fr.t:.1f} s** &nbsp; {badge(fr.tier)}", unsafe_allow_html=True)
            if top:
                k = st.columns(2)
                k[0].metric("Collision risk", f"{top.p_collision:.0%}")
                k[1].metric("Time to contact", "-" if top.ttc_s == float("inf") else f"{top.ttc_s:.1f} s")
            st.caption("Detection-only camera: " + ("🔴 firing" if fr.detection_only else "silent"))
        if play:  # play in real time (2 frames = 0.2 s), so the buzz lines up with the picture
            time.sleep(max(0.0, 2 * dt - (time.time() - t_frame)))

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
    next_link(NAV["real"], "run it on a real video")


# ---------------------------------------------------------------------------- 2b
def real_video():
    from pathlib import Path

    header("Step 3 · Real video", "Upload a video, get alerts",
           "Runs the full pipeline on real footage: YOLO tracks workers and vehicles, the world model predicts "
           "their next 5 seconds, and alerts go to the worker's phone as they happen.")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        st.error("YOLO isn't installed. Run `pip install -r requirements.txt`, then restart the UI.")
        return

    data = Path("data")
    with st.container(border=True):
        up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"],
                              help="A fixed, high camera works best (like CCTV).")
        existing = sorted(str(p) for p in list(data.glob("*.mp4")) + list(data.glob("samples/*.mp4"))
                          + list(data.glob("uploads/*")) if "_annotated" not in p.name and "_raw" not in p.name)
        pick = st.selectbox("…or choose a video already in data/", ["—"] + existing)
        c1, c2, c3 = st.columns(3)
        calib = c1.radio("Distances", ["Approximate (no calibration)", "Calibrated (config.yaml)"],
                         help="Calibrated uses the 4 floor markers from calibrate.py. Approximate assumes the "
                              "frame is roughly top-down and the width you set.")
        width_m = c2.number_input("Scene width (m)", 3.0, 100.0, 15.0, 1.0,
                                  disabled=calib.startswith("Calibrated"),
                                  help="Roughly how many metres the frame spans left to right.")
        stride = c3.select_slider("Speed", options=[1, 2, 3, 4], value=2,
                                  format_func=lambda n: f"every {n} frame{'s' if n > 1 else ''}",
                                  help="Process fewer frames to keep up on a laptop CPU.")
        o1, o2 = st.columns(2)
        buzz = o1.toggle("Send alerts to phone", value=phone_ready(), disabled=not phone_ready(),
                         help="ntfy app (NTFY_TOPIC in .env), plus phone.html if Mosquitto is running.")
        realtime = o2.toggle("Play in real time", value=True,
                             help="Off = process as fast as possible.")
        start = st.button("Analyse video", icon=":material/play_arrow:", type="primary")

    source = None
    if up is not None:
        (data / "uploads").mkdir(parents=True, exist_ok=True)
        source = data / "uploads" / up.name
        source.write_bytes(up.getbuffer())
    elif pick != "—":
        source = Path(pick)
    if not start:
        st.caption("Choose a video and press Analyse. Click any control (or Stop, top right) to stop early.")
        return
    if source is None:
        st.warning("Upload a video or choose one first.")
        return

    from iwm.perception import ApproxScale, Perception, video_info
    from iwm.pipeline import VideoPipeline
    from iwm.video import VideoRecorder

    w, h, fps = video_info(str(source))
    if not w:
        st.error(f"Couldn't open {source}.")
        return
    mapper = None if calib.startswith("Calibrated") else ApproxScale(w, h, width_m)
    with st.spinner("Loading YOLO (first run downloads the model)..."):
        try:
            perception = Perception(CFG, str(source), mapper=mapper, stride=stride)
        except Exception as e:  # usually: model download blocked by a corporate network
            model = CFG["camera"]["model"]
            st.error(f"Couldn't load the YOLO model `{model}`: {e.__class__.__name__}", icon=":material/error:")
            st.markdown(
                f"This usually means the network blocked the model download (corporate SSL inspection).\n\n"
                f"1. Download it in your browser: "
                f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model}\n"
                f"2. Put `{model}` in the project folder (next to `app.py`) and press **Analyse video** again.\n\n"
                f"Permanent fix: `pip install pip-system-certs`, then restart the UI.")
            return
    pipe = VideoPipeline(CFG, perception.homography, get_camera_publisher() if buzz else None)
    out_path = Path("data/processed") / f"{source.stem}_annotated.mp4"
    recorder = VideoRecorder(str(out_path), fps=(fps or 15) / stride)

    left, right = st.columns([3, 2], gap="large")
    view = left.empty()
    progress = left.progress(0.0, text="Analysing...")
    state, log = right.empty(), right.empty()
    total = max(1, int(cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FRAME_COUNT)) // stride)
    t0 = time.time()
    for i, (frame, t, dets) in enumerate(perception.stream()):
        annotated, info = pipe.step(frame, t, dets)
        recorder.write(annotated)
        for a in info.new_alerts:
            st.toast(f"{TIER_ICON[a.tier]} {a.tier.upper()} at {a.t:.1f} s" + (" · sent to phone" if buzz else ""))
        small = cv2.resize(annotated, (960, int(960 * annotated.shape[0] / annotated.shape[1])))
        view.image(small[:, :, ::-1], channels="RGB")
        progress.progress(min(1.0, (i + 1) / total), text=f"Analysing... {t:.1f} s")
        with state.container(border=True):
            st.markdown(f"**At {t:.1f} s** &nbsp; {badge(info.tier)}", unsafe_allow_html=True)
            k = st.columns(3)
            k[0].metric("Workers", info.workers)
            k[1].metric("Vehicles", info.vehicles)
            k[2].metric("Risk", f"{info.p_collision:.0%}")
        if pipe.alerts:
            log.dataframe([a.to_dict() | {"tier": f"{TIER_ICON[a.tier]} {a.tier}"} for a in pipe.alerts],
                          column_order=list(ALERT_COLUMNS), column_config=ALERT_COLUMNS, hide_index=True)
        if realtime:
            time.sleep(max(0.0, t - (time.time() - t0)))
    recorder.close()
    progress.empty()

    if pipe.alerts:
        first = pipe.alerts[0]
        st.error(f"**{len(pipe.alerts)} alerts, peak {pipe.peak.upper()}.** First alert at {first.t:.1f} s "
                 f"(risk {first.p_collision:.0%}, contact predicted in {first.ttc_s:.1f} s).", icon=":material/warning:")
    else:
        st.success("**No collision risk detected in this video.** If people and vehicles were visible but not "
                   "boxed, check the forklift_classes in config.yaml.", icon=":material/check_circle:")
    if out_path.exists():
        st.markdown("**Annotated video** (also saved to " + f"`{out_path}`)")
        st.video(str(out_path))
    next_link(NAV["warehouse"], "the whole warehouse at once")


# ---------------------------------------------------------------------------- 2c
def warehouse():
    from iwm.schema import TIERS
    from iwm.sim import WhatIf
    from iwm.warehouse import ZONES, WarehouseSession
    from iwm.warehouse_render import render_overview, render_zoom

    header("Step 4 · Warehouse", "Every blind corner at once",
           "Four forklifts and six workers share a warehouse with nine blind corners. The zoom panel follows "
           "the riskiest corner automatically, or pick a corner to watch it.")
    ss = st.session_state
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            react, mirror, limit = whatif_picker("wh_fix")
        seed = c2.number_input("Shift", 0, 10_000, 3, help="Change for a different shift (random start).")
        speed = c3.select_slider("Speed", options=[0.5, 1.0, 2.0, 4.0], value=1.0, format_func=lambda x: f"x{x:g}")
        z1, z2 = st.columns([4, 2])
        focus_pick = z1.segmented_control("Zoom", ["Auto"] + list(ZONES), default="Auto", key="wh_focus",
                                          help="Auto follows the riskiest corner; or pick one to pin the zoom.")
        focus_pick = focus_pick or "Auto"
        phone_to = z2.selectbox("Phone gets alerts for", ["Nobody", "All workers"] + [f"wk_{i}" for i in range(1, 7)],
                                index=1 if phone_ready() else 0, disabled=not phone_ready(),
                                help="Warning and Critical alerts go to the ntfy app (NTFY_TOPIC in .env).")
        b1, b2, b3 = st.columns([1, 1, 4])
        playing = ss.get("wh_playing", False)
        if b1.button("Pause" if playing else "Play", icon=":material/pause:" if playing else ":material/play_arrow:",
                     type="primary"):
            ss["wh_playing"] = playing = not playing
        reset = b2.button("Reset", icon=":material/restart_alt:")

    key = (seed, react, mirror, limit)
    if reset or ss.get("wh_key") != key or "wh" not in ss:
        ss["wh"] = WarehouseSession(CFG, int(seed), WhatIf(mirror=mirror, speed_limit=limit,
                                                           workers_react_to_alerts=react))
        ss["wh"].step()
        ss["wh_key"], ss["wh_auto"], ss["wh_contacts"], ss["wh_sent"] = key, None, [], {}
    s = ss["wh"]

    left, right = st.columns([3, 2], gap="medium")
    over_box, zoom_box = left.empty(), right.empty()
    k = st.columns(5)
    boxes = [col.empty() for col in k]
    log_box = st.empty()

    def focus_now():
        f = s.frame
        if focus_pick != "Auto":
            return ZONES[focus_pick], focus_pick
        hot, cur = s.hottest(), ss.get("wh_auto")
        if hot:
            cur_z = f.zones[cur[0][1]] if cur else None
            if cur_z is None or cur_z.tier == "none" or TIERS.index(hot.tier) > TIERS.index(cur_z.tier):
                ss["wh_auto"] = ((ZONES[hot.zone], hot.zone), f.t + 3.0)
            else:
                ss["wh_auto"] = (cur[0], f.t + 3.0)
        elif cur and f.t > cur[1]:
            ss["wh_auto"] = None
        return ss["wh_auto"][0] if ss.get("wh_auto") else None

    def show():
        f = s.frame
        focus = focus_now()
        over_box.image(render_overview(s, f, focus, ss["wh_contacts"]))
        if focus:
            zoom_box.image(render_zoom(s, f, focus))
        else:
            zoom_box.info("All corners clear. The zoom opens here when a collision is predicted, "
                          "or pick a corner above.", icon=":material/zoom_in:")
        st_ = s.stats()
        for box, (label, value) in zip(boxes, [("Time", f"{st_['t']:.0f} s"), ("Corner encounters", st_["encounters"]),
                                               ("Collisions", st_["contacts"]), ("Warnings", st_["warning"]),
                                               ("Critical", st_["critical"])]):
            box.metric(label, value)
        if s.alerts:
            log_box.dataframe([a.to_dict() | {"zone": s.alert_zone.get(id(a), ""),
                                              "tier": f"{TIER_ICON[a.tier]} {a.tier}",
                                              "pair": " / ".join(a.pair)} for a in s.alerts[-12:][::-1]],
                              column_order=["t", "zone", "tier", "pair", "p_collision", "ttc_s", "cause"],
                              column_config=ALERT_COLUMNS, hide_index=True)

    show()
    if not playing:
        st.caption("Press Play to run the shift. Change the zoom, fixes or speed at any time.")
    dt = CFG["sim"]["dt"]
    while playing and s.wh.t < 600:  # up to 10 minutes per shift
        t_frame = time.time()
        for _ in range(2):  # 0.2 s of warehouse time per screen update
            f = s.step()
            ss["wh_contacts"] = [c for c in ss["wh_contacts"] if f.t - c.t < 2.0] + f.new_contacts
            for a in f.new_alerts:
                if a.tier in ("warning", "critical"):
                    st.toast(f"{TIER_ICON[a.tier]} {a.tier.upper()} at {s.alert_zone.get(id(a), '')}: "
                             f"{a.pair[0]} / {a.pair[1]}")
                    wanted = phone_to == "All workers" or phone_to == a.pair[1]
                    if wanted and time.time() - ss["wh_sent"].get(tuple(a.pair), 0) > 4:
                        get_publisher().send(a)
                        ss["wh_sent"][tuple(a.pair)] = time.time()
        show()
        time.sleep(max(0.0, 2 * dt / speed - (time.time() - t_frame)))
    if playing and s.wh.t >= 600:
        ss["wh_playing"] = False
        st.success("Shift finished. Press Reset to start again.")
    next_link(NAV["videos"], "real video and simulation side by side")


# ----------------------------------------------------------------------------- 3
def demo_videos():
    from pathlib import Path

    header("Step 5 · Demo videos", "Real video and simulation, side by side",
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
    with st.container(border=True):
        st.markdown("**Warehouse overview · nine blind corners (pygame)**")
        wh = Path("data/warehouse_demo.mp4")
        if wh.exists():
            st.video(str(wh))
        else:
            st.caption("Not recorded yet. Record it with:")
            st.code("python demo_warehouse.py --record data/warehouse_demo.mp4", language="bash")
    next_link(NAV["results"], "the evidence from 1,000 runs")


# ----------------------------------------------------------------------------- 4
def results():
    header("Step 6 · Results", "The evidence: 1,000 simulated encounters",
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
    header("Step 7 · Shift reports", "What a site manager sees after a shift",
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
