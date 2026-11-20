# Industrial World Model — Detailed Implementation Plan

Team Future Frames · 19 Nov 2026 · the plan this repository was built from.
Source doc: https://claude.ai/artifact/FGMBefxbRgFxLrW4jqVLVg

## 1. Overview

This is a hackathon proof of concept, not a production system: in 4 days we show that a world model can warn a
worker at least 3 seconds before a forklift collision, using CCTV-style video and a phone buzz.

**What the prototype proves**

- **Perception works on real video:** workers and a stand-in forklift are tracked live from a phone mounted like a CCTV camera.
- **Prediction works:** the world model forecasts collisions 3–5 s ahead, measured against ground truth in simulation.
- **Alerts are fast and trusted:** a phone buzz and a laptop beep fire in under half a second, with few false alarms.
- **Sites can improve:** a simple what-if toggle and an Azure OpenAI incident brief show the bigger vision.

**Guiding principles**

1. **One data contract.** Real video and simulation emit the same track format, so the world model never knows the difference.
2. **Baseline first, then learn.** Constant velocity + Monte Carlo is the POC model; an LSTM is a stretch goal only if time allows.
3. **Keep it local.** Everything runs on one laptop; Azure is used only for the incident brief.
4. **Privacy-first.** Positions and motion only; no faces stored, no individual scoring.
5. **Always demoable.** Every day ends with something that runs; record a backup video of the demo.

## 2. Environment & setup

Keep the setup minimal: one laptop, two phones, a trolley and a single Python project.

**Hardware**

| Item | Use |
| --- | --- |
| Laptop (GPU helps, not required) | Runs everything; use YOLOv11n on CPU |
| Phone 1 on a tripod or taped high | The "CCTV" camera: record clips or stream over RTSP |
| Phone 2 | The worker's "wristband": a web page that vibrates on alert |
| Trolley, cart or RC car with a bright box | Stand-in forklift |
| 4 taped floor markers | Camera calibration; measure their positions in metres |
| ESP32 + vibration motor (stretch) | A real wristband, only if time allows |

**Software**

- **Python 3.11** with ultralytics (YOLOv11 + built-in ByteTrack), OpenCV, NumPy.
- **pygame** for the simulation.
- **paho-mqtt + Mosquitto** to send alerts to the phone (MQTT over WebSocket).
- **Streamlit** for the dashboard; no separate frontend.
- **Azure OpenAI** for the incident brief. Azure IoT Hub is optional, for the architecture story.

**Repo and conventions**

```
iwm/
├─ perception.py   # YOLOv11 + ByteTrack + homography -> tracks
├─ sim.py          # pygame warehouse -> tracks + ground truth
├─ worldmodel.py   # Kalman smoothing + Monte Carlo risk
├─ alerts.py       # tiers, hysteresis, MQTT publish
├─ phone.html      # vibrates on alert (phone 2)
├─ report.py       # Azure OpenAI incident brief
├─ app.py          # Streamlit dashboard
└─ config.yaml     # thresholds, horizon, keys
```

- Tunable numbers live in `config.yaml`; keys come from environment variables and are never committed.
- Modules pass plain lists of track dictionaries (section 3), nothing more.
- `main` always runs; merge small changes often.

## 3. Data format

Two small JSON records are the only contract between modules; agree on them in the first hour.

**Track** (one per object per frame, from the camera or the simulation; metres and seconds):

```json
{"t": 12.4, "id": "fl_02", "type": "forklift", "x": 14.2, "y": 6.8, "vx": 2.1, "vy": -0.3}
```

**Alert** (from the world model to the phone and dashboard):

```json
{"t": 12.4, "pair": ["fl_02", "wk_07"], "tier": "warning", "p_collision": 0.68, "ttc_s": 2.7}
```

- `type` is `worker` or `forklift`; `tier` is `advisory`, `warning` or `critical`.
- `ttc_s` is the predicted time to contact in seconds.
- Alerts go out on one MQTT topic, `iwm/alerts`; the phone page and the dashboard both subscribe to it.

## 4. What to build (6 modules)

Six small Python files make the whole POC; each is useful on its own and testable in an hour.

### 4.1 Perception — `perception.py`

- Run `YOLO("yolo11n.pt").track(source, tracker="bytetrack.yaml")` on the phone video; keep `person` plus the stand-in
  forklift (a bright box, or the `truck`/`car` class).
- Map each box's bottom-centre to floor metres with `cv2.findHomography` from the 4 taped markers.
- Velocity = change in position over the last 0.5 s; emit track records.
- **Done when:** the video shows boxes, IDs and speeds in m/s that look right when someone walks a known distance.

### 4.2 Simulation — `sim.py`

- A 20 × 12 m top-down warehouse with two aisles, one blind corner, 1–2 forklifts and 3–4 walkers on scripted paths
  with random speed and pauses.
- Each step writes the same track records, plus the true time of any contact (distance < 1.5 m).
- A `--scenario blind_corner` flag replays the demo case with a fixed seed.
  - **Live pygame demo (`demo_sim.py`):** draws the simulation top-down in real time (racking, forklift, walker,
    5 s predicted paths, risk panel, alert log) with what-if keys (A alerts, M mirror, S speed limit). This is the
    on-stage demo; the logic stays plain Python so 1,000 evaluation runs finish in about a minute.
- **Done when:** 1,000 runs generate in a few minutes with ground-truth contact times.

### 4.3 World model — `worldmodel.py`

- Smooth each track with a simple Kalman filter (position + velocity).
- For each worker–forklift pair, roll 50 futures forward 5 s in 0.1 s steps, adding noise to speed and heading and a
  chance that a walker stops.
- `p_collision` = share of futures where the pair comes within 1.5 m; `ttc_s` = median time of first contact.
- **Done when:** the blind-corner scenario shows risk rising seconds before contact, and parallel paths stay low.

### 4.4 Alerts — `alerts.py` + `phone.html`

| Tier | Rule | Action |
| --- | --- | --- |
| Advisory | p > 0.3 and ttc < 5 s | Short buzz |
| Warning | p > 0.6 and ttc < 3 s | Long buzz + laptop beep |
| Critical | p > 0.85 and ttc < 2 s | Continuous buzz + beep |

- Raise a tier only after 3 rising frames; drop it after 1 s below the threshold, so alerts don't flicker.
- Publish to `iwm/alerts`; `phone.html` (open in Chrome on Android) subscribes over WebSocket and calls `navigator.vibrate()`.
- **Done when:** the phone buzzes at least 3 s before contact in the blind-corner scenario.

### 4.5 Report — `report.py`

- After a near-miss, send the alert record plus 2–3 facts (speeds, location, tier) to Azure OpenAI with a short prompt
  asking for a 5-line incident brief and one suggested fix.
- **Done when:** a readable brief appears for the demo near-miss.

### 4.6 Dashboard — `app.py` (Streamlit)

- Left: live video or simulation view with predicted paths. Right: risk gauge, alert log, latest brief.
- One what-if toggle: "forklift speed limit 1 m/s near the corner" re-runs the scenario and shows the risk drop.
- **Done when:** the whole demo runs from this one page.

**Stretch goals (only if ahead of schedule):** an LSTM trained on simulated tracks compared against the baseline; an
ESP32 wristband with left/right motors; Azure IoT Hub for events.

## 5. 4-day schedule

Four people work in parallel from hour one and join their pieces at the end of Day 2; Day 4 is only polish and rehearsal.

| Day | A · Vision | B · Sim + world model | C · Alerts + phone | D · Dashboard + pitch |
| --- | --- | --- | --- | --- |
| **Day 1** | Record 3–5 clips at a blind corner; YOLO + ByteTrack running | Agree data format; pygame warehouse with walkers and a forklift | Mosquitto running; `phone.html` vibrates on a test message | Streamlit skeleton; Azure OpenAI key working |
| **Day 2** | Homography; track records from real video | Kalman + Monte Carlo risk on simulated tracks | Tiers + hysteresis on fake alerts | Show video, risk gauge, alert log |
| **Day 3** | Run world model on real tracks; tune thresholds | 1,000-run evaluation: lead time, catch rate, false alarms | End-to-end: camera → risk → phone buzz; measure delay | Incident brief; what-if toggle |
| **Day 4** | Record the final demo clip + backup video | Results chart for the slide | Fix bugs; test on the demo phone | Pitch deck; rehearse 3 times |

**Checkpoints**

- [ ] End of Day 1: each person's piece runs on its own.
- [ ] End of Day 2: risk score shows on simulated **and** real tracks.
- [ ] End of Day 3: phone buzzes before contact, end to end.
- [ ] End of Day 4: full demo rehearsed; backup video recorded.

## 6. How we know it works

Four numbers from the simulation runs go on the results slide; the live demo proves the same thing on real video.

| Metric | How we measure it | POC target |
| --- | --- | --- |
| Warning lead time | Seconds between first Warning alert and true contact | ≥ 3 s on average |
| Near-misses caught | Share of simulated contacts that got an alert first | ≥ 90% |
| False alarms | Warnings with no contact, per simulated hour | Low; report the number honestly |
| Alert delay | Camera frame to phone buzz | Under 0.5 s |

**Comparison for the slide:** a detection-only rule (alert when distance < 2 m right now) versus our world model on
the same 1,000 runs. The gap in lead time is the headline.

**Quick checks before the demo**

- Walk a measured 5 m in front of the camera; the tracked distance is within about 0.5 m.
- Parallel paths in the simulation should not trigger Warnings.
- Run the blind-corner scenario 10 times; the phone buzzes every time before contact.

## 7. Demo, risks and done

The demo is 3 minutes, with a recorded backup video in case anything fails live.

**Demo flow**

1. **Real video (60 s):** a teammate walks towards the blind corner while the "forklift" approaches; the predicted
   paths cross and the phone buzzes about 4 s before they meet.
2. **Simulation (45 s):** run the pygame demo live with `--mqtt` so the phone buzzes too, then say "We can't crash
   real forklifts, so we tested 1,000 scenarios." Show lead time, catch rate and false alarms versus detection-only.
3. **What-if (30 s):** turn on the speed limit near the corner; risk drops.
4. **Report (15 s):** show the Azure OpenAI incident brief.
5. **Close (30 s):** "Ravi feels a buzz four seconds before the forklift turns the corner."

**Risks and fallbacks**

| Risk | Fallback |
| --- | --- |
| Stand-in forklift not detected | Bright box or ArUco marker on it; or detect it as `car` |
| Real-video speeds are noisy | Longer smoothing window; keep the camera fixed and high |
| Phone doesn't vibrate | On-screen red flash plus laptop beep |
| Azure OpenAI unavailable | Pre-generated brief for the demo near-miss |
| Live demo glitch | Play the recorded backup video |

**Definition of done**

- [ ] Live tracking on real CCTV-style video
- [ ] Phone buzzes ≥ 3 s before contact in the blind-corner scenario
- [ ] Results slide with lead time, catch rate and false alarms versus detection-only
- [ ] What-if toggle and incident brief working on the dashboard
- [ ] Backup video recorded and the demo rehearsed 3 times

---

## Appendix: plan vs what was built

| Plan | Built | Why |
| --- | --- | --- |
| `sim.py` drawn in pygame | `iwm/sim.py` is plain Python; `demo_sim.py` draws it in pygame | 1,000 evaluation runs stay fast; pygame is only the stage view |
| One-page Streamlit dashboard | Six-page UI (`app.py` + `ui/`) with grouped sidebar navigation | Added Demo videos, Shift reports and Camera pages on request |
| Single what-if toggle | Three fixes: alerts reach worker and driver, convex mirror, speed limit | Shows which fix matters; simulation models driver braking and pedestrian priority |
| Explain the risk | Counterfactual `explain()` mapped to a practical site fix | Gives the "why" and a recommendation in each brief |
| Results target ≥ 3 s (Warning tier) | 2.2 s at Warning tier, 2.9 s at first alert, vs 0.3 s detection-only | Warning tier requires ttc < 3 s, so its lead is capped near 3 s; first alert includes Advisory |
| Stretch goals | Not built: LSTM, ESP32 wristband, Azure IoT Hub | Kept the POC simple |
