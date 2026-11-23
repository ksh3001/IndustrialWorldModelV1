# Industrial World Model

**Warn workers 3–5 seconds before a forklift collision, using the cameras a site already has.**

Hackathon proof of concept by Team Future Frames (Industry & Energy theme).

Today's video AI describes the current frame, so it alerts when danger is already happening.
Industrial World Model tracks every worker and forklift, rolls the scene forward into many
possible futures, and buzzes the worker's phone *before* the paths cross.

```
Camera / Simulation ──► Tracks ──► World model (Kalman + Monte Carlo futures)
                                        │
                                        ▼
                         Alert engine (tiers + hysteresis) ──► MQTT ──► phone buzz
                                        │
                                        ▼
                         Dashboard (Streamlit) + Azure OpenAI incident brief
```

## Status

Built in stages, one commit per stage:

| Stage | What | Status |
|---|---|---|
| 0 | Scaffold: config, data format | done |
| 1 · See | Perception on video + warehouse simulation | done |
| 2 · Predict | World model: Kalman + Monte Carlo collision risk | done |
| 3 · Alert | Alert tiers, MQTT, phone buzz, evaluation | done |
| 4 · Improve | pygame live demo, UI with reports, incident briefs | done |

## Docs

| File | What |
|---|---|
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | The implementation plan this repo was built from, plus plan-vs-build notes |
| [docs/Industrial_World_Model_Detailed_Plan.pdf](docs/Industrial_World_Model_Detailed_Plan.pdf) | Full solution plan: problem, architecture, scope, demo script, judge Q&A |
| [docs/SETUP.md](docs/SETUP.md) | Run it on your laptop (Windows PowerShell, step by step) |
| [docs/architecture.png](docs/architecture.png) | System architecture diagram |
| [docs/ui-preview/](docs/ui-preview/) | Previews of the UI and the pygame window |

## Data format

Every module exchanges two records (metres, seconds):

```json
{"t": 12.4, "id": "fl_02", "type": "forklift", "x": 14.2, "y": 6.8, "vx": 2.1, "vy": -0.3}
{"t": 12.4, "pair": ["fl_02", "wk_07"], "tier": "warning", "p_collision": 0.68, "ttc_s": 2.7}
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Stage 1 · See

```bash
python -m iwm.sim --scenario blind_corner --plot   # top-down run -> runs/sim.png
python calibrate.py --source data/demo.mp4          # click 4 floor markers -> image_points
```

- `iwm/sim.py`: 20 x 12 m warehouse, one blind corner, a forklift and a walker. Emits the same
  track records as the camera, plus the true contact time. Scenarios: `blind_corner`, `crossing`,
  `parallel`, `random`.
- `iwm/perception.py`: YOLOv11 + ByteTrack on video; box bottom-centre mapped to floor metres
  with a homography from 4 taped markers.

## Stage 2 · Predict

- `iwm/worldmodel.py`: a constant-velocity Kalman filter smooths each track and estimates its
  velocity. For every worker-forklift pair, 50 futures are rolled 5 s ahead with noise on speed and
  heading and a chance the walker stops. `p_collision` = share of futures within 1.5 m; `ttc_s` =
  median time of first contact.
- `explain()` re-runs the futures with one factor changed (forklift slows, worker stops, forklift
  stops) and ranks which change lowers risk most.
- `detection_only()` is the baseline we compare against: alert only when a pair is already < 2 m apart.

```bash
pytest -q
```

## Stage 3 · Alert

- `iwm/alerts.py`: Advisory / Warning / Critical tiers with hysteresis (3 rising frames to escalate,
  1 s below a threshold to drop). Alerts go to MQTT topic `iwm/alerts`, `logs/alerts.jsonl`, and a laptop beep.
- Phone buzz, two options: `phone.html` (local, via Mosquitto: open on an Android phone, tap Start) or the free
  **ntfy** Android app (set `NTFY_TOPIC` in `.env`; no broker or IP needed). See docs/SETUP.md.
- `run_camera.py`: the full live pipeline on real video, with predicted paths drawn on the frame.
- `iwm/evaluate.py`: 1,000 simulated runs, world model vs detection-only, plus what-if fixes.

```bash
mosquitto -c mosquitto.conf -v                      # terminal 1: broker (1883 + WebSocket 9001)
python -m http.server 8000                          # terminal 2: open http://<laptop-ip>:8000/phone.html?broker=<laptop-ip> on the phone
python run_camera.py --source data/demo.mp4         # terminal 3: live demo on real video
python -m iwm.evaluate --runs 1000                  # results -> results/results.json, results/results.png
```

### Results (1,000 simulated runs, 329 with contact)

| | Detection-only (< 2 m now) | World model |
|---|---|---|
| Contacts flagged before they happen | 100% (at the last moment) | 99% |
| Average warning before contact | 0.3 s | 2.2 s (Warning) · 2.9 s (first alert) |
| False alarms per simulated hour | 33 | 16 |

Crossing scenarios that end in contact: **47%** with no fix, 39% with a convex mirror, 42% with a
1 m/s speed limit, and **10%** when the worker and driver react to world-model alerts.

![results](results/results.png)

## Stage 4 · Improve

```bash
python demo_sim.py --mqtt      # live pygame demo (phone buzzes too); keys: A alerts, M mirror, S speed limit, R random, B demo
streamlit run app.py           # the demo UI
python -m iwm.report           # incident brief for the blind-corner near-miss
```

- `demo_warehouse.py`: **warehouse overview** in pygame. A 40 x 24 m site with 3 forklift aisles, 3 walkways and 9
  blind intersections (A1..C3); 4 forklifts and 6 workers moving at once. Each intersection is ringed by its live
  risk; when a collision is predicted a **zoomed-in pop-up** opens on that corner (the two people, predicted paths,
  risk, time to contact, top fix). Click any corner or point (or keys 1-9) to pin the zoom; Esc / 0 for auto.
  Keys A / M / S switch alerts, mirrors, speed limit; alerts go to the phone (`--phone wk_1` = only that worker).
  Over 15 simulated minutes: 30% of corner encounters end in contact with no fix, 18% with a speed limit, 6% when
  alerts reach the worker and driver, 3% with mirrors at every corner.
- `demo_sim.py`: the single blind-corner demo. Top-down warehouse in pygame with racking, forklift, walker,
  5 s predicted paths, risk panel, alert log, and what-if keys.
- `app.py` + `ui/`: the demo UI. The sidebar groups pages the way you present them, shows where you
  are in the presentation flow, and shows system status (results, Azure OpenAI, camera log):
  - **Present:** *Overview* (problem, headline numbers, how it works) · *Live demo* (scenario
    picker, what-if fix pills, outcome banner, time slider / Play, risk and time-to-contact at each moment, alerts,
    "why", incident brief, phone buzz on Play) · *Real video* (upload or pick a clip; YOLO + world model run frame by
    frame, predicted paths drawn live, alerts sent to the worker's phone as they happen, annotated video saved)
    · *Warehouse* (the whole warehouse in the browser: 9 blind corners, 4 forklifts, 6 workers; zoom panel follows
    the riskiest corner or a corner you pick; what-if fixes, Play / Pause / speed, counters, alert log, phone alerts)
    · *Demo videos* (real video and pygame recording side by side)
  - **Evidence:** *Results* (1,000-run comparison and what it means) · *Shift reports* (KPIs, then Summary,
    Incidents and Export tabs with hotspot map, executive summary, incident briefs, .md/.csv downloads)
  - **Sources:** *Camera alerts* (alerts from real video and connection steps)
  - Every page ends with a **Next** link, so you can click straight through the demo.
- `iwm/report.py`: Azure OpenAI incident briefs and shift summaries (set the variables in `.env.example`),
  with a template fallback so the demo never breaks.

## Showing the demos in the UI

pygame and the camera loop each open their own window, so record them once and the UI plays them:

```bash
python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4   # annotated real video
python demo_sim.py --mqtt --record data/sim_demo.mp4                      # pygame demo (press A mid-run)
streamlit run app.py                                                      # -> Demo videos page
```

Recordings are converted to H.264 with the ffmpeg bundled by `imageio-ffmpeg`, so the browser can play
them. On stage, use the UI as the main screen and keep the live pygame window as the "it's real" moment.

## 3-minute demo

1. **Real video (60 s):** `run_camera.py` on your phone footage. Tracks and predicted paths appear, and the phone buzzes before the "forklift" reaches the corner.
2. **Live simulation (45 s):** `demo_sim.py --mqtt`, blind corner. Contact at 3.9 s; the first alert came 3.1 s earlier, and detection-only fired 0.2 s before.
   Press **A**: the worker and driver react to the alert and there is no contact.
3. **Results (30 s):** UI → Results: 1,000 runs, 2.9 s vs 0.3 s warning, contacts 47% → 10%.
4. **Reports (30 s):** UI → Reports: shift KPIs, hotspots, Azure OpenAI summary, download.
5. **Close (15 s):** "Ravi feels a buzz four seconds before the forklift turns the corner."

## Repo map

```
iwm/
  schema.py      Track and Alert records
  sim.py         blind-corner warehouse simulation + what-if rules
  perception.py  YOLOv11 + ByteTrack + homography
  worldmodel.py  Kalman + Monte Carlo futures + counterfactual explain
  alerts.py      tiers, hysteresis, MQTT publisher
  evaluate.py    1,000-run evaluation
  demo.py        one session (shared by pygame demo and UI)
  warehouse.py   whole-warehouse simulation + session (demo_warehouse.py, UI Warehouse page)
  warehouse_render.py  Pillow drawing of the warehouse for the UI
  pipeline.py    real-video pipeline (UI Real video page, run_camera.py)
  shift.py       shift report: incidents, summary, exports
  report.py      Azure OpenAI briefs with template fallback
  plots.py       figures for the UI
app.py           Streamlit UI         demo_sim.py   pygame single-corner demo
demo_warehouse.py pygame warehouse overview with zoom pop-ups
run_camera.py    real-video pipeline  calibrate.py  camera calibration
phone.html       phone "wristband"    mosquitto.conf
```

All tunable numbers are in `config.yaml`; Azure keys go in environment variables (see `.env.example`).
