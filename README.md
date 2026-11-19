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
| 3 · Alert | Alert tiers, MQTT, phone buzz, evaluation | — |
| 4 · Improve | Incident brief, what-if, dashboard | — |

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

All tunable numbers are in `config.yaml`; Azure keys go in environment variables (see `.env.example`).
