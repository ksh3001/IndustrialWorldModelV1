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
| 1 · See | Perception on video + warehouse simulation | — |
| 2 · Predict | World model: Kalman + Monte Carlo collision risk | — |
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

All tunable numbers are in `config.yaml`; Azure keys go in environment variables (see `.env.example`).
