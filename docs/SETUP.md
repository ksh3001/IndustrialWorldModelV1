# Running Industrial World Model locally

Commands are for **Windows PowerShell** (VS Code terminal). On macOS/Linux use `source .venv/bin/activate`
instead of `.venv\Scripts\Activate.ps1`. Run every command from the project folder.

## 1. Install once

- **Python 3.11 or 3.12** (python.org, tick "Add to PATH"). Avoid 3.13 for now.
- **Git**
- **Mosquitto** (phone buzz): Windows installer from mosquitto.org
- **ffmpeg** (browser-playable recordings): `winget install ffmpeg`

## 2. Get the code and set up Python

```powershell
git clone https://github.com/ksh3001/IndustrialWorldModelV1.git
cd IndustrialWorldModelV1
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt     # a few minutes: YOLO pulls in PyTorch
```

## 3. Check it works

```powershell
pytest -q                                   # 11 passed
python -m iwm.sim --scenario blind_corner   # prints contact_t=3.9
```

## 4. The UI

```powershell
streamlit run app.py                        # http://localhost:8501
```

The Results page works immediately (results are in the repo). Re-run them with `python -m iwm.evaluate --runs 1000`.

## 5. The pygame demo

```powershell
python demo_sim.py                                  # keys: A alerts, M mirror, S speed limit, R random, B demo
python demo_sim.py --record data/sim_demo.mp4       # also save it for the UI's Demo videos page
```

## 6. Real video

1. Film the corner with a phone mounted 2.5–3 m high; tape 4 floor markers and measure them (`floor_points` in `config.yaml`).
2. Copy the clip to `data\demo.mp4`.
3. `python calibrate.py` → click the 4 markers → paste `image_points` into `config.yaml`.
4. `python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4`

The first run downloads `yolo11n.pt`. YOLO has no forklift class: an RC car (`car`) or wheeled suitcase (`suitcase`)
works best as the stand-in; see `forklift_classes` in `config.yaml`.

## 7. Phone buzz (phone and laptop on the same Wi-Fi)

```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -c mosquitto.conf -v    # terminal 1
python -m http.server 8000                                             # terminal 2
python demo_sim.py --mqtt                                              # terminal 3 (or run_camera.py)
```

Find the laptop IP with `ipconfig`, then on the phone open
`http://<laptop-ip>:8000/phone.html?broker=<laptop-ip>` in Chrome and tap **Start**.
Allow Python and Mosquitto through Windows Firewall (ports 1883, 9001, 8000).

## 8. Azure OpenAI (optional)

Copy `.env.example` to `.env` and fill in endpoint, key and deployment. It is loaded automatically; the sidebar's
System status shows "Azure OpenAI: connected". Without it, briefs use a built-in template.
