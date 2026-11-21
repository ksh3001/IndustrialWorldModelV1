# Running Industrial World Model locally

Commands are for **Windows PowerShell** (VS Code terminal). On macOS/Linux use `source .venv/bin/activate`
instead of `.venv\Scripts\Activate.ps1`. Run every command from the project folder.

## 1. Install once

- **Python 3.11 or 3.12** (python.org, tick "Add to PATH"). Avoid 3.13 for now.
- **Git**
- **Mosquitto** (phone buzz): Windows installer from mosquitto.org
- ffmpeg is bundled through pip (`imageio-ffmpeg`), nothing extra to install

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

## 5b. The warehouse overview (pygame)

```powershell
python demo_warehouse.py                                   # click a corner to zoom; A / M / S what-ifs
python demo_warehouse.py --phone wk_1                      # phone gets only worker wk_1's alerts
python demo_warehouse.py --record data/warehouse_demo.mp4  # for the UI's Demo videos page
```

## 6. Real video

**In the UI (easiest):** open **Real video**, upload a clip (or pick one from `data/`), choose
*Approximate* distances and the scene width (e.g. 15 m), and press **Analyse video**. The video plays with tracking
and predicted paths; alerts go to the phone (ntfy) as they happen and the annotated video is saved to `data/processed/`.

**From the terminal:**

1. Film the corner with a phone mounted 2.5–3 m high; tape 4 floor markers and measure them (`floor_points` in `config.yaml`).
2. Copy the clip to `data\demo.mp4`.
3. `python calibrate.py` → click the 4 markers → paste `image_points` into `config.yaml`.
4. `python run_camera.py --source data/demo.mp4 --save data/camera_demo.mp4`
   (no calibration? add `--approx 15` for a frame about 15 m wide)

The first run downloads `yolo11n.pt`. On an office network that blocks it (`CERTIFICATE_VERIFY_FAILED`), download
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt in your browser and put it in the project
folder, or run `pip install pip-system-certs` so Python trusts the company certificate. YOLO has no forklift class: an RC car (`car`) or wheeled suitcase (`suitcase`)
works best as the stand-in; see `forklift_classes` in `config.yaml`.

## 7. Phone buzz via phone.html + Mosquitto (phone and laptop on the same Wi-Fi)

```powershell
& "C:\Program Files\mosquitto\mosquitto.exe" -c mosquitto.conf -v    # terminal 1
python -m http.server 8000                                             # terminal 2
python demo_sim.py --mqtt                                              # terminal 3 (or run_camera.py)
```

Test it first: `python -m iwm.alerts warning` sends one test alert (the phone should vibrate).

Find the laptop IP with `ipconfig`, then on the phone open
`http://<laptop-ip>:8000/phone.html?broker=<laptop-ip>` in Chrome and tap **Start**.
Allow Python and Mosquitto through Windows Firewall (ports 1883, 9001, 8000).

## 7b. Phone buzz the easy way: the ntfy app (no Mosquitto, no IP addresses)

Both the laptop and the phone just need internet.

1. **Phone:** install **ntfy** from the Play Store. Tap **+**, enter a hard-to-guess topic name
   (e.g. `iwm-futureframes-7391`), tap **Subscribe**. Allow notifications.
2. **Laptop:** add the same name to `.env`: `NTFY_TOPIC=iwm-futureframes-7391`
3. **Test:** `python -m iwm.alerts warning`. The phone shows "WARNING: collision risk" with a long vibration.
   `advisory` = normal buzz, `critical` = urgent, long vibration bursts.
4. **Demo:** `python demo_sim.py` (and `run_camera.py`) now push alerts to the phone automatically.

Notes: ntfy goes over the internet, so expect roughly a second of delay (the local `phone.html` route is faster).
Some office networks block ntfy.sh: use the phone's hotspot. Anyone who knows the topic name can read the alerts,
so don't use an obvious one.

## 8. Azure OpenAI (optional)

Copy `.env.example` to `.env` and fill in endpoint, key and deployment. It is loaded automatically; the sidebar's
System status shows "Azure OpenAI: connected". Without it, briefs use a built-in template.
