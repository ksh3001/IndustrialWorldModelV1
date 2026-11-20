# Sample CCTV clips

Real factory security-camera clips for testing the camera pipeline (`run_camera.py`).

## Source and license

**Video Dataset for Safe and Unsafe Behaviours** — Oğuzhan Önal and Emre Dandıl, Bilecik Şeyh Edebali University;
footage shared by Kafaoğlu Metal Plastik Makine San. ve Tic. A.Ş., Eskişehir, Turkey.
Mendeley Data, DOI [10.17632/xjmtb22pff.1](https://data.mendeley.com/datasets/xjmtb22pff/1) ·
Hugging Face mirror: [Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours).
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Paper: Önal & Dandıl, *Video dataset for the
detection of safe and unsafe behaviours in workplaces*, Data in Brief 56 (2024) 110791.

Clips are 1920×1080, 24 fps, 1–20 s, unmodified unless noted below.

## Clips in this folder

| File | Original class | What it shows |
|---|---|---|
| _add rows as you add clips_ | | |

## Use one

```powershell
python run_camera.py --source data/samples/<clip>.mp4 --save data/camera_demo.mp4
```

These cameras weren't calibrated with floor markers: run `python calibrate.py --source data/samples/<clip>.mp4`, click
4 floor points you can estimate (walkway line corners; a walkway is roughly 1–1.5 m wide) and put the estimated
metres in `config.yaml` → `floor_points`. YOLO has no forklift class, so real forklifts may show as `truck`/`car` or be
missed.
