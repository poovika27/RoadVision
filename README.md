# Pothole & Road Damage Detection (CNN)

Upload a road video and the system flags **potholes** and **cracks** using an EfficientNetV2-S image classifier
(TensorFlow/Keras, transfer learning from ImageNet). Supports SDG 11 (Sustainable Cities and Communities).

## Accuracy

3 classes: `crack` / `normal` / `pothole`. Test sets were never used for training or tuning.

| Test set | Images | Accuracy | Macro-F1 |
|---|---|---|---|
| RDD2022 test crops | 3,408 | **90.0%** | 0.901 |
| Original dashcam test crops (harder, different camera/region) | 1,150 | **77.4%** | 0.757 |

Per class on RDD2022: crack F1 0.891, normal 0.902, pothole 0.910.
Per class on the original set: crack 0.785, normal 0.825, pothole 0.661.
Full reports and confusion matrices: `outputs/`.

How we got there (original test accuracy): MobileNetV2 59% -> EfficientNetV2-S 67% -> + RDD2022 data and full fine-tuning 77%.
A pretrained RDD2022 YOLOv8m used as a classifier scored only 44.5% on the same crops, so it was not used.

**Video-level:** the website works on tiles of video frames, so its real-world accuracy differs from the crop numbers above.
On unseen RDD2022 frames, frame-level pothole detection was about 75% accurate; the video verdict needs several frames to agree.

## Quick start (run the website)

```bash
git clone https://github.com/mourishantony/pothole-cnn.git
cd pothole-cnn
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, choose a video (mp4/avi/mov/mkv/webm, up to 300 MB) and click **Analyse video**.
You get a verdict (pothole detected / not), an annotated video (red = pothole, orange = crack) and the strongest pothole frame.
Pretrained weights are included: `models/final.keras` (83 MB) with `models/classes.json`.
On Windows with WSL2 + GPU you can instead double-click `start_app.bat`.

**Speed:** on CPU a 30-second clip takes a few minutes. With an NVIDIA GPU it takes about a minute (TensorFlow has no native
Windows GPU support after 2.10, so on Windows use WSL2 + `pip install "tensorflow[and-cuda]"`).

### Command line
```bash
python video_infer.py input.mp4 outputs/annotated.mp4     # whole-frame labels on a video
```

### How the website decides
1. Samples a frame every 0.5 s and splits the lower 75% (road area) into overlapping tiles (30% of the road area).
2. Classifies each tile; a tile counts if pothole probability >= 0.95 (crack >= 0.90).
3. Neighbouring tiles merge into one box; a pothole needs at least 2 agreeing tiles.
4. "Pothole detected" if at least 3 sampled frames contain one. Tune `THRESHOLDS`, `TILE_FRAC`, `MIN_TILES`, `MIN_FRAMES` in `app.py`.

## Retrain

```bash
pip install -r requirements-train.txt
python download_data.py lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes   # needs a Kaggle API token (~/.kaggle/kaggle.json)
python prepare_data.py            # crops from the road-damage dataset
python download_data.py aliabdelmenam/rdd-2022      # 10.6 GB
python prepare_rdd.py             # builds data/processed_v2 (original + RDD2022 crops)
DATA_DIR=processed_v2 BACKBONE=efficientnetv2s TAG=effv2s_v2 HEAD_EP=6 FT_EP=20 FT_LAYERS=120 python train.py
DATA_DIR=processed_v3 BACKBONE=efficientnetv2s TAG=effv2s_v3 RESUME=best_effv2s_v2.keras FT_EP=15 FT_LAYERS=600 FT_LR=3e-5 python train.py
```
`processed_v3` is `processed_v2` with the original-domain training crops duplicated 3x.
Datasets: [Road Damage: Potholes, Cracks and Manholes](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes),
[RDD2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022). Check their licenses before commercial use.

## Troubleshooting
- `could not be deserialized ... input_axes`: the weights file was saved by an older Keras than the one you run. `final.keras` is saved with Keras 3.12, which newer Keras versions load fine; if you see this error you are running an older Keras, so `pip install -U keras`.

## Limits
- Trained on crops, not full frames; heavily cracked pavement can be flagged as a pothole. Not tested on plain-road video.
- Decision support only: results need human review before maintenance decisions.
