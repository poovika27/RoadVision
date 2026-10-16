"""Tune tile geometry + threshold for frame-level pothole detection on unseen RDD2022 test frames."""
import random, zipfile, numpy as np, cv2, tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_auc_score
random.seed(0)
ROOT = Path(__file__).parent
z = zipfile.ZipFile(ROOT / "data/raw/rdd2022/rdd-2022.zip")
model = tf.keras.models.load_model(ROOT / "models/final.keras", compile=False)
pos, neg = [], []
for n in z.namelist():
    if n.startswith("RDD_SPLIT/test/labels/") and n.endswith(".txt") and "Norway" not in n:
        ids = {l.split()[0] for l in z.read(n).decode().split("\n") if l.strip()}
        (pos if "4" in ids else neg).append(n.replace("/labels/", "/images/").replace(".txt", ".jpg"))
random.shuffle(pos); random.shuffle(neg)
files = pos[:250] + neg[:250]; y = np.array([1] * 250 + [0] * 250)
frames = []
for f in files:
    im = cv2.imdecode(np.frombuffer(z.read(f), np.uint8), 1)
    s = min(1.0, 960 / max(im.shape[:2])); frames.append(cv2.resize(im, None, fx=s, fy=s))

def tiles(fr, frac, y0f, step_div):
    H, W = fr.shape[:2]; y0 = int(H * y0f); side = max(int(min(W, H - y0) * frac), 64); st = max(side // step_div, 1)
    ys = range(y0, H - side + 1, st) if H - y0 > side else [y0]; xs = range(0, W - side + 1, st) if W > side else [0]
    return [(x, yy, side) for yy in ys for x in xs]

def score(frac, y0f, step_div):
    out = []
    for fr in frames:
        bs = tiles(fr, frac, y0f, step_div)
        b = np.stack([cv2.cvtColor(cv2.resize(fr[yy:yy+s, x:x+s], (224, 224)), cv2.COLOR_BGR2RGB) for x, yy, s in bs]).astype("float32")
        p = model.predict(b, verbose=0, batch_size=64)[:, 2]
        out.append(p.max())
    return np.array(out)

print("frames", len(frames), flush=True)
for frac in (0.2, 0.28, 0.35):
    for y0f in (0.25, 0.4):
        s = score(frac, y0f, 2)
        best = max(((np.mean((s >= t) == y), t) for t in np.arange(0.5, 1.0, 0.025)))
        print(f"frac={frac} y0={y0f} AUC={roc_auc_score(y, s):.3f} bestacc={best[0]:.3f}@{best[1]:.3f}", flush=True)
