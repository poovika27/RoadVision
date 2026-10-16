"""Classify frames of a recorded video and write an annotated copy.

Usage: python video_infer.py input.mp4 [output.mp4] [--every N]
"""
import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("output", nargs="?", default=str(ROOT / "outputs" / "annotated.mp4"))
ap.add_argument("--every", type=int, default=3, help="run the model on every Nth frame")
ap.add_argument("--smooth", type=int, default=5, help="average predictions over last K inferences")
args = ap.parse_args()

model = tf.keras.models.load_model(ROOT / "models" / "final.keras", compile=False)
classes = json.loads((ROOT / "models" / "classes.json").read_text())
COLORS = {"normal": (0, 200, 0), "pothole": (0, 0, 255), "crack": (0, 165, 255)}

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise SystemExit(f"Cannot open {args.video}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
w, h = int(cap.get(3)), int(cap.get(4))
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

recent, label, conf, i = deque(maxlen=args.smooth), "...", 0.0, 0
counts = {c: 0 for c in classes}
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if i % args.every == 0:
        rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        recent.append(model.predict(rgb[None].astype("float32"), verbose=0)[0])
        p = np.mean(recent, 0)
        label, conf = classes[int(p.argmax())], float(p.max())
        counts[label] += 1
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"{label.upper()} {conf:.0%}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    writer.write(frame)
    i += 1

cap.release(); writer.release()
print("Saved", args.output, "| inference counts:", counts)
