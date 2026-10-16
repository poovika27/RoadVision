"""Evaluate the pretrained RDD2022 YOLOv8m detector as a 3-class image classifier on our crops.

Rule: any pothole det -> pothole; else any other damage det -> crack; else normal.
Confidence threshold is tuned on VAL, then reported on TEST (no tuning on test).
Run inside the WSL venv:  ~/tfgpu/bin/python eval_yolo.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

ROOT = Path(__file__).parent
W = ROOT / "external/rdd2022-yolov8m/best.pt"
CLASSES = ["crack", "normal", "pothole"]
model = YOLO(str(W))
print("YOLO classes:", model.names)
is_pothole = {i for i, n in model.names.items() if "d40" in n.lower() or "pothole" in n.lower()}


def collect(split):
    paths, y = [], []
    for ci, c in enumerate(CLASSES):
        for p in sorted((ROOT / "data/processed" / split / c).glob("*.jpg")):
            paths.append(str(p)); y.append(ci)
    # per image: (max pothole conf, max other-damage conf)
    scores = []
    for i in range(0, len(paths), 16):  # small chunks keep memory low
        for r in model.predict(paths[i:i + 16], imgsz=640, conf=0.01, verbose=False, device="cpu"):
            pc = oc = 0.0
            for cls, cf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(cls) in is_pothole:
                    pc = max(pc, cf)
                else:
                    oc = max(oc, cf)
            scores.append((pc, oc))
        if i % 160 == 0:
            print(f"  {split}: {i}/{len(paths)}", flush=True)
    return np.array(y), np.array(scores)


def decide(scores, t):
    out = np.ones(len(scores), dtype=int)           # normal
    out[scores[:, 1] >= t] = 0                      # crack
    out[scores[:, 0] >= t] = 2                      # pothole (priority)
    return out


yv, sv = collect("val")
best_t, best_f = 0.25, -1
for t in np.arange(0.05, 0.8, 0.025):
    f = classification_report(yv, decide(sv, t), output_dict=True, zero_division=0)["macro avg"]["f1-score"]
    if f > best_f:
        best_t, best_f = float(t), f
print(f"best conf on val = {best_t:.3f} (macro-F1 {best_f:.4f})")

yt, st = collect("test")
pred = decide(st, best_t)
rep = classification_report(yt, pred, target_names=CLASSES, digits=4, zero_division=0)
print(rep); print(confusion_matrix(yt, pred))
(ROOT / "outputs/report_yolov8m_rdd2022.txt").write_text(f"conf={best_t:.3f}\n{rep}")
