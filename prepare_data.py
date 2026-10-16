"""Build a crop-based classification set (crack / normal / pothole) from the COCO detection dataset.

- pothole / crack: square crop around each box (+context)
- normal: random box-free patches + manhole crops (so manholes are not flagged as damage)
- split: by BLOCKS of consecutive frames (frames are timestamped video frames; a random split would leak)
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

SEED = 42
ROOT = Path(__file__).parent
SRC = ROOT / "data/raw/road-damage-dataset-potholes-cracks-and-manholes/data"
OUT = ROOT / "data/processed"
BLOCK = 15            # consecutive frames per block
SPLITS = [("train", 0.70), ("val", 0.15), ("test", 0.15)]
MIN_BOX, CONTEXT, SIZE = 24, 1.4, 224
MAX_BG_PER_IMG = 1

random.seed(SEED)
d = json.load(open(SRC / "annotations_coco.json"))
cat = {c["id"]: c["name"] for c in d["categories"]}
anns = defaultdict(list)
for a in d["annotations"]:
    anns[a["image_id"]].append(a)
imgs = sorted(d["images"], key=lambda i: i["file_name"])

# assign blocks of consecutive frames to splits
blocks = [imgs[i:i + BLOCK] for i in range(0, len(imgs), BLOCK)]
random.shuffle(blocks)
split_of, n = {}, len(blocks)
cut1, cut2 = int(n * SPLITS[0][1]), int(n * (SPLITS[0][1] + SPLITS[1][1]))
for bi, b in enumerate(blocks):
    s = "train" if bi < cut1 else "val" if bi < cut2 else "test"
    for im in b:
        split_of[im["id"]] = s

if OUT.exists():
    shutil.rmtree(OUT)
for s, _ in SPLITS:
    for c in ("crack", "normal", "pothole"):
        (OUT / s / c).mkdir(parents=True)


def square(cx, cy, side, W, H):
    side = int(min(side, W, H))
    x0 = int(min(max(cx - side / 2, 0), W - side))
    y0 = int(min(max(cy - side / 2, 0), H - side))
    return x0, y0, side


def overlaps(x0, y0, side, boxes):
    return any(x0 < bx + bw and x0 + side > bx and y0 < by + bh and y0 + side > by for bx, by, bw, bh in boxes)


stats = Counter()
for im in imgs:
    frame = cv2.imread(str(SRC / "images" / im["file_name"]))
    if frame is None:
        continue
    H, W = frame.shape[:2]
    s, stem = split_of[im["id"]], Path(im["file_name"]).stem
    boxes = [tuple(a["bbox"]) for a in anns[im["id"]]]

    def save(label, x0, y0, side, tag):
        crop = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / s / label / f"{stem}_{tag}.jpg"), crop)
        stats[(s, label)] += 1

    for k, a in enumerate(anns[im["id"]]):
        x, y, w, h = a["bbox"]
        if max(w, h) < MIN_BOX:
            continue
        name = cat[a["category_id"]]
        label = "normal" if name == "manhole" else name
        x0, y0, side = square(x + w / 2, y + h / 2, max(w, h) * CONTEXT, W, H)
        save(label, x0, y0, max(side, 64), f"{name}{k}")

    for t in range(MAX_BG_PER_IMG):
        for _ in range(40):
            side = random.randint(120, 300)
            x0, y0 = random.randint(0, W - side), random.randint(0, H - side)
            if not overlaps(x0, y0, side, boxes):
                save("normal", x0, y0, side, f"bg{t}")
                break

for k in sorted(stats):
    print(k, stats[k])
