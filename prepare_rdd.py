"""Build data/processed_v2 = original crops (train/val) + RDD2022 crops; two separate test sets.

RDD2022 YOLO class ids (verified visually): 0/1/2 = cracks, 3 = road-marking blur (not damage -> normal),
4 = pothole.  Normal also gets box-free background patches.
Outputs: processed_v2/{train,val,test_rdd,test_orig}/{crack,normal,pothole}
"""
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
ZIP = ROOT / "data/raw/rdd2022/rdd-2022.zip"
ORIG = ROOT / "data/processed"
OUT = ROOT / "data/processed_v2"
SIZE, CONTEXT, MIN_PX = 224, 1.4, 20
# per-split quotas (crack, pothole, marking->normal, background->normal)
QUOTA = {"train": (6000, 99999, 2500, 3500), "val": (1300, 99999, 500, 700), "test_rdd": (1300, 99999, 500, 700)}
SPLIT_SRC = {"train": "train", "val": "val", "test_rdd": "test"}
random.seed(42)

_z = None


def work(job):
    """job = (zip image name, out dir, stem, [(label, cx, cy, w, h)], n_bg)"""
    global _z
    if _z is None:
        _z = zipfile.ZipFile(ZIP)
    name, out, stem, boxes, n_bg, all_boxes = job
    im = cv2.imdecode(np.frombuffer(_z.read(name), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return 0
    H, W = im.shape[:2]
    n = 0

    def save(label, x0, y0, side, tag):
        crop = cv2.resize(im[y0:y0 + side, x0:x0 + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/{label}/{stem}_{tag}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    for k, (label, cx, cy, w, h) in enumerate(boxes):
        bw, bh = w * W, h * H
        if max(bw, bh) < MIN_PX:
            continue
        side = int(min(max(max(bw, bh) * CONTEXT, 64), W, H))
        x0 = int(min(max(cx * W - side / 2, 0), W - side)); y0 = int(min(max(cy * H - side / 2, 0), H - side))
        save(label, x0, y0, side, f"b{k}"); n += 1
    px = [((cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H) for cx, cy, w, h in all_boxes]
    for t in range(n_bg):
        for _ in range(40):
            side = int(random.uniform(0.3, 0.7) * min(W, H))
            x0, y0 = random.randint(0, W - side), random.randint(0, H - side)
            if not any(x0 < bx + bw and x0 + side > bx and y0 < by + bh and y0 + side > by for bx, by, bw, bh in px):
                save("normal", x0, y0, side, f"bg{t}"); n += 1; break
    return n


def build_jobs(z, split):
    src = SPLIT_SRC[split]
    q_crack, q_pot, q_mark, q_bg = QUOTA[split]
    boxes_by_img, cands = {}, defaultdict(list)
    for n in z.namelist():
        if n.startswith(f"RDD_SPLIT/{src}/labels/") and n.endswith(".txt"):
            img = n.replace("/labels/", "/images/").replace(".txt", ".jpg")
            rows = []
            for l in z.read(n).decode().strip().splitlines():
                c, cx, cy, w, h = l.split(); rows.append((int(c), *map(float, (cx, cy, w, h))))
            boxes_by_img[img] = rows
            for i, r in enumerate(rows):
                cands["crack" if r[0] in (0, 1, 2) else "pothole" if r[0] == 4 else "mark"].append((img, i))
    chosen = defaultdict(list)
    for cls, q in (("crack", q_crack), ("pothole", q_pot), ("mark", q_mark)):
        random.shuffle(cands[cls])
        for img, i in cands[cls][:q]:
            c, cx, cy, w, h = boxes_by_img[img][i]
            chosen[img].append(("pothole" if cls == "pothole" else "crack" if cls == "crack" else "normal", cx, cy, w, h))
    bg_imgs = list(boxes_by_img)
    random.shuffle(bg_imgs)
    bg = defaultdict(int)
    for img in bg_imgs[:q_bg]:
        bg[img] += 1
    out = OUT / split
    jobs = []
    for img in set(chosen) | set(bg):
        stem = Path(img).stem
        jobs.append((img, str(out), stem, chosen.get(img, []), bg.get(img, 0),
                     [r[1:] for r in boxes_by_img[img]]))
    return jobs


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val", "test_rdd", "test_orig"):
        for c in ("crack", "normal", "pothole"):
            (OUT / s / c).mkdir(parents=True)
    z = zipfile.ZipFile(ZIP)
    for split in QUOTA:
        jobs = build_jobs(z, split)
        print(split, "images:", len(jobs), flush=True)
        with Pool(8) as p:
            done = sum(p.imap_unordered(work, jobs, chunksize=16))
        print(split, "crops:", done, flush=True)
    # original dataset: train/val merged in (prefixed), test kept separate
    for s_src, s_dst in (("train", "train"), ("val", "val"), ("test", "test_orig")):
        for c in ("crack", "normal", "pothole"):
            for f in (ORIG / s_src / c).glob("*.jpg"):
                shutil.copy2(f, OUT / s_dst / c / f"orig_{f.name}")
    for s in ("train", "val", "test_rdd", "test_orig"):
        print(s, {c: len(list((OUT / s / c).glob('*.jpg'))) for c in ("crack", "normal", "pothole")})
