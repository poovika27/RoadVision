"""RoadVision AI - Road Damage Intelligence Platform.

Deep Learning CNN (pothole / crack / normal) with multi-scale tile scanning
and spatial damage localization for both road images and video streams.

Run:  .venv\\Scripts\\python app.py   then open http://127.0.0.1:5000
"""
import json
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, abort, jsonify, render_template, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).parent
RUNS = ROOT / "web_runs"
RUNS.mkdir(exist_ok=True)
CLASSES = json.loads((ROOT / "models" / "classes.json").read_text())  # ['crack', 'normal', 'pothole']
MODEL = tf.keras.models.load_model(ROOT / "models" / "final.keras", compile=False)
IMG = 224

# Video analysis parameters
THRESHOLDS = {"pothole": 0.95, "crack": 0.90}
MIN_TILES = 2
TILE_FRAC = 0.30
MIN_FRAMES = 3
COLORS = {
    "pothole": (40, 40, 245),   # Crimson/Red (BGR)
    "crack": (0, 165, 255),     # Amber/Orange (BGR)
    "normal": (50, 205, 50),    # Emerald (BGR)
}
MAX_MB = 300

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = None

app = Flask(__name__, template_folder=str(ROOT / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def draw_detection_box(img, x, y, w, h, label, conf, color):
    """Draw a high-tech bounding box with corner brackets and label badge."""
    H, W = img.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))

    # Translucent box tint
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.14, img, 0.86, 0, img)

    # Crisp perimeter line
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    # Tech-style corner accents
    corner = min(18, w // 4, h // 4)
    if corner > 4:
        cv2.line(img, (x, y), (x + corner, y), color, 4, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + corner), color, 4, cv2.LINE_AA)
        cv2.line(img, (x + w, y), (x + w - corner, y), color, 4, cv2.LINE_AA)
        cv2.line(img, (x + w, y), (x + w, y + corner), color, 4, cv2.LINE_AA)
        cv2.line(img, (x, y + h), (x + corner, y + h), color, 4, cv2.LINE_AA)
        cv2.line(img, (x, y + h), (x, y + h - corner), color, 4, cv2.LINE_AA)
        cv2.line(img, (x + w, y + h), (x + w - corner, y + h), color, 4, cv2.LINE_AA)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner), color, 4, cv2.LINE_AA)

    # Label badge with dark background + colored accent bar
    tag = f"{label.upper()} {conf:.0%}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.52
    thickness = 1
    (tw, th), _ = cv2.getTextSize(tag, font, scale, thickness)
    badge_h = th + 10
    badge_w = tw + 18
    badge_y = y - 6 if y - badge_h >= 4 else y + badge_h + 4

    cv2.rectangle(img, (x, badge_y - badge_h), (x + badge_w, badge_y), (15, 23, 42), -1)
    cv2.rectangle(img, (x, badge_y - badge_h), (x + badge_w, badge_y), color, 1, cv2.LINE_AA)
    cv2.circle(img, (x + 8, badge_y - badge_h // 2), 3, color, -1)
    cv2.putText(img, tag, (x + 16, badge_y - 4), font, scale, (241, 245, 249), thickness, cv2.LINE_AA)


def process_image(path, out_dir):
    """Run dual-level whole-image and multi-scale tile CNN inference on a road photo."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("Cannot decode this image file. Supported formats: JPG, PNG, WEBP, BMP.")

    H, W = img.shape[:2]
    max_side = 1280
    scale = min(1.0, max_side / max(H, W))
    if scale < 1.0:
        W2, H2 = int(W * scale), int(H * scale)
        img = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
        H, W = H2, W2

    # Save original clean copy for comparison
    orig_path = out_dir / "original.jpg"
    cv2.imwrite(str(orig_path), img)

    # 1. Global image-level classification
    rgb_full = cv2.cvtColor(cv2.resize(img, (IMG, IMG)), cv2.COLOR_BGR2RGB)
    global_probs = MODEL.predict(rgb_full[None].astype("float32"), verbose=0)[0]
    prob_dict = {CLASSES[k]: float(global_probs[k]) for k in range(len(CLASSES))}
    top_global_k = int(np.argmax(global_probs))
    top_global_class = CLASSES[top_global_k]
    top_global_conf = float(global_probs[top_global_k])

    # 2. Tile scanning across the entire road image
    tile_scales = [0.32]
    if min(W, H) >= 360:
        tile_scales.append(0.48)

    boxes = []
    for frac in tile_scales:
        side = max(int(min(W, H) * frac), 64)
        step = max(side // 2, 1)
        for y in range(0, H - side + 1, step):
            for x in range(0, W - side + 1, step):
                boxes.append((x, y, side))

    if not boxes:
        boxes = [(0, 0, min(W, H))]

    batch = np.stack([
        cv2.cvtColor(cv2.resize(img[y:y + s, x:x + s], (IMG, IMG)), cv2.COLOR_BGR2RGB)
        for x, y, s in boxes
    ]).astype("float32")
    probs = MODEL.predict(batch, verbose=0, batch_size=64)

    IMG_THRESHOLDS = {"pothole": 0.75, "crack": 0.70}
    hits = {"pothole": [], "crack": []}
    for (x, y, s), p in zip(boxes, probs):
        k = int(p.argmax())
        cls_name = CLASSES[k]
        if cls_name != "normal" and p[k] >= IMG_THRESHOLDS.get(cls_name, 0.75):
            hits[cls_name].append((x, y, s, float(p[k])))

    dets = []
    for kind, hs in hits.items():
        if not hs:
            continue
        mask = np.zeros((H, W), np.uint8)
        for x, y, s, _ in hs:
            mask[y:y + s, x:x + s] = 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            bx, by, bw, bh = st[i][:4]
            inside = [h for h in hs if bx <= h[0] + h[2] / 2 <= bx + bw and by <= h[1] + h[2] / 2 <= by + bh]
            if inside:
                conf = float(max(h[3] for h in inside))
                dets.append({
                    "class": kind,
                    "confidence": conf,
                    "box": [int(bx), int(by), int(bw), int(bh)]
                })

    # Sort detections by confidence descending
    dets.sort(key=lambda d: d["confidence"], reverse=True)

    # Fallback: If global prediction clearly identified pothole or crack (>= 0.60)
    # but tile clustering didn't produce a box (e.g. tightly zoomed in photo), frame candidate region
    if not dets and top_global_class != "normal" and top_global_conf >= 0.60:
        dets.append({
            "class": top_global_class,
            "confidence": top_global_conf,
            "box": [int(W * 0.15), int(H * 0.15), int(W * 0.70), int(H * 0.70)]
        })

    annotated = img.copy()
    pothole_count = sum(1 for d in dets if d["class"] == "pothole")
    crack_count = sum(1 for d in dets if d["class"] == "crack")

    for d in dets:
        bx, by, bw, bh = d["box"]
        color = COLORS.get(d["class"], (0, 200, 255))
        draw_detection_box(annotated, bx, by, bw, bh, d["class"], d["confidence"], color)

    # Determine status & severity
    if pothole_count > 0 or (top_global_class == "pothole" and top_global_conf >= 0.55):
        verdict = "Pothole detected"
        status = "pothole"
        severity = "High"
    elif crack_count > 0 or (top_global_class == "crack" and top_global_conf >= 0.55):
        verdict = "Crack detected"
        status = "crack"
        severity = "Moderate"
    else:
        verdict = "Road surface normal"
        status = "normal"
        severity = "Safe"

    ann_path = out_dir / "annotated.jpg"
    cv2.imwrite(str(ann_path), annotated)

    run_id = out_dir.name
    return {
        "type": "image",
        "id": run_id,
        "verdict": verdict,
        "status": status,
        "severity": severity,
        "pothole_found": status == "pothole",
        "crack_found": status == "crack",
        "total_detections": len(dets),
        "pothole_count": pothole_count,
        "crack_count": crack_count,
        "top_class": top_global_class,
        "top_confidence": round(top_global_conf, 3),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "detections": dets,
        "image_width": W,
        "image_height": H,
        "annotated_url": f"/runs/{run_id}/annotated.jpg",
        "original_url": f"/runs/{run_id}/original.jpg"
    }


def tiles_for(frame):
    """Overlapping square tiles over the lower ~75% of the frame (road area)."""
    H, W = frame.shape[:2]
    y_start = int(H * 0.25)
    side = max(int(min(W, H - y_start) * TILE_FRAC), 64)
    step = max(side // 2, 1)
    boxes = []
    for y in range(y_start, H - side + 1, step) if H - y_start > side else [y_start]:
        for x in range(0, W - side + 1, step) if W > side else [0]:
            boxes.append((x, y, side))
    return boxes


def analyse(frame):
    boxes = tiles_for(frame)
    batch = np.stack([cv2.cvtColor(cv2.resize(frame[y:y + s, x:x + s], (IMG, IMG)), cv2.COLOR_BGR2RGB)
                      for x, y, s in boxes]).astype("float32")
    probs = MODEL.predict(batch, verbose=0, batch_size=64)
    hits = {"pothole": [], "crack": []}
    for (x, y, s), p in zip(boxes, probs):
        k = int(p.argmax())
        if CLASSES[k] != "normal" and p[k] >= THRESHOLDS[CLASSES[k]]:
            hits[CLASSES[k]].append((x, y, s, float(p[k])))
    dets = []
    for kind, hs in hits.items():  # merge overlapping tiles into one box per damaged area
        if not hs:
            continue
        mask = np.zeros(frame.shape[:2], np.uint8)
        for x, y, s, _ in hs:
            mask[y:y + s, x:x + s] = 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            bx, by, bw, bh = st[i][:4]
            inside = [h for h in hs if bx <= h[0] + h[2] / 2 <= bx + bw and by <= h[1] + h[2] / 2 <= by + bh]
            if kind == "pothole" and len(inside) < MIN_TILES:
                continue  # an isolated tile is likely a false alarm
            dets.append((kind, max(h[3] for h in inside), int(bx), int(by), (int(bw), int(bh))))
    return dets


def process(path, out_dir, every_sec=0.5):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot read this video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    scale = min(1.0, 960 / max(W, H))
    W2, H2 = int(W * scale) // 2 * 2, int(H * scale) // 2 * 2
    raw = out_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2))
    step = max(int(fps * every_sec), 1)
    i, dets, pothole_frames, crack_frames, sampled, best = 0, [], 0, 0, 0, (0.0, None)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W2, H2))
        if i % step == 0:
            dets = analyse(frame)
            sampled += 1
            kinds = {d[0] for d in dets}
            pothole_frames += "pothole" in kinds
            crack_frames += "crack" in kinds
            for d in dets:
                if d[0] == "pothole" and d[1] > best[0]:
                    ann = frame.copy()
                    for k, c, x, y, s in dets:
                        cv2.rectangle(ann, (x, y), (x + s[0], y + s[1]), COLORS.get(k, (0, 0, 255)), 3)
                    best = (d[1], ann)
        for k, c, x, y, s in dets:
            color = COLORS.get(k, (0, 0, 255))
            cv2.rectangle(frame, (x, y), (x + s[0], y + s[1]), color, 2)
            cv2.putText(frame, f"{k} {c:.0%}", (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    final = out_dir / "annotated.mp4"
    if FFMPEG:
        r = subprocess.run([FFMPEG, "-y", "-i", str(raw), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(final)], capture_output=True)
        if r.returncode != 0:
            raw.replace(final)
        else:
            raw.unlink(missing_ok=True)
    else:
        raw.replace(final)
    if best[1] is not None:
        cv2.imwrite(str(out_dir / "best.jpg"), best[1])
    return {
        "type": "video",
        "pothole_found": pothole_frames >= MIN_FRAMES,
        "pothole_frames": pothole_frames,
        "crack_frames": crack_frames,
        "sampled_frames": sampled,
        "duration_s": round(i / fps, 1),
        "has_best": best[1] is not None,
    }


@app.route("/")
def home():
    template_path = ROOT / "templates" / "index.html"
    if template_path.is_file():
        return render_template("index.html")
    return "Template index.html missing."


@app.route("/sample/<name>")
def get_sample(name):
    sample_dir = (ROOT / "static" / "images").resolve()
    target = (sample_dir / secure_filename(name)).resolve()
    if not target.is_relative_to(sample_dir) or not target.is_file():
        abort(404)
    return send_from_directory(sample_dir, target.name)


@app.route("/analyse/image", methods=["POST"])
@app.route("/analyse-image", methods=["POST"])
def analyse_image_endpoint():
    file = request.files.get("image") or request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No image file was provided"}), 400

    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported image format ({ext}). Please upload JPG, PNG, WEBP, or BMP."}), 400

    run_id = uuid.uuid4().hex
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = out_dir / ("input" + ext)
    file.save(input_path)

    try:
        result = process_image(input_path, out_dir)
        return jsonify(result)
    except Exception as e:
        app.logger.exception("Image analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/analyse/video", methods=["POST"])
@app.route("/analyse-video", methods=["POST"])
def analyse_video_endpoint():
    video = request.files.get("video") or request.files.get("file")
    if not video or not video.filename:
        return jsonify({"error": "Please select a video file"}), 400

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    extension = Path(video.filename).suffix.lower()

    if extension not in allowed_extensions:
        return jsonify({"error": f"Unsupported video format ({extension}). Please upload MP4, MOV, AVI, or MKV."}), 400

    run_id = uuid.uuid4().hex
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = out_dir / ("input" + extension)
    video.save(input_path)

    try:
        result = process(input_path, out_dir)
        return jsonify({"id": run_id, **result})
    except Exception as e:
        app.logger.exception("Video analysis failed")
        return jsonify({"error": str(e)}), 500


@app.route("/analyse", methods=["POST"])
def analyse_unified():
    if "image" in request.files:
        return analyse_image_endpoint()
    elif "video" in request.files:
        return analyse_video_endpoint()
    elif "file" in request.files:
        f = request.files["file"]
        ext = Path(f.filename or "").suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
            return analyse_image_endpoint()
        else:
            return analyse_video_endpoint()
    return jsonify({"error": "No media file provided. Please upload an image or video."}), 400


@app.route("/runs/<run_id>/<name>")
def get_run_file(run_id, name):
    run_dir = (RUNS / run_id).resolve()
    if not run_dir.is_relative_to(RUNS.resolve()):
        abort(404)
    return send_from_directory(run_dir, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
