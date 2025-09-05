
import os
import uuid
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify

import cv2
import mediapipe as mp
import numpy as np

# -------------------- Config --------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
LANDMARKS_DIR = OUTPUT_DIR / "landmarks"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LANDMARKS_DIR.mkdir(exist_ok=True)

EXTERNAL_LIBRARY_PATH = Path("/mnt/data/exercise_library.json")
LOCAL_LIBRARY_PATH = BASE_DIR / "data" / "exercise_library.json"
EXERCISE_LIBRARY_PATH = EXTERNAL_LIBRARY_PATH if EXTERNAL_LIBRARY_PATH.exists() else LOCAL_LIBRARY_PATH

ALLOWED_EXT = {"png", "jpg", "jpeg"}
SECRET_KEY = os.environ.get("FLASK_SECRET", "change-me-in-prod")

# -------------------- Flask app --------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# -------------------- MediaPipe --------------------
mp_pose = mp.solutions.pose

# -------------------- Utilities --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def save_uploads(files):
    saved = []
    for f in files:
        if f and allowed_file(f.filename):
            name = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
            dest = UPLOAD_DIR / name
            f.save(dest)
            saved.append(dest)
    return saved

# secure_filename fallback (lightweight)
def secure_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip().replace(" ", "_")

# -------------------- Pose extraction --------------------
def extract_landmarks_from_image(img_path: Path):
    """Return landmarks dict {idx: {x,y,z}} or None"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        d = {}
        for i, p in enumerate(lm):
            d[str(i)] = {"x": float(p.x * w), "y": float(p.y * h), "z": float(p.z)}
    return d

def run_pose_extractor(image_paths):
    """Process list of Path objects; save json files into outputs/landmarks"""
    saved = []
    for p in image_paths:
        lm = extract_landmarks_from_image(p)
        outfile = LANDMARKS_DIR / (p.stem + ".json")
        with open(outfile, "w") as fh:
            json.dump({"image": p.name, "landmarks": lm}, fh, indent=2)
        saved.append(outfile)
    return saved

# -------------------- Feature extraction & weakness detection --------------------
NOSE = "0"
LEFT_SHOULDER = "11"
RIGHT_SHOULDER = "12"
LEFT_HIP = "23"
RIGHT_HIP = "24"
LEFT_ANKLE = "27"
RIGHT_ANKLE = "28"

def euclid(a, b):
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5

def compute_features_from_landmarks():
    files = sorted(LANDMARKS_DIR.glob("*.json"))
    all_landmarks = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            if data.get("landmarks"):
                all_landmarks.append(data["landmarks"])
    if not all_landmarks:
        return {}
    shoulders, hips, ankles = [], [], []
    asym_sh, asym_hp = [], []
    heights = []
    for lm in all_landmarks:
        # estimate pixel height nose -> avg ankles (if available)
        if NOSE in lm and LEFT_ANKLE in lm and RIGHT_ANKLE in lm:
            nose = lm[NOSE]
            ankle = {"x": (lm[LEFT_ANKLE]["x"] + lm[RIGHT_ANKLE]["x"]) / 2,
                     "y": (lm[LEFT_ANKLE]["y"] + lm[RIGHT_ANKLE]["y"]) / 2}
            heights.append(abs(nose["y"] - ankle["y"]))
        # required landmarks
        if all(k in lm for k in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE)):
            ls, rs, lh, rh, la, ra = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER], lm[LEFT_HIP], lm[RIGHT_HIP], lm[LEFT_ANKLE], lm[RIGHT_ANKLE]
            shoulders.append(euclid(ls, rs))
            hips.append(euclid(lh, rh))
            ankles.append(euclid(la, ra))
            asym_sh.append(abs(ls["y"] - rs["y"]))
            asym_hp.append(abs(lh["y"] - rh["y"]))
    median_height = float(np.median(heights)) if heights else 1.0
    feat = {}
    if shoulders:
        mean_sh = float(np.mean(shoulders)); mean_hp = float(np.mean(hips)); mean_ank = float(np.mean(ankles))
        feat["mean_shoulder_width_px"] = mean_sh
        feat["mean_hip_width_px"] = mean_hp
        feat["mean_ankle_dist_px"] = mean_ank
        feat["shoulder_to_hip_ratio_px"] = mean_sh / (mean_hp + 1e-6)
        feat["avg_shoulder_vertical_asym_px"] = float(np.mean(asym_sh))
        feat["avg_hip_vertical_asym_px"] = float(np.mean(asym_hp))
        feat["mean_shoulder_width_norm"] = feat["mean_shoulder_width_px"] / median_height
        feat["mean_hip_width_norm"] = feat["mean_hip_width_px"] / median_height
        feat["ankle_dist_norm"] = feat["mean_ankle_dist_px"] / median_height
        feat["shoulder_vertical_asym_norm"] = feat["avg_shoulder_vertical_asym_px"] / median_height
        feat["hip_vertical_asym_norm"] = feat["avg_hip_vertical_asym_px"] / median_height
        feat["estimated_pixel_height"] = median_height
    return feat

def detect_weaknesses(features):
    """
    Basic rule-based weakness detection (toy thresholds).
    Later: replace with classifier trained on labeled data.
    """
    weak = []
    # shoulder_to_hip_ratio_norm (derived)
    # To compute normalized shoulder_to_hip, use normalized widths:
    s_norm = features.get("mean_shoulder_width_norm", 0)
    h_norm = features.get("mean_hip_width_norm", 0)
    ratio = (s_norm / (h_norm + 1e-6)) if h_norm > 0 else 0
    # heuristics (tunable)
    if ratio < 0.95:
        weak.append("upper_body_strength/shoulder_width")
    if features.get("mean_hip_width_norm", 0) < (features.get("mean_shoulder_width_norm", 0) * 0.9):
        weak.append("glutes/legs")
    if features.get("shoulder_vertical_asym_norm", 0) > 0.03:
        weak.append("shoulder_asymmetry")
    if features.get("hip_vertical_asym_norm", 0) > 0.03:
        weak.append("hip_asymmetry")
    return weak

def map_weakness_to_targets(weakness):
    mapping = {
        "upper_body_strength/shoulder_width": ["rear_delts", "upper_back", "shoulders"],
        "glutes/legs": ["glutes", "legs"],
        "shoulder_asymmetry": ["rear_delts", "upper_back"],
        "hip_asymmetry": ["glutes", "core"]
    }
    return mapping.get(weakness, [])

# -------------------- Exercise library loader & categorizer --------------------
def load_exercises():
    """Load exercise library JSON (list of exercises)."""
    path = EXERCISE_LIBRARY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Exercise library not found. Expected at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

_COMPOUND_KEYWORDS = {"squat", "press", "bench", "deadlift", "row", "pull", "thrust", "clean", "snatch", "lunge", "dip", "pull-up", "chin-up"}

def _is_compound(ex):
    name = (ex.get("name") or "").lower()
    if ex.get("type", "").lower() != "strength":
        return False
    return any(kw in name for kw in _COMPOUND_KEYWORDS)

def categorize_exercises(ex_list):
    """Bucket exercises per muscle_group into compound/accessory/mobility"""
    by_group = {}
    for e in ex_list:
        mg = e.get("muscle_group") or "other"
        bucket = by_group.setdefault(mg, {"compound": [], "accessory": [], "mobility": []})
        if _is_compound(e):
            bucket["compound"].append(e)
        else:
            name = (e.get("name") or "").lower()
            if e.get("type", "").lower() == "mobility" or "band" in name or "face pull" in name:
                bucket["mobility"].append(e)
            else:
                bucket["accessory"].append(e)
    return by_group

# -------------------- Weekly planner --------------------
BASE_TEMPLATES = {
    4: [
        ("Upper", ["chest", "upper_back", "shoulders"]),
        ("Lower", ["legs", "glutes", "calves"]),
        ("Push", ["chest", "shoulders", "triceps"]),
        ("Pull/Core", ["upper_back", "biceps", "core"])
    ],
    5: [
        ("Push", ["chest", "shoulders", "triceps"]),
        ("Pull", ["upper_back", "biceps"]),
        ("Legs", ["legs", "glutes", "calves"]),
        ("Core + Mobility", ["core", "upper_back"]),
        ("Full Body / Conditioning", ["chest", "legs", "core", "upper_back"])
    ],
    6: [
        ("Chest + Triceps", ["chest", "triceps"]),
        ("Back + Biceps", ["upper_back", "biceps"]),
        ("Legs (Quads)", ["legs"]),
        ("Shoulders + Rear Delts", ["shoulders", "upper_back"]),
        ("Glutes + Hamstrings", ["glutes", "legs"]),
        ("Core + Mobility", ["core", "upper_back"])
    ]
}

def _prioritize_muscles(features):
    weaknesses = detect_weaknesses(features)
    counts = Counter()
    for w in weaknesses:
        for t in map_weakness_to_targets(w):
            counts[t] += 1
    prioritized = [mg for mg, _ in counts.most_common()]
    return prioritized, weaknesses

def _choose_template(days, prioritized):
    days = int(days) if str(days).isdigit() else 4
    days = max(4, min(6, days))
    template = list(BASE_TEMPLATES[days])
    if prioritized:
        top = prioritized[0]
        present = any(top in mg_list for _, mg_list in template)
        if not present and days >= 5:
            template.insert(1, (f"Focused — {top.replace('_',' ').title()}", [top]))
            if len(template) > days:
                template.pop()
        elif not present and days == 4:
            template[-1] = (f"Focused — {top.replace('_',' ').title()}", [top])
    return template

def _pick_for_muscle(ex_by_group, muscle, cnt, used_ids, prefer_compound=True):
    bucket = ex_by_group.get(muscle, {"compound": [], "accessory": [], "mobility": []})
    picks = []
    if prefer_compound:
        cand = [e for e in bucket["compound"] if e["id"] not in used_ids]
        random.shuffle(cand)
        while cand and len(picks) < cnt:
            picks.append(cand.pop())
    if len(picks) < cnt:
        cand = [e for e in bucket["accessory"] if e["id"] not in used_ids and e not in picks]
        random.shuffle(cand)
        while cand and len(picks) < cnt:
            picks.append(cand.pop())
    if len(picks) < cnt:
        cand = [e for e in bucket["mobility"] if e["id"] not in used_ids and e not in picks]
        random.shuffle(cand)
        while cand and len(picks) < cnt:
            picks.append(cand.pop())
    return picks

def build_weekly_plan_dynamic(goal, features, days=4, exercises=None):
    if exercises is None:
        exercises = load_exercises()
    ex_by_group = categorize_exercises(exercises)
    prioritized, weaknesses = _prioritize_muscles(features)
    template = _choose_template(days, prioritized)
    per_day_count = 6 if int(days) >= 5 else 5
    weekly = {}
    used_ids = set()
    extra = defaultdict(int)
    if prioritized:
        extra[prioritized[0]] += 2
    if len(prioritized) > 1:
        extra[prioritized[1]] += 1
    day_no = 1
    for day_name, muscle_groups in template:
        label = f"Day {day_no} — {day_name}"
        session = []
        for mg in muscle_groups:
            desired = 1 + extra.get(mg, 0)
            picks = _pick_for_muscle(ex_by_group, mg, desired, used_ids, prefer_compound=True)
            for p in picks:
                session.append({"role": "primary", "exercise": p})
                used_ids.add(p["id"])
        while len(session) < per_day_count:
            added = False
            for mg in prioritized:
                if mg in muscle_groups:   
                    picks = _pick_for_muscle(ex_by_group, mg, 1, used_ids, prefer_compound=False)
                if picks:
                    session.append({"role": "weakness_fix", "exercise": picks[0]})
                    used_ids.add(picks[0]["id"])
                    added = True
                    break
            if added:
                continue
            day_pool = []
            for mg in muscle_groups:
                day_pool += [e for e in ex_by_group.get(mg, {}).get("accessory", []) if e["id"] not in used_ids]
                day_pool += [e for e in ex_by_group.get(mg, {}).get("mobility", []) if e["id"] not in used_ids]
            if not day_pool:
                all_acc = [e for g in ex_by_group.values() for e in g.get("accessory", []) if e["id"] not in used_ids]
                if not all_acc:
                    break
                pick = random.choice(all_acc)
            else:
                pick = random.choice(day_pool)
            session.append({"role": "accessory", "exercise": pick})
            used_ids.add(pick["id"])
        readable = []
        for item in session:
            e = item["exercise"]
            readable.append({
                "id": e["id"],
                "name": e["name"],
                "muscle_group": e.get("muscle_group"),
                "difficulty": e.get("difficulty"),
                "role": item["role"]
            })
        weekly[label] = readable
        day_no += 1
    return {"weekly_plan": weekly, "prioritized_muscles": prioritized, "weaknesses": weaknesses}

# -------------------- Flask routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    # Get form inputs
    goal = request.form.get("goal", "strength")
    days = int(request.form.get("days", 4))
    height = request.form.get("height")
    weight = request.form.get("weight")
    files = request.files.getlist("images")
    if not files or len(files) < 3:
        flash("Please upload at least 3 images (5-8 recommended).")
        return redirect(url_for("index"))
    saved = save_uploads(files)
    if not saved:
        flash("No valid images uploaded.")
        return redirect(url_for("index"))
    # Pose extraction
    run_pose_extractor(saved)
    # Features
    features = compute_features_from_landmarks()
    if not features:
        flash("Could not extract landmarks from images. Use clear full-body photos.")
        return redirect(url_for("index"))
    # Build weekly plan
    try:
        plan_result = build_weekly_plan_dynamic(goal, features, days)
    except FileNotFoundError as e:
        flash(str(e))
        return redirect(url_for("index"))
    weekly_plan = plan_result["weekly_plan"]
    prioritized = plan_result["prioritized_muscles"]
    weaknesses = plan_result["weaknesses"]
    # Save result snapshot (optional)
    run_id = uuid.uuid4().hex
    outpath = OUTPUT_DIR / f"result_{run_id}.json"
    with open(outpath, "w") as fh:
        json.dump({"features": features, "weaknesses": weaknesses, "weekly_plan": weekly_plan, "goal": goal, "height": height, "weight": weight}, fh, indent=2)
    # Render results
    return render_template("results.html", features=features, weaknesses=weaknesses, weekly_plan=weekly_plan, prioritized=prioritized, goal=goal)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Optional API endpoint: accepts multipart/form-data with images + form fields.
    Returns JSON with features, weaknesses, weekly_plan.
    """
    goal = request.form.get("goal", "strength")
    days = int(request.form.get("days", 4))
    files = request.files.getlist("images")
    if not files or len(files) < 3:
        return jsonify({"error": "upload at least 3 images (5-8 recommended)"}), 400
    saved = save_uploads(files)
    run_pose_extractor(saved)
    features = compute_features_from_landmarks()
    if not features:
        return jsonify({"error": "could not extract landmarks from images"}), 400
    plan_result = build_weekly_plan_dynamic(goal, features, days)
    return jsonify(plan_result)

# -------------------- Main --------------------
if __name__ == "__main__":
    # dev server
    app.run(debug=True, host="0.0.0.0", port=8501)