#!/usr/bin/env python3
"""
Passport photo compliance checker.

Usage:
    python check_passport.py <passport_photo> [--original <orig>] [--out-dir <dir>]

Checks performed:
  0. Eligibility — head pose (pitch/yaw/roll within ±8°) and both eyes open
  1. Dimensions & DPI
  2. Background whiteness
  3. Face height (chin-skin to hair top) >= 80% of photo height
  4. Skin tone stats (and comparison to original if provided)
  5. Pixel integrity vs original (body pixels unchanged except JPEG edge artefacts)

Outputs:
  - Console report
  - <out-dir>/measured.jpeg  — photo with hair/chin markers
  - <out-dir>/report.txt     — machine-readable summary
"""

import sys
import os
import math
import argparse
import colorsys
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# MediaPipe face landmarker model (downloaded on first run)
_MODEL_PATH = os.path.expanduser("~/.cache/mediapipe/face_landmarker.task")

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        import urllib.request
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print("    Downloading MediaPipe face landmarker model (first run)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            _MODEL_PATH,
        )

def _get_face_data(img_pil):
    """Run MediaPipe once; return (landmarks, transform_matrix).

    The 4x4 facial-transformation matrix is what lets us recover the head pose
    (pitch/yaw/roll) for the eligibility check. Returns (None, None) if no face.
    """
    _ensure_model()
    img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as det:
        result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np))
    if not result.face_landmarks:
        return None, None
    matrix = (result.facial_transformation_matrixes[0]
              if result.facial_transformation_matrixes else None)
    return result.face_landmarks[0], matrix


def _get_landmarks(img_pil):
    """Backward-compatible helper — landmarks only (or None)."""
    return _get_face_data(img_pil)[0]

# ── 0. eligibility (head pose & eyes) ─────────────────────────────────────────
# Full-frontal passport pose tolerances (degrees) + minimum eye-open ratio.
# Passport Seva / ICAO want the subject squarely facing the camera, head level,
# both eyes open. Thresholds are deliberately lenient so a slightly-imperfect but
# acceptable photo still passes; a candid selfie (turned/tilted head) does not.
MAX_ROLL  = 8.0   # head tilt (ear toward shoulder)
MAX_YAW   = 8.0   # head turned left/right
MAX_PITCH = 8.0   # chin raised / looking down
MIN_EYE_OPEN = 0.15

def head_pose(matrix):
    """(pitch, yaw, roll) in degrees from MediaPipe's 4x4 transform matrix."""
    R = np.array(matrix)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.degrees(math.atan2(-R[2, 0], sy))
    yaw   = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll

def eye_open_ratios(lm):
    """Vertical/horizontal opening ratio per eye (~0 closed, ~0.3 wide open)."""
    def d(a, b):
        return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
    left  = d(159, 145) / d(33, 133)
    right = d(386, 374) / d(362, 263)
    return left, right

def assess_eligibility(img, lm=None, matrix=None):
    """Is this photo *usable* as a passport photo at all?

    Cropping and background removal cannot fix a tilted head, a three-quarter
    turn, a downward gaze, or closed eyes — so check those up front and let the
    caller refuse rather than emit a non-compliant image. Returns the measured
    angles, a list of human-readable issues, and an `eligible` flag.

    NOT auto-detected (still need a human eye): tinted/reflective glasses or
    glare, lighting evenness/shadows, neutral expression (mouth closed), hair
    over the eyes, headgear, and source-background clutter. This gate covers
    head pose and open eyes only.
    """
    if lm is None or matrix is None:
        lm, matrix = _get_face_data(img)
    if lm is None or matrix is None:
        return {"eligible": False, "issues": ["No face detected in the photo."],
                "pitch": None, "yaw": None, "roll": None}

    pitch, yaw, roll = head_pose(matrix)
    eye_l, eye_r = eye_open_ratios(lm)
    issues = []
    if abs(roll) > MAX_ROLL:
        issues.append(f"Head tilted {abs(roll):.0f}° — keep it level (≤{MAX_ROLL:.0f}°).")
    if abs(yaw) > MAX_YAW:
        issues.append(f"Head turned {abs(yaw):.0f}° — face the camera squarely (≤{MAX_YAW:.0f}°).")
    if abs(pitch) > MAX_PITCH:
        issues.append(f"Chin up/down {abs(pitch):.0f}° — look straight ahead (≤{MAX_PITCH:.0f}°).")
    if eye_l < MIN_EYE_OPEN or eye_r < MIN_EYE_OPEN:
        issues.append("Eyes not clearly open — keep both eyes open and visible.")

    return {"eligible": not issues, "issues": issues,
            "pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1),
            "eye_open_l": round(eye_l, 3), "eye_open_r": round(eye_r, 3)}

def check_eligibility(img):
    """Console section for the eligibility gate; returns the assessment dict."""
    print("\n[0] Eligibility (head pose & eyes)")
    print("    Running MediaPipe face landmark detection...")
    elig = assess_eligibility(img)
    if elig["pitch"] is None:
        print("    No face detected  ✗")
        return elig
    print(f"    Pose : pitch={elig['pitch']:+.0f}°  yaw={elig['yaw']:+.0f}°  "
          f"roll={elig['roll']:+.0f}°   eyes L={elig['eye_open_l']} R={elig['eye_open_r']}")
    if elig["eligible"]:
        print("    Front-facing, level, eyes open  ✓")
    else:
        print("    Not a compliant pose  ✗")
        for msg in elig["issues"]:
            print(f"      • {msg}")
    print("    (Not auto-checked: tinted/reflective glasses, lighting/shadows,")
    print("     neutral expression — verify these by eye.)")
    return elig

# ── helpers ──────────────────────────────────────────────────────────────────

def hsv(r, g, b):
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

def is_skin(r, g, b):
    h, s, v = hsv(r, g, b)
    return s > 0.08 and 0.03 < h < 0.14 and v > 0.3

def is_dark(r, g, b):
    return hsv(r, g, b)[2] < 0.35

def is_white(r, g, b):
    return r == 255 and g == 255 and b == 255

# ── 1. dimensions & DPI ──────────────────────────────────────────────────────

def check_dimensions(img):
    W, H = img.size
    info = img.info
    dpi = info.get("dpi", (None, None))
    dpi_x = round(dpi[0]) if dpi[0] else "unknown"

    ok_size = (W == 630 and H == 810)
    ok_dpi  = (dpi_x == 600)

    print(f"\n[1] Dimensions & DPI")
    print(f"    Size : {W}×{H}px  {'✓' if ok_size else '✗ (expected 630×810)'}")
    print(f"    DPI  : {dpi_x}    {'✓' if ok_dpi  else '✗ (expected 600)'}")
    return {"width": W, "height": H, "dpi": dpi_x,
            "size_ok": ok_size, "dpi_ok": ok_dpi}

# ── 2. background whiteness ───────────────────────────────────────────────────

def check_background(img):
    px = img.load()
    W, H = img.size
    BORDER = 15           # px inward from each edge to sample
    # Only check top + left + right borders; bottom has shirt/shoulders which are
    # intentionally not white — checking it would always fail for passport photos.
    BOTTOM_SKIP = int(H * 0.75)
    total = white = 0

    # Top border (full width)
    for x in range(W):
        for y in range(BORDER):
            total += 1
            if is_white(*px[x, y]):
                white += 1

    # Left + right borders (top 75% of height only)
    for y in range(BORDER, BOTTOM_SKIP):
        for x in list(range(BORDER)) + list(range(W - BORDER, W)):
            total += 1
            if is_white(*px[x, y]):
                white += 1

    pct = white / total * 100
    ok  = pct >= 95.0
    print(f"\n[2] Background whiteness")
    print(f"    Border white pixels: {pct:.1f}%  {'✓' if ok else '✗ (expected ≥95%)'}")
    return {"bg_white_pct": round(pct, 1), "bg_ok": ok}

# ── 3. face height ────────────────────────────────────────────────────────────

def detect_hair_top(img_pil, lm):
    """Scan upward from forehead landmark (10) to find first hair row."""
    px = img_pil.load()
    W, H = img_pil.size
    fore_y = int(lm[10].y * H)
    for y in range(fore_y, -1, -1):
        if sum(1 for x in range(W // 3, 2 * W // 3) if is_dark(*px[x, y])) >= 6:
            hair_top = y
    # extend upward to find the very first hair row
    for y in range(0, fore_y):
        if sum(1 for x in range(W // 3, 2 * W // 3) if is_dark(*px[x, y])) >= 6:
            return y
    return hair_top

def detect_chin_skin(img_pil, lm):
    """Find the chin-skin row.

    For a *bearded* subject the chin landmark (152) sits inside the beard, so we
    scan upward from it, cross the dark beard, and return the first pure-skin row
    above it — including the beard would overstate face height by ~8 points.

    For a *clean-shaven* subject there is no beard band to cross, so the scan
    finds nothing and we fall back to the chin landmark itself, which is already
    the chin skin. (The old code returned None here, which on a clean-shaven face
    made the caller stop early at the lips — understating face height — or, if
    the landmark fell below a tight crop, crash with an index error.)
    """
    px = img_pil.load()
    W, H = img_pil.size
    cx = W // 2
    # Clamp so a chin landmark below the frame (tight crop) can't index out of range.
    chin_lm_y = min(int(lm[152].y * H), H - 1)

    in_beard   = False
    for y in range(chin_lm_y, 0, -1):
        dark = sum(1 for x in range(cx - 20, cx + 20) if is_dark(*px[x, y]))
        skin = sum(1 for x in range(cx - 20, cx + 20) if is_skin(*px[x, y]))
        if dark >= 5:
            in_beard = True
        elif in_beard and dark == 0 and skin > 30:
            return y   # first pure-skin row above the beard = chin skin
    return chin_lm_y   # clean-shaven: the chin landmark is the chin skin

def check_face_height(img):
    W, H = img.size
    print("    Running MediaPipe face landmark detection...")
    lm = _get_landmarks(img)
    if lm is None:
        print(f"\n[3] Face height  — no face detected by MediaPipe")
        return {"face_pct": None, "face_ok": False, "hair_top": None, "chin_bot": None}

    hair_top = detect_hair_top(img, lm)
    chin_bot = detect_chin_skin(img, lm)

    if hair_top is None or chin_bot is None:
        print(f"\n[3] Face height  — could not detect landmarks")
        return {"face_pct": None, "face_ok": False, "hair_top": None, "chin_bot": None}

    face_h = chin_bot - hair_top
    pct    = face_h / H * 100
    ok     = pct >= 80.0

    print(f"\n[3] Face height (chin-skin to hair top)")
    print(f"    Hair top : y={hair_top}px")
    print(f"    Chin     : y={chin_bot}px  (actual chin skin, not beard bottom)")
    print(f"    Face     : {face_h}px / {H}px = {pct:.1f}%  {'✓' if ok else '✗ (expected ≥80%)'}")
    return {"face_pct": round(pct, 1), "face_ok": ok,
            "hair_top": hair_top, "chin_bot": chin_bot}

# ── 4. skin tone stats ────────────────────────────────────────────────────────

SKIN_REGIONS = [
    ("Forehead", (0.35, 0.18, 0.65, 0.32)),
    ("L-cheek",  (0.15, 0.38, 0.38, 0.55)),
    ("R-cheek",  (0.62, 0.38, 0.85, 0.55)),
    ("Nose",     (0.42, 0.42, 0.58, 0.58)),
]

def sample_skin_avg(img):
    px = img.load()
    W, H = img.size
    all_r, all_g, all_b = [], [], []
    for _, (fx0, fy0, fx1, fy1) in SKIN_REGIONS:
        x0, y0, x1, y1 = int(W*fx0), int(H*fy0), int(W*fx1), int(H*fy1)
        for y in range(y0, y1, 3):
            for x in range(x0, x1, 3):
                r, g, b = px[x, y]
                if is_skin(r, g, b):
                    all_r.append(r); all_g.append(g); all_b.append(b)
    if not all_r:
        return None
    mr = sum(all_r) // len(all_r)
    mg = sum(all_g) // len(all_g)
    mb = sum(all_b) // len(all_b)
    mh, ms, mv = hsv(mr, mg, mb)
    return {"R": mr, "G": mg, "B": mb,
            "hue": round(mh * 360, 1), "sat": round(ms * 100, 1), "val": round(mv * 100, 1),
            "n": len(all_r)}

def check_skin_tone(img, ref_img=None):
    stats = sample_skin_avg(img)
    if stats is None:
        print(f"\n[4] Skin tone  — no skin pixels found")
        return {}

    print(f"\n[4] Skin tone")
    print(f"    Avg RGB : ({stats['R']}, {stats['G']}, {stats['B']})")
    print(f"    Hue={stats['hue']}°  Sat={stats['sat']}%  Val={stats['val']}%  (n={stats['n']} px)")

    result = {"skin": stats}

    if ref_img is not None:
        ref = sample_skin_avg(ref_img)
        if ref:
            dR = stats["R"] - ref["R"]
            dG = stats["G"] - ref["G"]
            dB = stats["B"] - ref["B"]
            dS = stats["sat"] - ref["sat"]
            dV = stats["val"] - ref["val"]
            dist = (dR**2 + dG**2 + dB**2) ** 0.5
            print(f"    vs original — ΔR={dR:+d} ΔG={dG:+d} ΔB={dB:+d}  "
                  f"ΔSat={dS:+.1f}%  ΔVal={dV:+.1f}%  dist={dist:.1f}")
            ok = dist < 30
            print(f"    Color distance: {dist:.1f}  {'✓ close to original' if ok else '✗ notable shift (>30)'}")
            result["skin_ref"] = ref
            result["skin_dist"] = round(dist, 1)
            result["skin_ok"]   = ok
    return result

# ── 5. pixel integrity ────────────────────────────────────────────────────────

def check_pixel_integrity(img, orig_img):
    if img.size != orig_img.size:
        print(f"\n[5] Pixel integrity — skipped (different sizes: {img.size} vs {orig_img.size})")
        return {}

    wp = img.load()
    op = orig_img.load()
    W, H = img.size

    body_ok = body_changed_edge = body_changed_interior = 0

    for y in range(H):
        for x in range(W):
            wr, wg, wb = wp[x, y]
            if is_white(wr, wg, wb):
                continue
            or_, og, ob = op[x, y]
            _, s, v = hsv(or_, og, ob)
            if not (s > 0.12 or v < 0.45):
                continue   # skip background fringe

            diff = max(abs(wr - or_), abs(wg - og), abs(wb - ob))
            if diff <= 5:
                body_ok += 1
                continue

            # Changed — check if it's within 16px of white (JPEG DCT artefact)
            near_bg = any(
                0 <= x+dx < W and 0 <= y+dy < H and is_white(*wp[x+dx, y+dy])
                for dx in range(-16, 17, 4) for dy in range(-16, 17, 4)
                if dx != 0 or dy != 0
            )
            if near_bg:
                body_changed_edge += 1
            else:
                body_changed_interior += 1

    total = body_ok + body_changed_edge + body_changed_interior
    interior_pct = body_changed_interior / total * 100 if total else 0
    ok = body_changed_interior == 0

    print(f"\n[5] Pixel integrity vs original")
    print(f"    Body pixels total    : {total:,}")
    print(f"    Unchanged (diff ≤5)  : {body_ok:,}  ({body_ok/total*100:.1f}%)")
    print(f"    Edge artefacts (DCT) : {body_changed_edge:,}  (expected — JPEG 8×8 block boundary)")
    print(f"    Interior changed     : {body_changed_interior:,}  ({interior_pct:.2f}%)  {'✓' if ok else '✗ body pixels modified'}")
    return {"body_total": total, "body_unchanged": body_ok,
            "body_edge_artefacts": body_changed_edge,
            "body_interior_changed": body_changed_interior,
            "integrity_ok": ok}

# ── measurement image ─────────────────────────────────────────────────────────

def make_measurement_image(img, hair_top, chin_bot, out_path):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    W, H = vis.size
    RED, LW = (220, 30, 30), 3

    face_h = chin_bot - hair_top
    pct    = face_h / H * 100

    draw.rectangle([0, hair_top - LW, W, hair_top + LW], fill=RED)
    draw.rectangle([0, chin_bot - LW, W, chin_bot + LW], fill=RED)

    BX = W - 18
    draw.line([(BX, hair_top), (BX, chin_bot)], fill=RED, width=2)
    draw.polygon([(BX-5, hair_top+10), (BX+5, hair_top+10), (BX, hair_top)], fill=RED)
    draw.polygon([(BX-5, chin_bot-10), (BX+5, chin_bot-10), (BX, chin_bot)], fill=RED)

    try:
        font  = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 19)
    except Exception:
        font = small = ImageFont.load_default()

    draw.text((4, hair_top + 5),  f"Hair top  y={hair_top}", fill=RED, font=font)
    draw.text((4, chin_bot - 28), f"Chin  y={chin_bot}",     fill=RED, font=font)
    draw.text((BX - 115, (hair_top + chin_bot) // 2 - 12),
              f"{face_h}px = {pct:.1f}%", fill=RED, font=small)

    vis.save(out_path, quality=95)
    print(f"\n    Measurement image → {out_path}")

# ── main ──────────────────────────────────────────────────────────────────────

def run(passport_path, orig_path=None, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(passport_path))
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 56)
    print(f"  Passport Photo Compliance Check")
    print(f"  Input : {passport_path}")
    if orig_path:
        print(f"  Ref   : {orig_path}")
    print("=" * 56)

    img  = Image.open(passport_path).convert("RGB")
    orig = Image.open(orig_path).convert("RGB") if orig_path else None

    results = {}
    elig = check_eligibility(img)
    results["eligible"] = elig.get("eligible")
    results["pose_pitch"] = elig.get("pitch")
    results["pose_yaw"] = elig.get("yaw")
    results["pose_roll"] = elig.get("roll")
    results["eligibility_issues"] = "; ".join(elig.get("issues", [])) or "none"
    results.update(check_dimensions(img))
    results.update(check_background(img))
    face = check_face_height(img)
    results.update(face)
    results.update(check_skin_tone(img, orig))
    if orig:
        results.update(check_pixel_integrity(img, orig))

    # Measurement image
    if face.get("hair_top") and face.get("chin_bot"):
        meas_path = os.path.join(out_dir, "measured.jpeg")
        make_measurement_image(img, face["hair_top"], face["chin_bot"], meas_path)

    # Summary
    checks = [
        ("Eligible pose (front-facing, eyes open)", results.get("eligible")),
        ("Size 630×810",          results.get("size_ok")),
        ("DPI 600",                results.get("dpi_ok")),
        ("Background white ≥95%", results.get("bg_ok")),
        ("Face height ≥80%",       results.get("face_ok")),
        ("Skin tone vs original",  results.get("skin_ok")),
        ("Pixel integrity",        results.get("integrity_ok")),
    ]

    print(f"\n{'─'*56}")
    print("  SUMMARY")
    print(f"{'─'*56}")
    all_pass = True
    for label, status in checks:
        if status is None:
            icon = "–"
        elif status:
            icon = "✓"
        else:
            icon = "✗"
            all_pass = False
        print(f"  {icon}  {label}")

    print(f"{'─'*56}")
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'─'*56}\n")

    # Write report
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"  Report → {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passport photo compliance checker")
    parser.add_argument("passport",              help="Passport photo path")
    parser.add_argument("--original", "-o",      help="Original photo for comparison")
    parser.add_argument("--out-dir",  "-d",      help="Output directory (default: same as input)")
    args = parser.parse_args()
    run(args.passport, args.original, args.out_dir)
