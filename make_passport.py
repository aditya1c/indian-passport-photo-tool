import sys
import os
from PIL import Image
from whiten_bg import whiten_image, clean_speckles, feather_matte, remove_bg_ml
from check_passport import _get_landmarks, detect_hair_top, detect_chin_skin, assess_eligibility


def detect_face_coords(orig):
    """Auto-detect (HAIR_TOP, CHIN_BOT, FACE_CX) in the *original* photo.

    Uses the same MediaPipe landmark + scan logic as check_passport.py, so the
    crop and the compliance check agree on where the hair crown and chin skin
    are. Works on any uploaded photo — nothing is hardcoded per image.

      HAIR_TOP : first hair row above the forehead landmark (dark pixels)
      CHIN_BOT : actual chin skin, last pure-skin row before the beard starts
      FACE_CX  : horizontal midpoint of the left/right face-edge landmarks (234/454)
    """
    lm = _get_landmarks(orig)
    if lm is None:
        raise RuntimeError("No face detected in source photo — cannot compute crop.")
    W, H = orig.size
    hair_top = detect_hair_top(orig, lm)
    chin_bot = detect_chin_skin(orig, lm)
    if hair_top is None or chin_bot is None:
        raise RuntimeError("Could not detect hair top / chin skin in source photo.")
    face_cx = int((lm[234].x + lm[454].x) / 2 * W)
    return hair_top, chin_bot, face_cx

# ── Target output ─────────────────────────────────────────────────────────────
OUT_W, OUT_H = 630, 810
FACE_FRAC    = 0.807  # chin-skin-to-hair / crop-height target (≥80% after resize rounding + feather softening of the hair crown)
HEADROOM     = 0.10   # fraction of extra vertical space placed above hair
                      # (small → tight whitespace above head, more shoulder below;
                      #  doesn't change crop height, so face % is unaffected)

# ── Args ──────────────────────────────────────────────────────────────────────
#   make_passport.py [SRC] [DST] [--no-ml] [--force]
#   ML (rembg / U²-Net) matte is the DEFAULT; it silently falls back to the
#   Pillow flood-fill path if rembg/onnxruntime aren't installed.
#   --no-ml : force the Pillow flood-fill path.  (--ml is accepted as a no-op.)
#   --force : generate even if the photo fails the eligibility (pose) gate.
positional = [a for a in sys.argv[1:] if not a.startswith("--")]
flags      = {a for a in sys.argv[1:] if a.startswith("--")}
USE_ML     = "--no-ml" not in flags
FORCE      = "--force" in flags

SRC  = positional[0] if len(positional) > 0 else os.path.expanduser("~/Desktop/photo.jpeg")
DST  = positional[1] if len(positional) > 1 else os.path.expanduser("~/Desktop/photo_passport.jpeg")

# ── Step 1: load the original + auto-detect face coordinates ───────────────────
print(f"Loading {SRC} ...")
orig = Image.open(SRC).convert("RGB")

# ── Step 1a: eligibility gate ─────────────────────────────────────────────────
# A tilted/turned head, a downward gaze, or closed eyes can't be fixed by
# cropping, so refuse rather than emit a non-compliant photo. --force overrides.
print("Checking eligibility (head pose, eyes)...")
elig = assess_eligibility(orig)
if elig["pitch"] is not None:
    print(f"  Pose: pitch={elig['pitch']:+.0f}°  yaw={elig['yaw']:+.0f}°  roll={elig['roll']:+.0f}°")
if not elig["eligible"]:
    print("\n✗ This photo is NOT passport-eligible:")
    for msg in elig["issues"]:
        print(f"    • {msg}")
    print("\n  These can't be fixed by cropping. Use a front-facing photo with a")
    print("  level head, both eyes open, and a neutral expression. Also verify by")
    print("  eye: no tinted/reflective glasses, even lighting, plain background.")
    if not FORCE:
        print("\n  No image written. Re-run with --force to generate anyway.")
        sys.exit(2)
    print("\n  --force given: generating despite the issues above.")

print("Detecting face coordinates (MediaPipe)...")
HAIR_TOP, CHIN_BOT, FACE_CX = detect_face_coords(orig)
print(f"Detected: HAIR_TOP={HAIR_TOP}  CHIN_BOT={CHIN_BOT}  FACE_CX={FACE_CX}")

# ── Step 2: build the subject over a white background ─────────────────────────
#   --ml path : rembg soft matte (already anti-aliased + island-free)
#   default   : Pillow flood-fill whiten → feathered matte (anti-aliased edge)
if USE_ML:
    try:
        img = remove_bg_ml(orig)
    except ImportError as e:
        print(f"[ml] {e}")
        print("[ml] falling back to the Pillow flood-fill method")
        USE_ML = False
if not USE_ML:
    white = whiten_image(orig)
    img = feather_matte(orig, white, radius=3.0)   # full-res anti-aliased matte
W, H = img.size

# ── Step 3: calculate crop ────────────────────────────────────────────────────
face_h  = CHIN_BOT - HAIR_TOP
crop_h  = int(face_h / FACE_FRAC)
crop_w  = int(crop_h * OUT_W / OUT_H)

extra    = crop_h - face_h
headroom = int(extra * HEADROOM)
top      = max(0, HAIR_TOP - headroom)
bottom   = top + crop_h
left     = max(0, FACE_CX - crop_w // 2)
right    = left + crop_w

if right > W:
    right = W; left = right - crop_w
if bottom > H:
    bottom = H; top = bottom - crop_h

print(f"Crop : ({left},{top}) → ({right},{bottom})  {right-left}×{bottom-top}")
print(f"Face : {face_h/(bottom-top)*100:.1f}%  |  Shoulder room: {bottom-CHIN_BOT}px below chin")

# ── Step 4: crop, resize, save ────────────────────────────────────────────────
final = img.crop((left, top, right, bottom)).resize((OUT_W, OUT_H), Image.LANCZOS)
if not USE_ML:
    final = clean_speckles(final)   # remove floating non-white islands near ear/beard
final.save(DST, dpi=(600, 600), quality=95)
print(f"Saved → {DST}  ({OUT_W}×{OUT_H}, 600 DPI)")
