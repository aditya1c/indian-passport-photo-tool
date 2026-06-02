import sys
import os
import io
from PIL import Image, ImageOps, ImageCms, ImageDraw
from whiten_bg import whiten_image, clean_speckles, feather_matte, remove_bg_ml
from check_passport import _get_landmarks, detect_hair_top, detect_chin_skin, assess_eligibility

# sRGB profile bytes — embedded on the output so viewers never have to guess.
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def load_srgb(path):
    """Open a source photo as orientation-fixed, sRGB-correct RGB.

    Two silent-corruption traps this closes (both were real footguns — see
    CLAUDE.md "Input prep"):
      • Orientation — a portrait phone photo carries EXIF orientation 6; read
        raw it looks ~90° head-rolled and the eligibility gate rejects it.
        exif_transpose() bakes the rotation into the pixels.
      • Colour — iPhone photos are tagged Display P3 (wide gamut). Reading the
        raw numbers and dropping the profile makes a viewer assume sRGB, so warm
        skin reds desaturate (cooler/flatter). profileToProfile() remaps the
        numbers P3→sRGB so the *appearance* is preserved, then we drop to plain
        sRGB. Untagged input is assumed sRGB already and passes through.
    """
    im = ImageOps.exif_transpose(Image.open(path))
    icc = im.info.get("icc_profile")
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "sRGB" not in (ImageCms.getProfileDescription(src) or ""):
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im, src, dst, renderingIntent=0, outputMode="RGB")
            print(f"[color] converted source {ImageCms.getProfileDescription(src).strip()} → sRGB")
    return im.convert("RGB")


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


# ── 6-up print collage ─────────────────────────────────────────────────────────
COLLAGE_COLS, COLLAGE_ROWS = 2, 3
COLLAGE_GUIDE = (200, 200, 200)   # light-gray 1px cut guides around each photo


def make_collage(photo, dst):
    """Tile 6 copies of the 630×810 passport photo onto a 4×6 inch print sheet.

    Each cell is the photo at its native size, which is exactly 35×45 mm at
    18 px/mm (the Indian *physical* passport size) — so the photo drops in
    pixel-for-pixel with no resampling, and the sheet comes out 1829×2743 px =
    4.00×6.00 inch at 457 DPI. Thin gray guides frame each photo for cutting.
    Print at 100% / actual size (never "fit to page") or the cut size drifts.
    """
    cw, ch = photo.size
    pxmm    = cw / 35.0                          # 18 px/mm (photo is 35 mm wide)
    sheet_w = round(101.6 * pxmm)                # 4 inch
    sheet_h = round(152.4 * pxmm)                # 6 inch
    dpi     = round(pxmm * 25.4)                 # 457
    # Distribute leftover space as equal gutters (incl. outer margins), centered.
    gx = (sheet_w - COLLAGE_COLS * cw) // (COLLAGE_COLS + 1)
    gy = (sheet_h - COLLAGE_ROWS * ch) // (COLLAGE_ROWS + 1)
    ox = (sheet_w - (COLLAGE_COLS * cw + (COLLAGE_COLS + 1) * gx)) // 2
    oy = (sheet_h - (COLLAGE_ROWS * ch + (COLLAGE_ROWS + 1) * gy)) // 2

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw  = ImageDraw.Draw(sheet)
    for r in range(COLLAGE_ROWS):
        for c in range(COLLAGE_COLS):
            x = ox + gx + c * (cw + gx)
            y = oy + gy + r * (ch + gy)
            sheet.paste(photo, (x, y))
            draw.rectangle([x - 1, y - 1, x + cw, y + ch], outline=COLLAGE_GUIDE, width=1)
    sheet.save(dst, dpi=(dpi, dpi), quality=95, icc_profile=SRGB_PROFILE)
    return dst, sheet.size, dpi

# ── Target output ─────────────────────────────────────────────────────────────
OUT_W, OUT_H = 630, 810
FACE_FRAC    = 0.807  # chin-skin-to-hair / crop-height target (≥80% after resize rounding + feather softening of the hair crown)
HEADROOM     = 0.10   # fraction of extra vertical space placed above hair
                      # (small → tight whitespace above head, more shoulder below;
                      #  doesn't change crop height, so face % is unaffected)

# ── Args ──────────────────────────────────────────────────────────────────────
#   make_passport.py [SRC] [DST] [--no-ml] [--force] [--no-level]
#   ML (rembg / U²-Net) matte is the DEFAULT; it silently falls back to the
#   Pillow flood-fill path if rembg/onnxruntime aren't installed.
#   --no-ml    : force the Pillow flood-fill path.  (--ml is accepted as a no-op.)
#   --force    : generate even if the photo fails the eligibility (pose) gate.
#   --no-level : skip the automatic roll-leveling (keep the source's head tilt).
positional = [a for a in sys.argv[1:] if not a.startswith("--")]
flags      = {a for a in sys.argv[1:] if a.startswith("--")}
USE_ML     = "--no-ml" not in flags
FORCE      = "--force" in flags
AUTO_LEVEL = "--no-level" not in flags
ROLL_EPS   = 0.5   # don't rotate for sub-0.5° roll (resampling loss > the gain)

SRC  = positional[0] if len(positional) > 0 else os.path.expanduser("~/Desktop/photo.jpeg")
DST  = positional[1] if len(positional) > 1 else os.path.expanduser("~/Desktop/photo_passport.jpeg")

# ── Step 1: load the original + auto-detect face coordinates ───────────────────
print(f"Loading {SRC} ...")
orig = load_srgb(SRC)   # EXIF-orientation fixed + P3→sRGB if needed

# ── Step 1a: eligibility gate + auto-level roll ───────────────────────────────
# A turned head (yaw), a downward gaze (pitch), or closed eyes can't be fixed by
# cropping, so refuse rather than emit a non-compliant photo. --force overrides.
# Roll, however, is *in-plane* tilt and CAN be honestly zeroed by rotating the
# source — so by default we auto-level it to 0 (rotate the source by -roll, white
# fill) and re-measure before gating. --no-level keeps the source's tilt.
print("Checking eligibility (head pose, eyes)...")
elig = assess_eligibility(orig)
if elig["pitch"] is not None:
    print(f"  Pose: pitch={elig['pitch']:+.0f}°  yaw={elig['yaw']:+.0f}°  roll={elig['roll']:+.0f}°")

if AUTO_LEVEL and elig["roll"] is not None and abs(elig["roll"]) >= ROLL_EPS:
    roll0 = elig["roll"]
    orig = orig.rotate(-roll0, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    print(f"[level] rotated source {-roll0:+.1f}° to zero the {roll0:+.1f}° head roll")
    elig = assess_eligibility(orig)
    if elig["pitch"] is not None:
        print(f"  Pose (leveled): pitch={elig['pitch']:+.0f}°  yaw={elig['yaw']:+.0f}°  roll={elig['roll']:+.0f}°")

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
final.save(DST, dpi=(600, 600), quality=95, icc_profile=SRGB_PROFILE)
print(f"Saved → {DST}  ({OUT_W}×{OUT_H}, 600 DPI, sRGB)")

# ── Step 5: 6-up print collage (always emitted alongside the photo) ───────────
collage_dst = "{}_collage_4x6{}".format(*os.path.splitext(DST))
cdst, (cw, ch), cdpi = make_collage(final, collage_dst)
print(f"Saved → {cdst}  ({cw}×{ch}, {cdpi} DPI, 6×[35×45mm], sRGB)")
print("  Print at 100% / actual size — do NOT 'fit to page' or the cut size drifts.")
