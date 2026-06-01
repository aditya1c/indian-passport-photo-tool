# Passport Photo Tool

Generate and validate passport-compliant photos from a smartphone image.
Background removal uses **rembg (U²-Net)** by default, falling back to a
dependency-free Pillow flood-fill if rembg/onnxruntime aren't installed.

## Layout

```
passport-photo-tool/
├── whiten_bg.py       # bg removal: rembg matte (remove_bg_ml) + Pillow fallback (whiten/feather/clean)
├── make_passport.py   # auto-detect face → crop → resize to 630×810 @ 600 DPI (bg removal internal)
├── check_passport.py  # compliance checker → measured.jpeg + report.txt
└── README.md / CLAUDE.md / .gitignore
```

Working files live on the Desktop, not in this folder:
- `photo.jpeg` — source smartphone photo (never modified)
- `photo_passport.jpeg` — final output of `make_passport.py`
- `measured.jpeg` — annotated hair/chin markers from `check_passport.py` (gitignored)
- `report.txt` — key=value check summary (gitignored)
- `photo_white.jpeg` — only from running `whiten_bg.py` standalone; unused by the pipeline

## Target spec (Indian passport)

| Property | Value |
|---|---|
| Size | 630 × 810 px |
| DPI | 600 (EXIF only; doesn't change pixel count) |
| Background | Pure white RGB(255,255,255) |
| Face height | ≥ 80% of photo height (chin skin → hair crown) |
| Shoulders | Visible at bottom |

**Indian passport formats.** The current **digital** requirement (Passport Seva
online upload) is **630 × 810 px**. The **physical/printed** photo is **35 mm ×
45 mm**. The older **2 × 2 inch** square format is legacy and no longer the
standard. This tool targets the 630 × 810 px digital spec.

Change `OUT_W`/`OUT_H`/`FACE_FRAC` in `make_passport.py` for other countries.

## Workflow

```bash
python3 make_passport.py                  # generate (rembg; auto-falls back to Pillow)
python3 make_passport.py --no-ml          # force the Pillow flood-fill path
python3 make_passport.py SRC DST          # override input/output paths (positional)
python3 make_passport.py SRC DST --force  # generate even if the eligibility gate fails
python3 check_passport.py photo_passport.jpeg --original photo.jpeg
```

Paths are positional/relative (or the `~/Desktop` defaults) — nothing absolute is
hardcoded, so the repo is portable. On first run the MediaPipe model downloads to
`~/.cache/mediapipe/` and the ~176 MB rembg U²-Net model to `~/.u2net/`; install
`onnxruntime` explicitly, as rembg doesn't always pull it in (e.g. Python 3.9 /
Apple Silicon).

## Eligibility gate (pose & eyes)

Some defects can't be fixed by cropping — a tilted/turned head, a downward gaze,
or closed eyes. `make_passport.py` checks these **before** generating and, if the
photo fails, prints the reasons and writes **no image** (exit 2). Pass `--force`
to override. `assess_eligibility()` (in `check_passport.py`) measures head pose
from MediaPipe's facial-transformation matrix (`output_facial_transformation_matrixes`)
and eye-open ratios:

| Limit | Constant | Meaning |
|---|---|---|
| Roll ≤ 8° | `MAX_ROLL` | head tilt (ear→shoulder) |
| Yaw ≤ 8° | `MAX_YAW` | head turned left/right |
| Pitch ≤ 8° | `MAX_PITCH` | chin up / looking down |
| Eye-open ≥ 0.15 | `MIN_EYE_OPEN` | both eyes open (≈0.3 wide open, ≈0 closed) |

Thresholds are lenient enough that a good frontal photo passes (the sample
photo reads ~0°/−2°/−3°) but a candid selfie is caught (e.g. a turned-head
kitchen photo reads yaw ≈ 14° and is rejected).

**Not auto-detected** — still need a human eye: tinted/reflective glasses or
glare, lighting evenness/shadows, neutral expression (mouth closed), hair over
the eyes, headgear, and source-background clutter.

## Face coordinates — auto-detected

`make_passport.py` auto-detects the crop anchors from the source photo via
`detect_face_coords()` — nothing is hardcoded per image, so any photo works. It
reuses `check_passport.py`'s MediaPipe landmarks + scans, so the crop and the
check agree on the same hair-crown / chin-skin points:

| Anchor | How it's found |
|---|---|
| `HAIR_TOP` | First center-third row with ≥6 dark pixels (V<0.35), above forehead landmark 10. Matches the rembg matte's head crown on real photos. |
| `CHIN_BOT` | **Chin skin.** *Bearded:* from chin landmark 152, scan up through the beard to the last pure-skin row (the beard bottom overstates face height ~8 pts). *Clean-shaven:* no beard band to cross, so fall back to landmark 152 itself (already the chin skin). |
| `FACE_CX` | Midpoint of the left/right face-edge landmarks (234 / 454) |

**Clean-shaven fix.** `detect_chin_skin` used to return `None` when it found no
beard to cross — on a clean-shaven face that made the caller stop early at the
lips (understating face height, e.g. 77% instead of 80%) or, if the chin landmark
fell below a tight crop, crash with an index error. It now clamps the landmark to
the frame and returns it as the chin when no beard is found.

Detection runs on the original (pre-bg-removal); as long as the background is
lighter than the hair/skin (studio grey, bright sky, indoor wall), the dark-pixel
scans hold. No face → raises a clear error instead of a bad crop. (The sample
photo detects ~424 / 2265 / 1522.)

## Crop & framing (`make_passport.py`)

- `FACE_FRAC = 0.807` — chin-to-hair / crop-height target; keeps face ≥80% after
  resize rounding + the ~1px hair-crown softening from the matte.
- `HEADROOM = 0.10` — fraction of spare vertical space placed above the hair.
  Lower → the crop shifts down (tighter top, more shoulder); crop height and face %
  are unchanged. Don't go below ~0.10: by 0.06 the hair/shoulders intrude on the
  border and background whiteness drops under the 95% floor.
- LANCZOS resize.

Pipeline: **default** `remove_bg_ml` → crop → resize → save.
**`--no-ml`** `whiten_image` → `feather_matte` → crop → resize → `clean_speckles` → save.

## Background removal (`whiten_bg.py`)

**ML matte (default, `remove_bg_ml`)** — rembg returns an RGBA cut-out with a real
soft alpha matte, composited over white. Edges are anti-aliased and no stray
islands form, so the feather/clean passes are skipped. Missing rembg/onnxruntime
raises ImportError → `make_passport.py` prints `[ml]` and falls back.

**Pillow fallback** (all Pillow-only):
- `whiten_image` — BFS flood-fill from border pixels passing `S<0.12 AND V≥0.35`
  (skips the dark shirt). Each candidate compares to its *accepted neighbour*, not a
  global reference, so gradient backgrounds need no fuzz param (`local_tolerance=20`).
  Stops at the warm-skin silhouette.
- `feather_matte` — the 1-bit flood mask is aliased/wobbly; this Gaussian-blurs a
  binary subject mask (`radius=3.0` full-res ≈ 1px final) into a soft alpha and
  composites `orig*alpha + white*(1-alpha)` for a true anti-aliased edge.
- `clean_speckles` — warm specks survive as islands in white; keeps only the largest
  8-connected component (head+neck+shirt), whitens the rest. Runs on the final image.

## Compliance checks (`check_passport.py`)

0. **Eligibility (pose & eyes)** — head pitch/yaw/roll within ±8° and both eyes
   open (see the Eligibility gate section). Reported as `[0]` and first in the
   summary; it's the same `assess_eligibility()` the generator gates on.
1. **Size / DPI** — exactly 630×810, EXIF DPI = 600.
2. **Background whiteness ≥95%** — top + left/right borders up to 75% height; the
   bottom 25% (shirt) is excluded.
3. **Face height ≥80%** — MediaPipe landmarker (model auto-downloads to
   `~/.cache/mediapipe/`). Hair top: from landmark 10, first row with ≥6 dark pixels.
   Chin skin: from landmark 152, cross the beard then the first pure-skin row above
   it (clean-shaven: the landmark itself). Draws both lines + `{face_h}px = {pct}%`
   into `measured.jpeg`.
4. **Skin tone vs original** (needs `--original`) — samples warm skin pixels, reports
   RGB/HSV deltas + distance (threshold 30). NOTE: it samples fixed *fractional*
   boxes, so a tight passport vs a full-body original inflates the distance —
   landmark-aligned sampling on the sample photo gives ~14 with identical hue. rembg
   never recolors the subject; any real shift comes only from the LANCZOS downscale
   + JPEG save.
5. **Pixel integrity** (needs same-size `--original`, i.e. `photo_white.jpeg`) —
   expects ~10k changed edge pixels (JPEG DCT at the silhouette, within 16px of the
   background), zero interior.

## Notes

- `measured.jpeg` / `report.txt` are gitignored generated artifacts.
- Slash command `/passport-check <photo> [--original <orig>]`
  (`~/.claude/commands/passport-check.md`).
</content>
</invoke>
