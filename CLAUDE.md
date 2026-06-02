# Passport Photo Tool

Generate and validate passport-compliant photos from a smartphone image.
Background removal uses **rembg (U²-Net)** by default, falling back to a
dependency-free Pillow flood-fill if rembg/onnxruntime aren't installed.

## Layout

```
passport-photo-tool/
├── whiten_bg.py       # bg removal: rembg matte (remove_bg_ml) + Pillow fallback (whiten/feather/clean)
├── make_passport.py   # load_srgb (EXIF + P3→sRGB) → auto-detect face → crop → resize to 630×810 @ 600 DPI, save sRGB (bg removal internal)
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

**Correcting roll to 0.** Roll is *in-plane* tilt, so it can be honestly
zeroed by rotating the **source** photo before generating (yaw/pitch can't —
they're out-of-plane and would need to fabricate unseen face). Rotate the
EXIF-transposed source by the negative of the measured roll (`im.rotate(+3,
resample=BICUBIC, fillcolor=(255,255,255))` countered a −3° roll on the sample),
re-measure with `assess_eligibility()` to confirm roll ≈ 0 and yaw/pitch are
unchanged, save the leveled image (e.g. `photo_level.jpeg`), then run
`make_passport.py photo_level.jpeg ...` on it. Don't try to "fix" yaw/pitch —
within the ±8° gate they're acceptable; a real frontal pose needs a re-shoot.

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

Pipeline (`load_srgb` runs first, then): **default** `remove_bg_ml` → crop →
resize → save (sRGB-tagged).
**`--no-ml`** `whiten_image` → `feather_matte` → crop → resize → `clean_speckles` → save (sRGB-tagged).

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

## Input prep (smartphone source) — handled automatically

`make_passport.py` loads every source through `load_srgb()`, which closes two
silent-corruption traps so you don't have to pre-process the photo by hand. Both
of these *were* operator footguns (documented but easy to forget); they're now
structural:

**Colour (P3 → sRGB).** The pixel pipeline is **not colour-managed** — it works in
raw RGB and assumes sRGB. iPhone photos are usually tagged **Display P3** (wide
gamut). Feeding those raw numbers in and saving without a profile makes any viewer
assume sRGB, so the warm skin reds desaturate and the face reads **cooler/flatter**
— nothing is recolored, it's a misread of unchanged data. `load_srgb()` now detects
a non-sRGB profile and runs `ImageCms.profileToProfile(im, srcP3, sRGB,
renderingIntent=0)`, remapping the numbers so the *appearance* is preserved (prints
`[color] converted source Display P3 → sRGB`). The output is then saved **tagged
sRGB** (`icc_profile=SRGB_PROFILE`) so downstream viewers never guess. Untagged
input is assumed already-sRGB and passes through untouched.

> Note: for a P3→sRGB pair, `renderingIntent` has **no effect** — both are
> matrix/TRC profiles with no gamut-mapping LUT, so littleCMS uses the same matrix
> for every intent. Don't expect intent=1 vs 0 to change the result.

**Orientation (EXIF).** `load_srgb()` calls `ImageOps.exif_transpose()` first.
Without it, a portrait phone photo stored with orientation tag 6 reads as a ~90°
head **roll** and the eligibility gate rejects a perfectly good photo.

**Still manual: roll.** A small *real* head tilt (not the EXIF kind) is in-plane,
so it's correctable — but the pipeline does **not** auto-level it (see "Correcting
roll to 0" under the Eligibility gate). If you roll-correct by hand, keep the
profile intact: convert/rotate/save as **sRGB-tagged** (or `load_srgb()` will see
no profile, assume sRGB, and the conversion is skipped — fine *only* if the file
is genuinely already sRGB).

## Working agreement — narrate every operation

When running this pipeline, **tell the user every operation performed on the
image**, not just the final result. Photo edits are silent and lossy, so the user
needs to know exactly what touched their pixels. For each run, surface:

- the source loaded and any **auto-corrections** applied (`[color]` P3→sRGB,
  EXIF orientation fix);
- any **manual pre-steps** (e.g. a +N° roll rotation, with the angle and why);
- the **detected anchors** (HAIR_TOP / CHIN_BOT / FACE_CX), crop box, and
  resulting **face %**;
- the **background-removal path** taken (rembg matte vs Pillow fallback);
- the **output** written (path, size, DPI, colour profile);
- anything **skipped, assumed, or left unfixed** (e.g. yaw not corrected, a check
  not run) — call it out explicitly rather than letting it pass silently.

Prefer listing the concrete steps over a vague "done." If a transform can't be
done honestly (warping yaw/pitch, recolouring skin), say so and why.

## Notes

- `measured.jpeg` / `report.txt` are gitignored generated artifacts.
- Slash command `/passport-check <photo> [--original <orig>]`
  (`~/.claude/commands/passport-check.md`).
</content>
</invoke>
