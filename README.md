# Passport Photo Tool

Python scripts to generate and validate a passport-compliant photo from a regular
smartphone image. Targets the **Indian passport** spec — **630×810 px @ 600 DPI,
white background, face ≥80%** (tweak the constants for other countries).

**Indian passport formats.** The current **digital** requirement (Passport Seva
online upload) is **630×810 px**. The **physical/printed** photo is **35 mm ×
45 mm**. The older **2×2 inch** square format is legacy and no longer the standard.
This tool produces the 630×810 px digital photo.

Background removal uses **rembg (U²-Net)** by default for clean anti-aliased edges,
and falls back automatically to a dependency-free **Pillow flood-fill** if rembg
isn't installed. Face cropping anchors (hair top / chin / center) are **auto-detected**
with MediaPipe — no per-photo editing.

Before generating, the tool runs an **eligibility check**: if the head is tilted
or turned, the gaze is down, or the eyes are closed, it reports *why the photo
isn't usable and writes no image* (pass `--force` to override) — because cropping
can't fix a bad pose.

## Requirements

```bash
pip install Pillow mediapipe numpy          # required
pip install rembg onnxruntime               # optional, enables the high-quality ML matte
```

Tested on Python 3.9+. Install `onnxruntime` explicitly — rembg doesn't always pull
it in (e.g. Python 3.9 / Apple Silicon). On first run, the MediaPipe face-landmarker
model auto-downloads to `~/.cache/mediapipe/`, and the ~176 MB rembg U²-Net model to
`~/.u2net/`.

## Quick start

Pass your photo and the output path (or drop a `photo.jpeg` on your Desktop and
use the defaults):

```bash
python3 make_passport.py photo.jpeg photo_passport.jpeg     # 630×810, 600 DPI
python3 check_passport.py photo_passport.jpeg --original photo.jpeg
```

Paths are relative/positional — nothing absolute is baked in, so the repo works
as-is for anyone who clones it.

That's the whole flow. `make_passport.py` does background removal, face detection,
crop, and resize in one step. `check_passport.py` validates and writes an annotated
`measured.jpeg` so you can eyeball the hair-top / chin markers.

## Scripts

### `make_passport.py` — generate the passport photo

Detects the face, removes the background to white, crops, and resizes to 630×810.

Every source is loaded through `load_srgb()`, which **fixes EXIF orientation**
(a portrait phone photo would otherwise read as a ~90° head roll and get rejected)
and **converts wide-gamut Display P3 → sRGB** (iPhone default), so warm skin tones
don't desaturate. The output is saved **tagged sRGB**. You no longer have to
pre-convert the photo — just feed it the raw smartphone JPEG.

```bash
python3 make_passport.py                 # defaults: ~/Desktop/photo.jpeg → ~/Desktop/photo_passport.jpeg
python3 make_passport.py SRC DST         # custom input/output paths (positional)
python3 make_passport.py --no-ml         # force the Pillow flood-fill (skip rembg)
python3 make_passport.py SRC DST --force # generate even if the eligibility gate fails
```

It first runs the **eligibility check** (head pose + eyes). If the pose isn't
compliant it lists the reasons and exits without writing a file, unless `--force`
is given. Tunables at the top of the file: `OUT_W`/`OUT_H` (output size),
`FACE_FRAC` (face/height ratio, default 0.807), `HEADROOM` (whitespace above the
head, default 0.10). The crop anchors `HAIR_TOP`/`CHIN_BOT`/`FACE_CX` are
auto-detected per photo — nothing to edit. If no face is detected, it raises a
clear error instead of producing a bad crop.

### `whiten_bg.py` — background removal (used internally; also standalone)

`make_passport.py` calls these functions for you. You can also run it on its own to
just whiten a background:

```bash
python3 whiten_bg.py            # ~/Desktop/photo.jpeg → ~/Desktop/photo_white.jpeg
python3 whiten_bg.py SRC DST
```

The Pillow path uses a BFS flood-fill seeded from border pixels, comparing each
candidate to its accepted neighbour (so gradient/uneven backgrounds work without a
fuzz parameter), then feathers the silhouette for an anti-aliased edge.

### `check_passport.py` — compliance checker

```bash
python3 check_passport.py photo_passport.jpeg                          # basic
python3 check_passport.py photo_passport.jpeg --original photo.jpeg    # + skin/pixel checks
python3 check_passport.py photo_passport.jpeg --original photo.jpeg --out-dir ./results
```

| # | Check | Pass condition |
|---|---|---|
| 0 | Eligibility (pose & eyes) | Head pitch/yaw/roll within ±8°, both eyes open |
| 1 | Dimensions | Exactly 630×810 px |
| 2 | DPI | EXIF DPI tag = 600 |
| 3 | Background whiteness | ≥95% of border pixels are pure white (top + sides, top 75%) |
| 4 | Face height | Chin-skin to hair-top ≥ 80% of photo height |
| 5 | Skin tone | Color distance from original < 30 (needs `--original`) |
| 6 | Pixel integrity | Zero interior body pixels changed (needs same-size `--original`) |

Outputs: a console report, `measured.jpeg` (red hair/chin lines + percentage), and
`report.txt` (key=value summary).

## Notes & design decisions

- **Eligibility first.** A tilted/turned head, downward gaze, or closed eyes can't be
  fixed by cropping, so the generator refuses (and the checker flags it as `[0]`) rather
  than emit a non-compliant photo. Pose comes from MediaPipe's facial-transformation
  matrix; glasses tint/glare, lighting, and expression aren't auto-detected — verify those
  by eye.
- **Chin vs beard.** Face height is measured to the actual chin *skin*, not the beard
  bottom — including the beard overstates face height by ~8 points for bearded subjects.
  The detector crosses the beard from the chin landmark and stops at the first pure-skin
  row above it; on a clean-shaven face (no beard to cross) it uses the chin landmark directly.
- **Colour & orientation are auto-handled.** `make_passport.py` runs each source
  through `load_srgb()`: `ImageOps.exif_transpose()` bakes in the EXIF rotation, and a
  non-sRGB profile (e.g. iPhone Display P3) is converted to sRGB via
  `ImageCms.profileToProfile`, with the output re-tagged sRGB. This prevents two silent
  bugs — a portrait photo read as 90° roll, and warm skin reads as cool/flat when a P3
  file is misread as sRGB. (Roll from a *real* head tilt is still a manual pre-step; see
  `CLAUDE.md`.)
- **Skin tone.** Background removal never recolors the subject. The skin-tone check
  samples fixed fractional regions, so comparing a tight passport crop to a full-body
  original can inflate the reported distance; sampling the same anatomical spots shows
  hue/saturation essentially unchanged. Any real shift comes only from the LANCZOS
  downscale + JPEG save (a 3rd-party "beauty" tool, by contrast, adds contrast/saturation).
- **Other countries.** Adjust `OUT_W`, `OUT_H`, and `FACE_FRAC` in `make_passport.py`,
  and the corresponding constants in `check_passport.py`.

See `CLAUDE.md` for the detailed algorithm/parameter reference.
</content>
