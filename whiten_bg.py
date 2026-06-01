from PIL import Image, ImageFilter
from collections import deque
import colorsys


def is_background_pixel(r, g, b, max_sat=0.12, min_val=0.35):
    _, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return s <= max_sat and v >= min_val


def whiten_image(img, local_tolerance=20, max_sat=0.12, min_val=0.35):
    """Whiten background in-memory. Returns a new PIL Image."""
    img = img.convert("RGB")
    pixels = img.load()
    w, h = img.size

    visited = [[False] * h for _ in range(w)]
    queue = deque()

    for x in range(w):
        for y in [0, h - 1]:
            r, g, b = pixels[x, y]
            if not visited[x][y] and is_background_pixel(r, g, b, max_sat, min_val):
                queue.append((x, y, (r, g, b)))
                visited[x][y] = True
    for y in range(h):
        for x in [0, w - 1]:
            r, g, b = pixels[x, y]
            if not visited[x][y] and is_background_pixel(r, g, b, max_sat, min_val):
                queue.append((x, y, (r, g, b)))
                visited[x][y] = True

    filled = 0
    while queue:
        x, y, ref = queue.popleft()
        pixels[x, y] = (255, 255, 255)
        filled += 1
        rr, rg, rb = ref
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[nx][ny]:
                nr, ng, nb = pixels[nx, ny]
                if (is_background_pixel(nr, ng, nb, max_sat, min_val) and
                        max(abs(nr - rr), abs(ng - rg), abs(nb - rb)) <= local_tolerance):
                    visited[nx][ny] = True
                    queue.append((nx, ny, (nr, ng, nb)))

    print(f"Whitened {filled:,} / {w*h:,} background pixels")
    return img


def clean_speckles(img, white_min=248):
    """Remove floating non-white islands left over by the flood-fill.

    After whiten_image, warm-toned specks (stray hairs, JPEG noise, lens
    blur near the silhouette) survive as small islands surrounded by white,
    because they fail the achromatic-background test and the flood can't pass
    through them. This finds connected components of non-white pixels, keeps
    only the largest one (the subject — head + neck + shirt are all connected),
    and whitens every other component.

    A pixel counts as "white" when all three channels are >= white_min, so the
    anti-aliased halo around a speck is removed along with the speck itself.
    """
    img = img.convert("RGB")
    pixels = img.load()
    w, h = img.size

    def is_white(x, y):
        r, g, b = pixels[x, y]
        return r >= white_min and g >= white_min and b >= white_min

    label = [[-1] * h for _ in range(w)]   # -1 unvisited, -2 white, >=0 component id
    sizes = []

    for sx in range(w):
        for sy in range(h):
            if label[sx][sy] != -1:
                continue
            if is_white(sx, sy):
                label[sx][sy] = -2
                continue
            cid = len(sizes)
            count = 0
            stack = [(sx, sy)]
            label[sx][sy] = cid
            while stack:
                x, y = stack.pop()
                count += 1
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and label[nx][ny] == -1:
                            if is_white(nx, ny):
                                label[nx][ny] = -2
                            else:
                                label[nx][ny] = cid
                                stack.append((nx, ny))
            sizes.append(count)

    if not sizes:
        return img
    keep = max(range(len(sizes)), key=lambda i: sizes[i])
    removed_px = removed_comp = 0
    for x in range(w):
        col = label[x]
        for y in range(h):
            c = col[y]
            if c >= 0 and c != keep:
                pixels[x, y] = (255, 255, 255)
                removed_px += 1
    removed_comp = sum(1 for i, s in enumerate(sizes) if i != keep and s > 0)
    print(f"Cleaned {removed_comp:,} stray island(s), {removed_px:,} pixels "
          f"(kept subject = {sizes[keep]:,}px)")
    return img


def remove_bg_ml(img, bgcolor=(255, 255, 255)):
    """Optional high-quality background removal via the `rembg` library (U²-Net).

    rembg returns an RGBA cut-out with a real soft alpha matte, so this single
    call replaces the whole whiten_image → feather_matte → clean_speckles chain:
    the edges come out anti-aliased and the stray-island problem can't occur
    (there's no flood-fill to leave islands behind).

    Raises ImportError if rembg isn't installed — callers should catch it and
    fall back to the Pillow flood-fill path. The ~176 MB model auto-downloads
    to ~/.u2net/ on first use.
    """
    try:
        from rembg import remove
    except ImportError as e:
        # Names the actual missing module — rembg also needs onnxruntime, which
        # isn't always pulled in on its own (e.g. Python 3.9 / Apple Silicon).
        raise ImportError(
            f"--ml path unavailable ({e}). Run `pip install rembg onnxruntime`."
        ) from e

    img = img.convert("RGB")
    cut = remove(img, post_process_mask=True).convert("RGBA")  # soft alpha matte
    bg = Image.new("RGB", cut.size, bgcolor)
    bg.paste(cut, mask=cut.split()[3])
    print("Removed background with rembg (U²-Net) → soft anti-aliased matte")
    return bg


def feather_matte(orig, white_img, radius=3.0, white_min=248):
    """Re-composite the subject over white with a *feathered* (anti-aliased) edge.

    whiten_image cuts the background with a hard 1-bit mask, so the silhouette
    is aliased — stair-stepped, and it wobbles in/out by a pixel because the
    flood-fill stops at slightly different points per row. That reads as a
    "pixelated" edge at the ear / jaw.

    This rebuilds the result as a true matte:
      1. binary subject mask  = wherever white_img is NOT near-white
      2. soft alpha           = that mask, Gaussian-blurred by `radius`
      3. out = orig*alpha + white*(1-alpha)

    The blur smooths the silhouette shape and gives partial-coverage pixels at
    the boundary (real skin fading into white), exactly like alpha matting —
    while the interior stays the untouched original and the far background
    stays pure white. Returns a full-resolution RGB image.

    `radius` is in original-image pixels; pick it relative to the eventual
    downscale so the feather lands at ~1px in the final photo.
    """
    orig = orig.convert("RGB")
    white_img = white_img.convert("RGB")
    w, h = white_img.size

    # 1-bit subject mask from the whitened image
    px = white_img.load()
    mask = Image.new("L", (w, h), 0)
    mpx = mask.load()
    for x in range(w):
        for y in range(h):
            r, g, b = px[x, y]
            if not (r >= white_min and g >= white_min and b >= white_min):
                mpx[x, y] = 255

    alpha = mask.filter(ImageFilter.GaussianBlur(radius))
    white_bg = Image.new("RGB", (w, h), (255, 255, 255))
    out = Image.composite(orig, white_bg, alpha)
    print(f"Feathered silhouette (Gaussian r={radius}px) → anti-aliased edge")
    return out


if __name__ == "__main__":
    import sys, os
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/photo.jpeg")
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".jpeg", "_white.jpeg").replace(".jpg", "_white.jpg")
    result = whiten_image(Image.open(src))
    result.save(dst, quality=95)
    print(f"Saved to {dst}")
