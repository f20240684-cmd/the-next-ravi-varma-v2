"""Image utilities used by dataset validation/preprocessing and ControlNet
conditioning. Depends only on Pillow + numpy (both lightweight, always
installed), never on torch/diffusers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

RESAMPLE = {
    "lanczos": Image.LANCZOS,
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}


# Museum scans in the corpus reach ~100 megapixels, which trips Pillow's
# decompression-bomb *warning* (89MP). These are local files we curated, so
# raise the ceiling to a generous but finite limit instead of disabling it.
Image.MAX_IMAGE_PIXELS = 300_000_000


def safe_open_image(path: "str | Path", max_side: Optional[int] = None) -> Optional[Image.Image]:
    """Open an image, returning None (never raising) if it is corrupt,
    truncated, or not a recognizable image format.

    `max_side`, if given, lets the JPEG decoder downscale by a power of two
    while decoding (`Image.draft`), so hashing/captioning a 100MP scan
    doesn't decode all 100MP. The result is still >= max_side on its
    shortest side, so callers that need full quality can resize afterwards.
    """
    try:
        img = Image.open(path)
        if max_side is not None and img.format == "JPEG":
            w, h = img.size
            scale = max_side / min(w, h)
            if scale < 1:
                img.draft("RGB", (int(w * scale) + 1, int(h * scale) + 1))
        img.load()  # force-read pixel data to catch truncated files
        return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None


def get_resolution(path: "str | Path") -> Optional[tuple[int, int]]:
    img = safe_open_image(path)
    if img is None:
        return None
    return img.size  # (width, height)


def perceptual_hash(path: "str | Path", hash_size: int = 8) -> Optional[str]:
    """A minimal average-hash (aHash) implementation for near-duplicate
    detection, avoiding an extra dependency like `imagehash`."""
    img = safe_open_image(path)
    if img is None:
        return None
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    # Pack bits into a hex string.
    bit_str = "".join("1" if b else "0" for b in bits)
    return f"{int(bit_str, 2):0{hash_size * hash_size // 4}x}"


def difference_hash(path_or_img: "str | Path | Image.Image", hash_size: int = 16) -> Optional[str]:
    """Gradient (difference) hash. More robust than aHash to global colour /
    contrast shifts (e.g. an oleograph print vs. a photo of the original
    canvas), which is the typical near-duplicate pattern in this corpus."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else safe_open_image(path_or_img)
    if img is None:
        return None
    small = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS), dtype=np.float32)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    bit_str = "".join("1" if b else "0" for b in bits)
    return f"{int(bit_str, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    int_a, int_b = int(hash_a, 16), int(hash_b, 16)
    return bin(int_a ^ int_b).count("1")


def file_sha256(path: "str | Path", chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def trim_uniform_border(img: Image.Image, tolerance: float = 12.0, max_trim_fraction: float = 0.15) -> Image.Image:
    """Remove flat scanner/mount margins (white or black bands) around a
    painting. A row/column counts as border if its pixels have low variance
    and match the colour of the image corner. Never trims more than
    `max_trim_fraction` from any side, so painted dark backgrounds survive.
    Ornate gilt frames are *not* uniform and are deliberately left alone."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    corner = np.median(np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]), axis=0)

    def is_border(line: np.ndarray) -> bool:
        return line.std(axis=0).max() < tolerance and np.abs(line.mean(axis=0) - corner).max() < tolerance * 2

    max_y, max_x = int(h * max_trim_fraction), int(w * max_trim_fraction)
    top = 0
    while top < max_y and is_border(arr[top]):
        top += 1
    bottom = h
    while h - bottom < max_y and is_border(arr[bottom - 1]):
        bottom -= 1
    left = 0
    while left < max_x and is_border(arr[:, left]):
        left += 1
    right = w
    while w - right < max_x and is_border(arr[:, right - 1]):
        right -= 1
    if (top, bottom, left, right) == (0, h, 0, w):
        return img
    return img.crop((left, top, right, bottom))


def aspect_buckets(
    base_resolution: int = 512, step: int = 64, max_ratio: float = 1.75, area_tolerance: float = 0.15
) -> list[tuple[int, int]]:
    """(width, height) buckets whose pixel count is within `area_tolerance`
    of base_resolution**2, with sides divisible by `step` (64 keeps every
    UNet down-sampling stage at an integer size) and aspect ratio <=
    `max_ratio`. For 512 this yields 512x512, 448x576, 448x640, 384x640,
    512x576 and their landscape counterparts -- 448x640 matches the ~1.4
    height/width ratio that most Ravi Varma canvases in the corpus have."""
    target_area = base_resolution * base_resolution
    lo, hi = target_area * (1 - area_tolerance), target_area * (1 + area_tolerance)
    buckets = set()
    for w in range(step, 2 * base_resolution + 1, step):
        for h in range(w, 2 * base_resolution + 1, step):
            if lo <= w * h <= hi and h / w <= max_ratio:
                buckets.add((w, h))
                buckets.add((h, w))
    return sorted(buckets)


def nearest_bucket(width: int, height: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Bucket whose aspect ratio is closest (in log space) to the image's."""
    ratio = np.log(width / height)
    return min(buckets, key=lambda b: abs(np.log(b[0] / b[1]) - ratio))


def resize_to_bucket(img: Image.Image, bucket: tuple[int, int], interpolation: str = "lanczos") -> Image.Image:
    """Scale so the image covers `bucket`, then center-crop the (small)
    remainder. Because the bucket already matches the aspect ratio, this
    crops a few percent at most instead of cutting heads/feet off tall
    portraits the way a square center-crop does."""
    resample = RESAMPLE.get(interpolation, Image.LANCZOS)
    bw, bh = bucket
    w, h = img.size
    scale = max(bw / w, bh / h)
    new_w, new_h = max(bw, round(w * scale)), max(bh, round(h * scale))
    resized = img.resize((new_w, new_h), resample)
    left, top = (new_w - bw) // 2, (new_h - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))


def resize_for_training(
    img: Image.Image,
    target_resolution: int = 512,
    mode: Literal["center_crop", "pad", "stretch"] = "center_crop",
    interpolation: str = "lanczos",
) -> Image.Image:
    """Resize/crop an image to a square `target_resolution`, matching the
    strategy used by most Diffusers LoRA training scripts."""
    resample = RESAMPLE.get(interpolation, Image.LANCZOS)
    w, h = img.size

    if mode == "stretch":
        return img.resize((target_resolution, target_resolution), resample)

    if mode == "pad":
        scale = target_resolution / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((new_w, new_h), resample)
        canvas = Image.new("RGB", (target_resolution, target_resolution), (0, 0, 0))
        canvas.paste(resized, ((target_resolution - new_w) // 2, (target_resolution - new_h) // 2))
        return canvas

    # center_crop (default): scale so the shortest side == target, then crop.
    scale = target_resolution / min(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((new_w, new_h), resample)
    left = (new_w - target_resolution) // 2
    top = (new_h - target_resolution) // 2
    return resized.crop((left, top, left + target_resolution, top + target_resolution))
