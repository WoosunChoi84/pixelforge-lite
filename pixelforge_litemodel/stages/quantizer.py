"""
PixelForge Lite — Stage 3: Color Quantization

CIELAB perceptual quantization + optional dithering (floyd/bayer/atkinson).
Fast path when palette_sufficient is true and user didn't pin a palette.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from PIL import Image

from ..models import PipelineConfig, StageResult
from ..palettes import PaletteRegistry
from ..utils import rgb_to_lab as _rgb_to_lab


def process(result: StageResult, config: PipelineConfig) -> StageResult:
    if result.image.dtype != np.uint8 or result.image.ndim != 3:
        raise ValueError(
            f"Input image must be (H, W, 3) uint8, got {result.image.shape} "
            f"{result.image.dtype}"
        )

    profile = result.metadata.get("input_profile")

    if (
        profile is not None
        and profile.palette_sufficient
        and config.palette is None
    ):
        palette_list = profile.existing_palette if profile.existing_palette else []
        return StageResult(
            image=result.image.copy(),
            palette=palette_list,
            alpha=result.alpha,
            metadata={
                **result.metadata,
                "stage": "quantizer",
                "quantizer_skipped": True,
                "skip_reason": (
                    f"palette_sufficient: {profile.unique_colors} colors "
                    f"<= max_colors"
                ),
                "palette_size": len(palette_list),
                "dithering_applied": False,
                "dither_method": None,
            },
        )

    palette = _resolve_palette(config, result.image)

    if config.max_colors and len(palette) > config.max_colors:
        palette = _reduce_palette(palette, result.image, config.max_colors)

    if config.dithering:
        quantized_image = _apply_dithering(
            result.image, palette, config.dither_method
        )
    else:
        quantized_image = _quantize_to_palette(result.image, palette)

    palette_list = [(int(r), int(g), int(b)) for r, g, b in palette]

    return StageResult(
        image=quantized_image,
        palette=palette_list,
        alpha=result.alpha,
        metadata={
            **result.metadata,
            "stage": "quantizer",
            "palette_size": len(palette_list),
            "dithering_applied": config.dithering,
            "dither_method": config.dither_method if config.dithering else None,
        },
    )


def _resolve_palette(config: PipelineConfig, image: np.ndarray) -> np.ndarray:
    if isinstance(config.palette, str):
        registry = PaletteRegistry()
        colors = registry.get(config.palette)
        return np.array(colors, dtype=np.uint8)
    if isinstance(config.palette, list):
        return np.array(config.palette, dtype=np.uint8)

    max_colors = config.max_colors or 16
    return _generate_palette_from_image(image, max_colors)


def _generate_palette_from_image(image: np.ndarray, max_colors: int) -> np.ndarray:
    if max_colors <= 0:
        max_colors = 16

    pil_image = Image.fromarray(image, mode="RGB")
    quantized_pil = pil_image.quantize(
        colors=max_colors, method=Image.Quantize.MEDIANCUT
    )
    raw_palette = quantized_pil.getpalette()

    if raw_palette is None:
        pixels = image.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        if len(unique_colors) > max_colors:
            indices = np.linspace(0, len(unique_colors) - 1, max_colors, dtype=int)
            return unique_colors[indices]
        return unique_colors

    palette_flat = np.array(raw_palette[: max_colors * 3], dtype=np.uint8)
    return palette_flat.reshape(-1, 3)


def _reduce_palette(
    palette: np.ndarray, image: np.ndarray, max_colors: int
) -> np.ndarray:
    pixels = image.reshape(-1, 3)
    pixels_lab = _rgb_to_lab(pixels)
    palette_lab = _rgb_to_lab(palette)

    chunk_size = 10000
    color_counts = np.zeros(len(palette), dtype=np.int64)
    for start in range(0, len(pixels_lab), chunk_size):
        chunk = pixels_lab[start : start + chunk_size]
        distances = np.sum(
            (chunk[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]) ** 2, axis=2
        )
        nearest = np.argmin(distances, axis=1)
        for idx in nearest:
            color_counts[idx] += 1

    top_indices = np.argsort(color_counts)[::-1][:max_colors]
    return palette[top_indices]


def _find_nearest_lab(
    pixel_rgb_f: np.ndarray,
    palette_f: np.ndarray,
    palette_lab: np.ndarray,
) -> tuple[int, np.ndarray]:
    pixel_clamped = np.clip(pixel_rgb_f, 0, 255).astype(np.uint8).reshape(1, 3)
    pixel_lab = _rgb_to_lab(pixel_clamped)[0]
    distances = np.sum((pixel_lab - palette_lab) ** 2, axis=1)
    nearest_idx = int(np.argmin(distances))
    return nearest_idx, palette_f[nearest_idx]


def _quantize_to_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    pixels_lab = _rgb_to_lab(pixels)
    palette_lab = _rgb_to_lab(palette)

    chunk_size = 5000
    quantized_pixels = np.empty_like(pixels)
    for start in range(0, len(pixels_lab), chunk_size):
        end = min(start + chunk_size, len(pixels_lab))
        chunk = pixels_lab[start:end]
        distances = np.sum(
            (chunk[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]) ** 2, axis=2
        )
        nearest_indices = np.argmin(distances, axis=1)
        quantized_pixels[start:end] = palette[nearest_indices]

    return quantized_pixels.reshape(h, w, 3).astype(np.uint8)


def _apply_dithering(
    original: np.ndarray,
    palette: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "floyd":
        return _floyd_steinberg_dither(original, palette)
    if method == "bayer":
        return _bayer_dither(original, palette)
    if method == "atkinson":
        return _atkinson_dither(original, palette)

    # Defensive fallback: PipelineConfig._validate rejects unknown methods, so
    # this branch is unreachable via the normal API. Kept for direct callers.
    warnings.warn(f"Unknown dither method '{method}', falling back to no-dither quant")
    return _quantize_to_palette(original, palette)


def _floyd_steinberg_dither(original: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Floyd-Steinberg error-diffusion dithering with CIELAB-based palette matching.

    DataFlow: _floyd_steinberg_dither(original, palette)
    IN:  original (H, W, 3) uint8; palette (N, 3) uint8 ← _apply_dithering()
    CHAIN: per-row, per-pixel: (1) copy current working pixel to break view-aliasing
           → (2) vectorized CIELAB nearest-palette lookup → (3) diffuse error to
           right / bottom-left / bottom / bottom-right neighbors per the
           canonical Floyd-Steinberg weights 7/16, 3/16, 5/16, 1/16.
           [Sequential Error Diffusion]
    OUT: (H, W, 3) uint8 → _apply_dithering() consumer

    The inner loop is intrinsically sequential (each pixel's value depends on
    errors diffused by its upper and left neighbors), so the row-wise Python
    loop is unavoidable for bit-exact output. Palette LAB conversion and
    palette_f float view are hoisted outside the loop.

    Historical note: prior versions had a view-aliasing bug — `old_pixel` was
    bound as a view into `result`, so `result[y,x] = new_pixel` overwrote the
    source and `error = old_pixel - new_pixel` collapsed to zero. This is the
    fix: explicit `.copy()` of `old_pixel` preserves the pre-assignment value.
    """
    h, w = original.shape[:2]
    result = original.astype(np.float32).copy()
    palette_f = palette.astype(np.float32)
    palette_lab = _rgb_to_lab(palette)

    for y in range(h):
        for x in range(w):
            # Critical: .copy() breaks the view alias between old_pixel and
            # the storage that `result[y, x] = new_pixel` writes to.
            old_pixel = result[y, x].copy()

            # CIELAB nearest-palette lookup for this single (potentially error-
            # accumulated) pixel. Clamp to uint8 range before LAB conversion.
            clamped = np.clip(old_pixel, 0, 255).astype(np.uint8).reshape(1, 3)
            old_lab = _rgb_to_lab(clamped)[0]
            distances = np.sum((old_lab - palette_lab) ** 2, axis=1)
            nearest_idx = int(np.argmin(distances))
            new_pixel = palette_f[nearest_idx]

            result[y, x] = new_pixel
            error = old_pixel - new_pixel

            # Diffuse to right (7/16)
            if x + 1 < w:
                result[y, x + 1] += error * (7.0 / 16.0)
            # Diffuse to bottom-left (3/16)
            if y + 1 < h and x - 1 >= 0:
                result[y + 1, x - 1] += error * (3.0 / 16.0)
            # Diffuse to bottom (5/16)
            if y + 1 < h:
                result[y + 1, x] += error * (5.0 / 16.0)
            # Diffuse to bottom-right (1/16)
            if y + 1 < h and x + 1 < w:
                result[y + 1, x + 1] += error * (1.0 / 16.0)

    return np.clip(result, 0, 255).astype(np.uint8)


def _bayer_dither(original: np.ndarray, palette: np.ndarray) -> np.ndarray:
    bayer_matrix = (
        np.array(
            [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
            dtype=np.float32,
        )
        / 16.0
    )

    h, w = original.shape[:2]
    threshold_map = np.tile(
        bayer_matrix, (math.ceil(h / 4), math.ceil(w / 4))
    )[:h, :w]

    dithered = original.astype(np.float32) / 255.0
    dithered = np.clip(
        dithered + threshold_map[..., np.newaxis] * 0.1, 0, 1
    ) * 255.0
    dithered_uint8 = np.clip(dithered, 0, 255).astype(np.uint8)
    return _quantize_to_palette(dithered_uint8, palette)


def _atkinson_dither(original: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Atkinson error-diffusion dithering (6 neighbors, 1/8 each; 2/8 un-diffused).

    DataFlow: _atkinson_dither(original, palette)
    IN:  original (H, W, 3) uint8; palette (N, 3) uint8 ← _apply_dithering()
    CHAIN: per-row, per-pixel: copy current pixel → CIELAB nearest-palette
           lookup → diffuse error/8 to 6 neighbors
           (right, right+2, bottom-left, bottom, bottom-right, bottom+2).
           Characteristic Atkinson signature: preserved detail with higher
           contrast vs Floyd-Steinberg's smoother blend.
           [Sequential Partial Error Diffusion]
    OUT: (H, W, 3) uint8 → _apply_dithering() consumer

    Same view-aliasing fix as Floyd-Steinberg — explicit .copy() on old_pixel.
    """
    h, w = original.shape[:2]
    result = original.astype(np.float32).copy()
    palette_f = palette.astype(np.float32)
    palette_lab = _rgb_to_lab(palette)

    for y in range(h):
        for x in range(w):
            old_pixel = result[y, x].copy()

            clamped = np.clip(old_pixel, 0, 255).astype(np.uint8).reshape(1, 3)
            old_lab = _rgb_to_lab(clamped)[0]
            distances = np.sum((old_lab - palette_lab) ** 2, axis=1)
            nearest_idx = int(np.argmin(distances))
            new_pixel = palette_f[nearest_idx]

            result[y, x] = new_pixel
            error = (old_pixel - new_pixel) / 8.0

            # Right
            if x + 1 < w:
                result[y, x + 1] += error
            # Right+2
            if x + 2 < w:
                result[y, x + 2] += error
            # Bottom-left
            if y + 1 < h and x - 1 >= 0:
                result[y + 1, x - 1] += error
            # Bottom
            if y + 1 < h:
                result[y + 1, x] += error
            # Bottom-right
            if y + 1 < h and x + 1 < w:
                result[y + 1, x + 1] += error
            # Bottom+2
            if y + 2 < h:
                result[y + 2, x] += error

    return np.clip(result, 0, 255).astype(np.uint8)
