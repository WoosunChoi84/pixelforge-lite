"""
PixelForge Lite — Stage 0: Input Analyzer

Measures unique colors, scaling direction, palette sufficiency, and
dominant (background) color. No CTS, no mode_hint, no input_type.

T3.1: for large inputs (> SUBSAMPLE_THRESHOLD pixels) a quick subsample is
used first to estimate the unique color count. If the estimate clearly
exceeds max_colors (palette_sufficient will be False with certainty — since
a sample's unique count is a *lower bound* on the full image's unique
count), approximate statistics are used and existing_palette is populated
from the sample only. Otherwise a full scan runs to preserve exact
palette_sufficient behavior. This keeps the fast-path bit-identical for
small or few-color inputs while cutting 4-10× off the time cost on large
continuous-tone inputs.
"""

from __future__ import annotations

import numpy as np

from ..models import InputProfile, PipelineConfig
from ..utils import pack_rgb

_IDENTITY_TOLERANCE = 0.05
_SUBSAMPLE_THRESHOLD = 400_000     # ≤ 632×632: always exact
_SUBSAMPLE_TARGET = 200_000        # step chosen to reach this when subsampling
_MARGIN = 1.5                      # approximate must exceed max_colors * MARGIN


def analyze(image: np.ndarray, config: PipelineConfig) -> InputProfile:
    h, w = image.shape[:2]
    total_pixels = h * w
    max_colors = config.max_colors or 16

    unique_colors, unique_count, unique_ratio, dominant_color, dominant_ratio = (
        _analyze_colors(image, total_pixels, max_colors)
    )

    orig_dim = min(h, w)
    grid_dim = min(config.grid_size)
    ratio = orig_dim / grid_dim

    if ratio > (1.0 + _IDENTITY_TOLERANCE):
        direction = "down"
    elif ratio < (1.0 - _IDENTITY_TOLERANCE):
        direction = "up"
    else:
        direction = "identity"

    palette_sufficient = unique_count <= max_colors

    existing_palette = [tuple(int(c) for c in color) for color in unique_colors]

    return InputProfile(
        unique_colors=unique_count,
        unique_color_ratio=unique_ratio,
        scaling_direction=direction,
        reduction_ratio=ratio,
        palette_sufficient=palette_sufficient,
        existing_palette=existing_palette,
        dominant_color=dominant_color,
        dominant_color_ratio=dominant_ratio,
    )


def _unpack_to_rgb_array(unique_packed: np.ndarray) -> np.ndarray:
    """Convert packed uint32 array back to (N, 3) uint8 RGB."""
    return np.stack(
        [
            (unique_packed >> 16).astype(np.uint8),
            ((unique_packed >> 8) & 0xFF).astype(np.uint8),
            (unique_packed & 0xFF).astype(np.uint8),
        ],
        axis=1,
    )


def _analyze_colors(
    image: np.ndarray, total_pixels: int, max_colors: int
) -> tuple[np.ndarray, int, float, tuple[int, int, int], float]:
    pixels = image.reshape(-1, 3)

    # T3.1: for large inputs, try a quick subsample first.
    if total_pixels > _SUBSAMPLE_THRESHOLD:
        step = max(1, total_pixels // _SUBSAMPLE_TARGET)
        sampled = pixels[::step]
        sampled_packed = pack_rgb(sampled)
        sampled_unique, sampled_counts = np.unique(sampled_packed, return_counts=True)
        sampled_unique_count = len(sampled_unique)

        # Sample's unique count is a strict lower bound on the full image's.
        # If the lower bound already clearly exceeds the threshold, the full
        # image definitely has more colors than max_colors → palette_sufficient
        # is False with certainty, and we can safely use sampled statistics.
        if sampled_unique_count > max_colors * _MARGIN:
            unique_colors = _unpack_to_rgb_array(sampled_unique)
            # Approximate full-image unique count via scaling the sample.
            unique_ratio = sampled_unique_count / len(sampled)
            approx_unique_count = int(min(
                total_pixels,
                max(sampled_unique_count, round(unique_ratio * total_pixels)),
            ))
            dom_idx = np.argmax(sampled_counts)
            dom_packed = sampled_unique[dom_idx]
            dominant_color = (
                int((dom_packed >> 16) & 0xFF),
                int((dom_packed >> 8) & 0xFF),
                int(dom_packed & 0xFF),
            )
            dominant_ratio = float(sampled_counts[dom_idx] / len(sampled))
            return (
                unique_colors,
                approx_unique_count,
                unique_ratio,
                dominant_color,
                dominant_ratio,
            )
        # Sample's unique count is near-or-below threshold — full scan needed
        # to correctly decide palette_sufficient with bit-identity guarantees.

    # Full scan (default, bit-identical to pre-T3.1 behavior)
    packed = pack_rgb(pixels)
    unique_packed, counts = np.unique(packed, return_counts=True)

    unique_count = len(unique_packed)
    unique_ratio = unique_count / total_pixels if total_pixels > 0 else 1.0

    dom_idx = np.argmax(counts)
    dom_packed = unique_packed[dom_idx]
    dominant_color = (
        int((dom_packed >> 16) & 0xFF),
        int((dom_packed >> 8) & 0xFF),
        int(dom_packed & 0xFF),
    )
    dominant_ratio = float(counts[dom_idx] / total_pixels) if total_pixels > 0 else 0.0

    unique_colors = _unpack_to_rgb_array(unique_packed)

    return unique_colors, unique_count, unique_ratio, dominant_color, dominant_ratio
