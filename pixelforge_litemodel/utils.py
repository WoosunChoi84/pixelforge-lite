"""
PixelForge Lite — Shared Utilities

Cross-stage helpers that don't belong to any single stage:
  - rgb_to_lab: CIELAB conversion (used by resampler + quantizer)
  - ALPHA_THRESHOLD, composite_rgba_on_white, resample_alpha_binary:
    alpha handling helpers used end-to-end when input is RGBA
"""

from __future__ import annotations

import numpy as np

ALPHA_THRESHOLD: int = 128


def pack_rgb(pixels: np.ndarray) -> np.ndarray:
    """
    Pack (N, 3) uint8 RGB pixels into a (N,) uint32 array via R*65536 + G*256 + B.

    DataFlow: pack_rgb(pixels)
    IN:  pixels (N, 3) uint8 (flat pixel list, not an image)
    CHAIN: shift+add per channel → single uint32 per pixel [BitPack]
    OUT: (N,) uint32

    Used by input_analyzer and resampler for fast `np.unique(return_counts=True)`
    over color values. Replaces 4 duplicate inline implementations (T2.2).
    """
    return (
        pixels[:, 0].astype(np.uint32) * 65536
        + pixels[:, 1].astype(np.uint32) * 256
        + pixels[:, 2].astype(np.uint32)
    )


def unpack_rgb(packed: int) -> tuple[int, int, int]:
    """Inverse of pack_rgb for a single packed uint32 value."""
    p = int(packed)
    return ((p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF)


def composite_rgba_on_white(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split RGBA into (RGB composited on white, binary alpha mask).

    DataFlow: composite_rgba_on_white(rgba)
    IN:  rgba (H, W, 4) uint8
    CHAIN: extract alpha → alpha-blend RGB onto white background (RGB pipeline
           sees a well-defined RGB image where transparent areas look white)
           → threshold alpha at ALPHA_THRESHOLD to produce binary mask
    OUT: (rgb (H, W, 3) uint8, alpha_mask (H, W) uint8 ∈ {0, 255})

    The RGB path statistics (dom color, k-means, palette) naturally include
    the white-composited transparent pixels, so no stage needs alpha-masking.
    The binary alpha mask is propagated in parallel and recombined at output.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA (H, W, 4), got shape {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError(f"Expected uint8, got {rgba.dtype}")

    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white_bg = np.full_like(rgb, 255.0)
    composited = alpha * rgb + (1.0 - alpha) * white_bg
    rgb_out = np.clip(composited, 0, 255).astype(np.uint8)

    alpha_binary = np.where(
        rgba[:, :, 3] >= ALPHA_THRESHOLD, np.uint8(255), np.uint8(0)
    )
    return rgb_out, alpha_binary


def resample_alpha_binary(
    alpha: np.ndarray, grid_size: tuple[int, int], direction: str
) -> np.ndarray:
    """
    Resample a binary alpha mask in parallel with the RGB image.

    DataFlow: resample_alpha_binary(alpha, grid_size, direction)
    IN:  alpha (H, W) uint8 ∈ {0, 255}; grid_size (w, h); direction enum
    CHAIN:
      up       → cv2.INTER_NEAREST (preserves binary values exactly)
      identity → copy if exact match, else INTER_NEAREST
      down     → block-wise majority vote with tie → transparent:
                 opaque iff count_opaque > block_size / 2
    OUT: (grid_h, grid_w) uint8 ∈ {0, 255}

    Tie-breaker "transparent wins" is a policy decision (REPORT design):
    preserves sprite cutout semantics — leans toward conservative silhouette.
    """
    import cv2

    grid_w, grid_h = grid_size
    h, w = alpha.shape[:2]

    if direction == "up":
        return cv2.resize(alpha, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    if direction == "identity":
        if h == grid_h and w == grid_w:
            return alpha.copy()
        return cv2.resize(alpha, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)

    # down: block-wise majority (strict > half for opaque)
    output = np.zeros((grid_h, grid_w), dtype=np.uint8)
    row_b = np.clip(np.round(np.linspace(0, h, grid_h + 1)).astype(int), 0, h)
    col_b = np.clip(np.round(np.linspace(0, w, grid_w + 1)).astype(int), 0, w)
    for i in range(grid_h):
        for j in range(grid_w):
            block = alpha[row_b[i] : row_b[i + 1], col_b[j] : col_b[j + 1]]
            if block.size == 0:
                continue
            opaque = int(np.sum(block > 0))
            if opaque * 2 > block.size:  # strict majority; tie → transparent
                output[i, j] = 255
    return output


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 → CIELAB float32 via sRGB→XYZ→Lab with D65 illuminant.

    DataFlow: rgb_to_lab(rgb)
    IN:  rgb (..., 3) uint8 RGB (any leading shape)
    CHAIN: normalize [0, 1] → inverse gamma → linear sRGB → XYZ (D65 matrix)
           → divide by D65 white point → f(t) nonlinearity → Lab triplet
    OUT: (..., 3) float32 — L in [0, 100], a/b roughly [-128, 127]

    Used by:
      - quantizer: CIELAB argmin palette mapping, dithering error selection
      - resampler: k-means clustering in CIELAB, cluster→bg distance selection
    """
    rgb_float = rgb.astype(np.float32) / 255.0
    mask = rgb_float > 0.04045
    rgb_linear = np.where(
        mask, ((rgb_float + 0.055) / 1.055) ** 2.4, rgb_float / 12.92
    )

    r, g, b = rgb_linear[..., 0], rgb_linear[..., 1], rgb_linear[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    epsilon = 0.008856
    kappa = 903.3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > epsilon, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_lab = 200.0 * (fy - fz)
    return np.stack([L, a, b_lab], axis=-1)
