"""
PixelForge Lite — Stage 1: Preprocessor

Aspect-ratio adjustment at original resolution. RGBA is already composited
on white at the SDK normalization boundary (see __init__.py); any 4-channel
input reaching this stage is an internal bug.

When the StageResult carries an `alpha` mask, the same aspect-ratio
operation is applied in parallel:
  - letterbox: alpha padded with 0 (transparent) where RGB is padded with white
  - fill (crop): alpha cropped identically to RGB
  - stretch: no-op here (resampler handles both channels)
"""

from __future__ import annotations

import warnings

import numpy as np

from ..models import PipelineConfig, StageResult

# T1.3: when fit-mode padding would dominate (>4x aspect mismatch), emit a
# warning and auto-fallback to 'fill' to avoid producing a pure-bg output.
EXTREME_ASPECT_THRESHOLD: float = 4.0


def process(result: StageResult, config: PipelineConfig) -> StageResult:
    image = result.image
    alpha = result.alpha

    image, alpha = _handle_aspect_ratio(image, alpha, config)

    new_result = StageResult(
        image=image,
        palette=result.palette,
        metadata=dict(result.metadata),
        alpha=alpha,
    )
    new_result.metadata["preprocessor"] = {
        "aspect_ratio": config.aspect_ratio,
        "output_shape": tuple(image.shape),
        "grid_size": config.grid_size,
        "has_alpha": alpha is not None,
    }
    return new_result


def _handle_aspect_ratio(
    image: np.ndarray,
    alpha: np.ndarray | None,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    target_w, target_h = config.grid_size
    target_aspect = target_w / target_h
    h, w = image.shape[:2]
    current_aspect = w / h

    # T1.3: detect extreme aspect mismatch under fit mode
    if config.aspect_ratio == "fit":
        mismatch = max(current_aspect, target_aspect) / min(
            current_aspect, target_aspect
        )
        if mismatch > EXTREME_ASPECT_THRESHOLD:
            warnings.warn(
                f"Extreme aspect mismatch (source {w}x{h} aspect {current_aspect:.2f} "
                f"vs target aspect {target_aspect:.2f}, ratio {mismatch:.1f}x). "
                f"'fit' padding would dominate the output; falling back to 'fill' "
                f"to preserve content. Set aspect_ratio='stretch' or 'fill' "
                f"explicitly to silence this warning.",
                stacklevel=2,
            )
            return _fill_aspect_only(image, alpha, target_aspect)
        return _letterbox_aspect_only(image, alpha, target_aspect)

    if config.aspect_ratio == "fill":
        return _fill_aspect_only(image, alpha, target_aspect)
    return image, alpha


def _letterbox_aspect_only(
    image: np.ndarray, alpha: np.ndarray | None, target_aspect: float
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = image.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 0.01:
        return image, alpha

    if current_aspect > target_aspect:
        new_h = int(w / target_aspect)
        canvas = np.full((new_h, w, 3), 255, dtype=np.uint8)
        offset_y = (new_h - h) // 2
        canvas[offset_y : offset_y + h, :] = image
        if alpha is not None:
            alpha_canvas = np.zeros((new_h, w), dtype=np.uint8)
            alpha_canvas[offset_y : offset_y + h, :] = alpha
            alpha = alpha_canvas
        image = canvas
    else:
        new_w = int(h * target_aspect)
        canvas = np.full((h, new_w, 3), 255, dtype=np.uint8)
        offset_x = (new_w - w) // 2
        canvas[:, offset_x : offset_x + w] = image
        if alpha is not None:
            alpha_canvas = np.zeros((h, new_w), dtype=np.uint8)
            alpha_canvas[:, offset_x : offset_x + w] = alpha
            alpha = alpha_canvas
        image = canvas

    return image, alpha


def _fill_aspect_only(
    image: np.ndarray, alpha: np.ndarray | None, target_aspect: float
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = image.shape[:2]
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 0.01:
        return image, alpha

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        start_x = (w - new_w) // 2
        cropped = image[:, start_x : start_x + new_w]
        cropped_alpha = alpha[:, start_x : start_x + new_w] if alpha is not None else None
        return cropped, cropped_alpha

    new_h = int(w / target_aspect)
    start_y = (h - new_h) // 2
    cropped = image[start_y : start_y + new_h, :]
    cropped_alpha = alpha[start_y : start_y + new_h, :] if alpha is not None else None
    return cropped, cropped_alpha
