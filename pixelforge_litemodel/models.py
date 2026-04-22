"""
PixelForge Lite — Shared Data Models

Pixel-art-only variant derived from pixelforge_basemodel. Drops all fields and
structures related to:
  - rendering mode selection (mode is implicitly "pixel_art")
  - edge detection / outline composition (outline_*, canny_*, edge_map)
  - photo-path preprocessing (denoise_sigma)
  - orphan cleanup postprocessing (cleanup_orphans, orphan_threshold)
  - CTS (color transition sharpness) classification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# T2.3: shared validation routines used by both StageResult and ConvertResult.
# Module-level functions avoid inheritance/mixin complexity while keeping a
# single source of truth.

def _validate_image_array(image: Any) -> None:
    """Validate an (H, W, 3) uint8 RGB image array."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")


def _validate_palette_list(palette: Any, required: bool = False) -> None:
    """Validate an RGB palette list.

    If required=True, palette must be a non-None list (ConvertResult).
    If required=False, palette may be None (StageResult).
    """
    if palette is None:
        if required:
            raise TypeError("palette must be a list of RGB tuples")
        return
    if not isinstance(palette, list):
        raise TypeError(
            "palette must be a list of RGB tuples"
            if required
            else "palette must be a list or None"
        )
    for i, color in enumerate(palette):
        if not isinstance(color, (tuple, list)) or len(color) != 3:
            raise ValueError(f"palette[{i}] must be RGB tuple/list, got {color}")
        if not all(isinstance(x, int) and 0 <= x <= 255 for x in color):
            raise ValueError(
                f"palette[{i}] RGB values must be 0-255 integers, got {color}"
            )


def _validate_alpha_mask(alpha: Any, image_shape: tuple) -> None:
    """Validate an optional binary alpha mask against the image shape."""
    if alpha is None:
        return
    if not isinstance(alpha, np.ndarray):
        raise TypeError("alpha must be a numpy ndarray or None")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D (H, W), got shape {alpha.shape}")
    if alpha.shape != image_shape[:2]:
        raise ValueError(
            f"alpha shape {alpha.shape} must match image shape {image_shape[:2]}"
        )
    if alpha.dtype != np.uint8:
        raise ValueError(f"alpha dtype must be uint8, got {alpha.dtype}")


@dataclass
class InputProfile:
    """Stage 0 output — scaling state + palette state (no CTS, no input_type)."""

    unique_colors: int
    unique_color_ratio: float
    scaling_direction: str
    reduction_ratio: float
    palette_sufficient: bool
    existing_palette: list[tuple[int, int, int]] | None
    dominant_color: tuple[int, int, int] | None = None
    dominant_color_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.scaling_direction not in ("down", "up", "identity"):
            raise ValueError(
                f"scaling_direction must be 'down', 'up', or 'identity', "
                f"got '{self.scaling_direction}'"
            )
        if self.unique_colors < 0:
            raise ValueError(f"unique_colors must be >= 0, got {self.unique_colors}")
        if self.reduction_ratio <= 0:
            raise ValueError(f"reduction_ratio must be > 0, got {self.reduction_ratio}")


@dataclass
class PipelineConfig:
    """Pipeline-wide configuration for the lite pipeline."""

    grid_size: tuple[int, int]
    aspect_ratio: str = "fit"
    max_colors: int | None = None
    palette: str | list[tuple[int, int, int]] | None = None
    dithering: bool = False
    dither_method: str = "floyd"

    def __post_init__(self) -> None:
        self._validate()

    # T1.4: safety upper bound on grid dimensions to prevent accidental OOM
    MAX_GRID_DIM: int = 4096

    def _validate(self) -> None:
        if not isinstance(self.grid_size, (tuple, list)) or len(self.grid_size) != 2:
            raise ValueError("grid_size must be a tuple/list of 2 elements")
        if not all(
            isinstance(x, int) and 0 < x <= PipelineConfig.MAX_GRID_DIM
            for x in self.grid_size
        ):
            raise ValueError(
                f"grid_size elements must be positive integers "
                f"≤ {PipelineConfig.MAX_GRID_DIM}, got {self.grid_size}"
            )

        valid_aspect = {"fit", "fill", "stretch"}
        if self.aspect_ratio not in valid_aspect:
            raise ValueError(
                f"aspect_ratio must be one of {valid_aspect}, got '{self.aspect_ratio}'"
            )

        if self.max_colors is not None and not (1 <= self.max_colors <= 256):
            raise ValueError("max_colors must be between 1 and 256")

        if self.palette is not None:
            if isinstance(self.palette, str):
                pass
            elif isinstance(self.palette, list):
                # T1.4: reject empty palette list
                if len(self.palette) == 0:
                    raise ValueError("palette list must not be empty")
                for color in self.palette:
                    if not isinstance(color, (tuple, list)) or len(color) != 3:
                        raise ValueError("palette colors must be RGB tuples/lists")
                    if not all(isinstance(x, int) and 0 <= x <= 255 for x in color):
                        raise ValueError("palette RGB values must be 0-255 integers")
            else:
                raise ValueError("palette must be None, str, or list of RGB tuples")

        valid_methods = {"floyd", "bayer", "atkinson"}
        if self.dither_method not in valid_methods:
            raise ValueError(f"dither_method must be one of {valid_methods}")


@dataclass
class StageResult:
    """
    Per-stage carrier. `alpha` (optional) is a binary mask carried in parallel
    with `image` when the input was RGBA; shape matches image's (H, W), dtype
    uint8 with values {0, 255}. None for RGB-only inputs — in that case all
    alpha code paths short-circuit and behavior is bit-identical to RGB-only.
    """

    image: np.ndarray
    palette: list[tuple[int, int, int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    alpha: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_image_array(self.image)
        _validate_palette_list(self.palette, required=False)
        _validate_alpha_mask(self.alpha, self.image.shape)


@dataclass
class ConvertResult:
    """
    Public-facing pipeline result with save/to_bytes/to_pil helpers. When the
    input was RGBA, `alpha` (H, W) uint8 binary mask accompanies `image` and
    the serialization helpers emit RGBA PNG automatically.
    """

    image: np.ndarray
    palette: list[tuple[int, int, int]]
    config: PipelineConfig
    metadata: dict[str, Any] = field(default_factory=dict)
    alpha: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_image_array(self.image)
        _validate_palette_list(self.palette, required=True)
        _validate_alpha_mask(self.alpha, self.image.shape)

    def _to_pil_rgba(self):
        # DataFlow: combine self.image (H, W, 3) + self.alpha (H, W) → PIL RGBA
        from PIL import Image

        rgba = np.zeros((self.image.shape[0], self.image.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = self.image
        rgba[:, :, 3] = self.alpha
        return Image.fromarray(rgba, mode="RGBA")

    def save(self, path: str | Path, format: str | None = None) -> None:
        from PIL import Image

        if self.alpha is not None:
            pil_image = self._to_pil_rgba()
        else:
            pil_image = Image.fromarray(self.image, mode="RGB")
        if format is None:
            pil_image.save(str(path))
        else:
            pil_image.save(str(path), format=format)

    def to_bytes(self, format: str = "png") -> bytes:
        from io import BytesIO
        from PIL import Image

        if self.alpha is not None:
            pil_image = self._to_pil_rgba()
        else:
            pil_image = Image.fromarray(self.image, mode="RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format=format.upper())
        return buffer.getvalue()

    def to_pil(self) -> "Image.Image":  # type: ignore
        from PIL import Image

        if self.alpha is not None:
            return self._to_pil_rgba()
        return Image.fromarray(self.image, mode="RGB")
