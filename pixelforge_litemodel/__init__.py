"""
PixelForge Lite — Public SDK API.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import ConvertResult, PipelineConfig
from .palettes import PaletteRegistry
from .pipeline import Pipeline
from .presets import PresetRegistry
from .tuning import AdaptiveTuner
from .utils import composite_rgba_on_white

__version__ = "0.1.0"


class PixelArtPipeline:
    """Pixel-art conversion pipeline (lite)."""

    def __init__(
        self,
        preset: str | None = None,
        grid_size: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        if preset is None and grid_size is None:
            available = ", ".join(list_presets())
            raise ValueError(
                f"Either 'preset' or 'grid_size' is required. "
                f"Use preset='{available.split(',')[0].strip()}' or grid_size=(32, 32). "
                f"Available presets: {available}"
            )

        if preset is not None:
            try:
                base_config = PresetRegistry().get(preset)
            except KeyError as e:
                available = ", ".join(list_presets())
                raise ValueError(
                    f"Invalid preset: '{preset}'. Available presets: {available}"
                ) from e
        else:
            base_config = None  # sentinel — validated below once grid_size is parsed

        # Validate explicit grid_size (if given) — applies whether preset is used or not.
        if grid_size is not None:
            if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
                raise ValueError(
                    f"grid_size must be a tuple (width, height), got {grid_size}"
                )
            if not all(isinstance(x, int) and x > 0 for x in grid_size):
                raise ValueError(
                    f"grid_size elements must be positive integers, got {grid_size}"
                )

        if base_config is None:
            base_config = PipelineConfig(grid_size=tuple(grid_size))  # type: ignore

        config_dict = {
            "grid_size": base_config.grid_size,
            "aspect_ratio": base_config.aspect_ratio,
            "max_colors": base_config.max_colors,
            "palette": base_config.palette,
            "dithering": base_config.dithering,
            "dither_method": base_config.dither_method,
        }
        # Explicit grid_size overrides the preset's grid_size (T1.2 fix).
        if grid_size is not None:
            config_dict["grid_size"] = tuple(grid_size)

        # kwargs override preset/grid_size-derived values.
        config_dict.update(kwargs)

        config = PipelineConfig(**config_dict)
        self.config = AdaptiveTuner().resolve(config)

    def convert(
        self, source: str | Path | np.ndarray | bytes
    ) -> ConvertResult:
        image_array, alpha = self._normalize_input(source)
        pipeline = Pipeline()
        stage_result = pipeline.run(image_array, self.config, alpha=alpha)
        return ConvertResult(
            image=stage_result.image,
            palette=stage_result.palette if stage_result.palette is not None else [],
            config=self.config,
            metadata=stage_result.metadata,
            alpha=stage_result.alpha,
        )

    def convert_bytes(self, data: bytes, format: str = "png") -> bytes:
        return self.convert(data).to_bytes(format=format)

    def _normalize_input(
        self, source: str | Path | np.ndarray | bytes
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        DataFlow: _normalize_input(source)
        IN:  str|Path|bytes|ndarray
        CHAIN: load via PIL (path/bytes) or validate directly (ndarray)
               → if RGBA: composite_rgba_on_white → (rgb, binary_alpha)
               → else: convert to RGB, alpha=None
        OUT: (rgb_array (H, W, 3) uint8, alpha (H, W) uint8 | None)
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            try:
                pil_image = Image.open(path)
            except Exception as e:
                raise ValueError(f"Failed to open image at {path}: {e}") from e
        elif isinstance(source, bytes):
            try:
                pil_image = Image.open(BytesIO(source))
            except Exception as e:
                raise ValueError(f"Failed to open image from bytes: {e}") from e
        elif isinstance(source, np.ndarray):
            if source.dtype != np.uint8:
                raise ValueError(f"NumPy array must be uint8, got {source.dtype}")
            if source.ndim == 3 and source.shape[2] == 4:
                rgb, alpha = composite_rgba_on_white(source)
                return rgb, alpha
            if source.ndim != 3 or source.shape[2] != 3:
                raise ValueError(
                    f"NumPy array must have shape (H, W, 3) or (H, W, 4), got {source.shape}"
                )
            return source, None
        else:
            raise ValueError(
                f"source must be str, Path, np.ndarray, or bytes, got {type(source)}"
            )

        # RGBA → alpha-composite on white + binary mask
        if pil_image.mode == "RGBA":
            rgba = np.array(pil_image, dtype=np.uint8)
            return composite_rgba_on_white(rgba)

        # Other non-RGB modes (L, P, etc.) → straight convert
        if pil_image.mode != "RGB":
            # Preserve alpha if palette mode carries it (e.g., 'PA', 'LA')
            if "A" in pil_image.mode:
                rgba = np.array(pil_image.convert("RGBA"), dtype=np.uint8)
                return composite_rgba_on_white(rgba)
            pil_image = pil_image.convert("RGB")

        image_array = np.array(pil_image, dtype=np.uint8)
        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                f"Converted image has unexpected shape: {image_array.shape}"
            )
        return image_array, None


def list_presets() -> list[str]:
    return PresetRegistry().list_names()


def list_palettes() -> list[str]:
    return PaletteRegistry().list_names()


def get_preset_info(name: str) -> dict[str, Any]:
    return PresetRegistry().get_info(name)


__all__ = [
    "PixelArtPipeline",
    "ConvertResult",
    "PipelineConfig",
    "list_presets",
    "list_palettes",
    "get_preset_info",
    "__version__",
]
