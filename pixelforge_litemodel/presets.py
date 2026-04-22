"""
PixelForge Lite — Preset Registry (14 presets, outline fields stripped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PipelineConfig


@dataclass
class _PresetDefinition:
    name: str
    grid_size: tuple[int, int]
    description: str
    max_colors: int | None = None
    palette: str | None = None
    dithering: bool = False
    dither_method: str = "floyd"

    def to_config(self) -> "PipelineConfig":
        from .models import PipelineConfig

        return PipelineConfig(
            grid_size=self.grid_size,
            max_colors=self.max_colors,
            palette=self.palette,
            dithering=self.dithering,
            dither_method=self.dither_method,
        )


class PresetRegistry:
    def __init__(self) -> None:
        self._presets: dict[str, _PresetDefinition] = {}
        self._register_all_presets()

    def _register_all_presets(self) -> None:
        self._register(_PresetDefinition(
            name="nes_sprite", grid_size=(8, 16),
            description="NES sprite format (8x16 grid, 4 colors)",
            max_colors=4, palette="nes_54",
        ))
        self._register(_PresetDefinition(
            name="gameboy", grid_size=(8, 8),
            description="Game Boy format (8x8 grid, 4 colors, Bayer dithering)",
            max_colors=4, palette="gameboy_4",
            dithering=True, dither_method="bayer",
        ))
        self._register(_PresetDefinition(
            name="snes_small", grid_size=(16, 16),
            description="SNES sprite small (16x16 grid, 16 colors)",
            max_colors=16, palette="snes_256",
        ))
        self._register(_PresetDefinition(
            name="snes_large", grid_size=(32, 32),
            description="SNES sprite large (32x32 grid, 16 colors)",
            max_colors=16, palette="snes_256",
        ))
        self._register(_PresetDefinition(
            name="gba", grid_size=(16, 16),
            description="Game Boy Advance sprite (16x16 grid, 16 colors)",
            max_colors=16,
        ))
        self._register(_PresetDefinition(
            name="stardew", grid_size=(16, 16),
            description="Stardew Valley style (16x16 grid, 24 colors)",
            max_colors=24,
        ))
        self._register(_PresetDefinition(
            name="cryptopunks", grid_size=(24, 24),
            description="CryptoPunk format (24x24 grid, 8 colors)",
            max_colors=8,
        ))
        self._register(_PresetDefinition(
            name="icon_32", grid_size=(32, 32),
            description="32x32 application icon", max_colors=32,
        ))
        self._register(_PresetDefinition(
            name="icon_48", grid_size=(48, 48),
            description="48x48 application icon", max_colors=32,
        ))
        self._register(_PresetDefinition(
            name="icon_64", grid_size=(64, 64),
            description="64x64 application icon", max_colors=48,
        ))
        self._register(_PresetDefinition(
            name="mid_128", grid_size=(128, 128),
            description="Medium resolution (128x128 grid, 48 colors)",
            max_colors=48,
        ))
        self._register(_PresetDefinition(
            name="mid_256", grid_size=(256, 256),
            description="High resolution (256x256 grid, 64 colors)",
            max_colors=64,
        ))
        self._register(_PresetDefinition(
            name="hd_384", grid_size=(384, 384),
            description="Ultra HD (384x384 grid, 64 colors)",
            max_colors=64,
        ))
        self._register(_PresetDefinition(
            name="hd_512", grid_size=(512, 512),
            description="Maximum resolution (512x512 grid, 64 colors)",
            max_colors=64,
        ))

    def _register(self, preset: _PresetDefinition) -> None:
        self._presets[preset.name] = preset

    def get(self, name: str) -> "PipelineConfig":
        if name not in self._presets:
            available = ", ".join(sorted(self._presets.keys()))
            raise KeyError(
                f"Preset '{name}' not found. Available presets: {available}"
            )
        return self._presets[name].to_config()

    def list_names(self) -> list[str]:
        return sorted(self._presets.keys())

    def get_info(self, name: str) -> dict:
        if name not in self._presets:
            available = ", ".join(sorted(self._presets.keys()))
            raise KeyError(
                f"Preset '{name}' not found. Available presets: {available}"
            )
        preset = self._presets[name]
        return {
            "name": preset.name,
            "grid_size": preset.grid_size,
            "description": preset.description,
        }
