"""
PixelForge Lite — Palette Registry

Loads JSON palette files from ./data and exposes RGB lookup by name.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path


class PaletteRegistry:
    def __init__(self) -> None:
        self._palettes: dict[str, list[tuple[int, int, int]]] = {}
        self._load_palettes()

    def _load_palettes(self) -> None:
        data_dir = Path(__file__).parent / "data"
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Palette data directory not found at {data_dir}. "
                "Ensure palettes/data/*.json files exist."
            )

        # T1.5: isolate per-file failures — a single malformed palette JSON
        # must not take down the whole registry. Emit a warning and continue.
        for json_file in sorted(data_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    palette_data = json.load(f)

                if not isinstance(palette_data, dict):
                    raise ValueError("Palette file must contain a JSON object")

                name = palette_data.get("name")
                colors = palette_data.get("colors")

                if not name or not isinstance(name, str):
                    raise ValueError("Palette must have a 'name' field (string)")
                if not colors or not isinstance(colors, list):
                    raise ValueError("Palette must have a 'colors' field (list)")

                validated: list[tuple[int, int, int]] = []
                for i, color in enumerate(colors):
                    if not isinstance(color, (list, tuple)) or len(color) != 3:
                        raise ValueError(
                            f"Color {i} must be [R, G, B] list or tuple, got {color}"
                        )
                    r, g, b = color
                    if not all(isinstance(c, int) and 0 <= c <= 255 for c in (r, g, b)):
                        raise ValueError(
                            f"Color {i} RGB values must be 0-255 integers, got {color}"
                        )
                    validated.append((int(r), int(g), int(b)))

                self._palettes[name] = validated
            except (json.JSONDecodeError, ValueError, OSError, TypeError) as e:
                warnings.warn(
                    f"Skipping malformed palette file {json_file.name}: {e}",
                    stacklevel=2,
                )
                continue

        # At least one palette must load — if all files fail, raise.
        if len(self._palettes) == 0:
            raise FileNotFoundError(
                f"No valid palette files loaded from {data_dir}. "
                "Ensure at least one well-formed palette JSON exists."
            )

    def get(self, name: str) -> list[tuple[int, int, int]]:
        if name not in self._palettes:
            raise KeyError(
                f"Palette '{name}' not found. Available: {self.list_names()}"
            )
        return self._palettes[name]

    def list_names(self) -> list[str]:
        return sorted(self._palettes.keys())

    def register(self, name: str, colors: list[tuple[int, int, int]]) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Palette name must be a non-empty string")
        if not isinstance(colors, list) or len(colors) == 0:
            raise ValueError("Colors must be a non-empty list of (R, G, B) tuples")
        for i, color in enumerate(colors):
            if not isinstance(color, (tuple, list)) or len(color) != 3:
                raise ValueError(f"Color {i} must be (R, G, B) tuple, got {color}")
            r, g, b = color
            if not all(isinstance(c, int) and 0 <= c <= 255 for c in (r, g, b)):
                raise ValueError(
                    f"Color {i} RGB values must be 0-255 integers, got {color}"
                )
        self._palettes[name] = [(int(r), int(g), int(b)) for r, g, b in colors]
