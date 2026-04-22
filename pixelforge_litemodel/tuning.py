"""
PixelForge Lite — Adaptive Parameter Tuning

Only max_colors is auto-computed in the lite pipeline. Uses logarithmic
interpolation across 9 reference grid sizes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import PipelineConfig


class AdaptiveTuner:
    _REFERENCE_POINTS: list[tuple[int, int]] = [
        (8, 12),
        (16, 16),
        (32, 24),
        (48, 28),
        (64, 24),
        (128, 32),
        (256, 48),
        (384, 56),
        (512, 64),
    ]

    def resolve(self, config: "PipelineConfig") -> "PipelineConfig":
        target_grid = min(config.grid_size[0], config.grid_size[1])
        if target_grid <= 0:
            raise ValueError(
                f"grid_size must be positive, got min dimension {target_grid}"
            )

        if config.max_colors is not None:
            return config

        log_target = math.log2(target_grid)
        lower, upper, t = self._find_interpolation_range(log_target)
        _, max_l = lower
        _, max_u = upper
        max_colors = int(round(max_l + t * (max_u - max_l)))

        # T1.4: clamp max_colors so it can never exceed the number of output pixels
        output_pixels = config.grid_size[0] * config.grid_size[1]
        max_colors = max(1, min(max_colors, output_pixels))

        return replace(config, max_colors=max_colors)

    def _find_interpolation_range(
        self, log_target: float
    ) -> tuple[tuple[int, int], tuple[int, int], float]:
        log_sizes = [
            (math.log2(size), (size, max_c)) for size, max_c in self._REFERENCE_POINTS
        ]

        if log_target <= log_sizes[0][0]:
            return self._REFERENCE_POINTS[0], self._REFERENCE_POINTS[1], 0.0
        if log_target >= log_sizes[-1][0]:
            return self._REFERENCE_POINTS[-2], self._REFERENCE_POINTS[-1], 1.0

        for i in range(len(log_sizes) - 1):
            log_l, _ = log_sizes[i]
            log_u, _ = log_sizes[i + 1]
            if log_l <= log_target <= log_u:
                t = (log_target - log_l) / (log_u - log_l)
                return self._REFERENCE_POINTS[i], self._REFERENCE_POINTS[i + 1], t

        return self._REFERENCE_POINTS[-2], self._REFERENCE_POINTS[-1], 1.0
