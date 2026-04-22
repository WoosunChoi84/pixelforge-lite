"""
PixelForge Lite — Pipeline Orchestrator (4 stages).
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import numpy as np

from .models import InputProfile, PipelineConfig, StageResult
from .stages import input_analyzer, preprocessor, quantizer, resampler


def _normalize_metadata(result: StageResult) -> StageResult:
    """
    Task B: Replace non-serializable objects in metadata with primitive dicts.

    Currently the only known offender is `metadata['input_profile']`, an
    `InputProfile` dataclass instance carried through stages for algorithm
    decisions. After the pipeline completes we flatten it to a plain dict so
    that consumers can json.dumps(ConvertResult.metadata) without surprises.

    DataFlow: _normalize_metadata(result)
    IN:  result (StageResult) — may contain dataclass instances in metadata
    CHAIN: shallow copy metadata → for each dataclass value, convert via
           dataclasses.asdict [Serialization-Safe Projection]
    OUT: StageResult with primitive-only metadata values
    """
    normalized: dict[str, Any] = {}
    for key, value in result.metadata.items():
        if is_dataclass(value) and not isinstance(value, type):
            normalized[key] = asdict(value)
        else:
            normalized[key] = value
    return StageResult(
        image=result.image,
        palette=result.palette,
        metadata=normalized,
        alpha=result.alpha,
    )


class PipelineError(Exception):
    def __init__(self, message: str, stage_name: str = ""):
        self.stage_name = stage_name
        super().__init__(message)


class Pipeline:
    def __init__(self) -> None:
        # Note: AdaptiveTuner runs in PixelArtPipeline.__init__ (SDK layer),
        # not here — the Pipeline receives a fully-resolved PipelineConfig.
        self._execution_log: Dict[str, Any] = {}

    def run(
        self,
        image: np.ndarray,
        config: PipelineConfig,
        alpha: np.ndarray | None = None,
    ) -> StageResult:
        self._validate_input_image(image)
        self._validate_config(config)

        start_time = time.time()
        self._execution_log = {
            "start_time": start_time,
            "start_image_shape": image.shape,
            "has_alpha": alpha is not None,
            "config": {
                "grid_size": config.grid_size,
                "max_colors": config.max_colors,
                "aspect_ratio": config.aspect_ratio,
                "dithering": config.dithering,
            },
        }

        try:
            profile = self._run_stage_analyze(image, config)
            result = self._run_stage_preprocess(image, config, profile, alpha=alpha)
            result = self._run_stage_resample(result, config)
            result = self._run_stage_quantize(result, config)

            # Task B: ensure metadata is JSON-serializable by replacing the
            # InputProfile dataclass with a primitive dict. Internal stages
            # still use the dataclass via metadata during execution; this
            # normalization happens once at the pipeline boundary so consumers
            # can safely json.dumps(result.metadata) or log it.
            result = _normalize_metadata(result)

            end_time = time.time()
            self._execution_log["final_status"] = "success"
            self._execution_log["execution_time"] = end_time - start_time
            self._execution_log["final_shape"] = result.image.shape
            return result

        except PipelineError:
            raise
        except Exception as e:
            self._execution_log["final_status"] = "failed"
            raise PipelineError(f"Unexpected error in pipeline: {e}") from e

    def _validate_input_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy ndarray")
        if image.ndim != 3:
            raise ValueError(f"Input image must be 3D (H, W, 3), got {image.ndim}D")
        if image.shape[2] != 3:
            raise ValueError(
                f"Input image must have 3 channels (RGB), got {image.shape[2]}"
            )
        if image.dtype != np.uint8:
            raise ValueError(f"Input image dtype must be uint8, got {image.dtype}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(
                f"Input image dimensions must be positive, got {image.shape}"
            )

    def _validate_config(self, config: PipelineConfig) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig instance")

    def _run_stage_analyze(
        self, image: np.ndarray, config: PipelineConfig
    ) -> InputProfile:
        try:
            stage_start = time.time()
            profile = input_analyzer.analyze(image, config)
            duration = time.time() - stage_start
            self._execution_log["stage_0_analyze"] = {
                "scaling_direction": profile.scaling_direction,
                "reduction_ratio": profile.reduction_ratio,
                "unique_colors": profile.unique_colors,
                "unique_color_ratio": round(profile.unique_color_ratio, 4),
                "palette_sufficient": profile.palette_sufficient,
                "duration": duration,
                "status": "success",
            }
            return profile
        except Exception as e:
            self._execution_log["stage_0_analyze"] = {
                "status": "failed", "error": str(e),
            }
            raise PipelineError(
                f"Stage 'input_analyzer' failed: {e}",
                stage_name="input_analyzer",
            ) from e

    def _run_stage_preprocess(
        self,
        image: np.ndarray,
        config: PipelineConfig,
        profile: InputProfile,
        alpha: np.ndarray | None = None,
    ) -> StageResult:
        try:
            stage_start = time.time()
            initial_result = StageResult(
                image=image,
                metadata={"input_profile": profile},
                alpha=alpha,
            )
            result = preprocessor.process(initial_result, config)
            duration = time.time() - stage_start
            self._execution_log["stage_1_preprocess"] = {
                "input_shape": image.shape,
                "output_shape": result.image.shape,
                "duration": duration,
                "status": "success",
            }
            return result
        except Exception as e:
            self._execution_log["stage_1_preprocess"] = {
                "status": "failed", "error": str(e),
            }
            raise PipelineError(
                f"Stage 'preprocessor' failed: {e}", stage_name="preprocessor"
            ) from e

    def _run_stage_resample(
        self, result: StageResult, config: PipelineConfig
    ) -> StageResult:
        try:
            stage_start = time.time()
            result_out = resampler.process(result, config)
            duration = time.time() - stage_start
            self._execution_log["stage_2_resample"] = {
                "input_shape": result.image.shape,
                "output_shape": result_out.image.shape,
                "grid_size": config.grid_size,
                "aspect_ratio": config.aspect_ratio,
                "duration": duration,
                "status": "success",
            }
            return result_out
        except Exception as e:
            self._execution_log["stage_2_resample"] = {
                "status": "failed", "error": str(e),
            }
            raise PipelineError(
                f"Stage 'resampler' failed: {e}", stage_name="resampler"
            ) from e

    def _run_stage_quantize(
        self, result: StageResult, config: PipelineConfig
    ) -> StageResult:
        try:
            stage_start = time.time()
            result_out = quantizer.process(result, config)
            duration = time.time() - stage_start
            self._execution_log["stage_3_quantize"] = {
                "max_colors": config.max_colors,
                "palette_size": len(result_out.palette) if result_out.palette else 0,
                "dithering_enabled": config.dithering,
                "duration": duration,
                "status": "success",
            }
            return result_out
        except Exception as e:
            self._execution_log["stage_3_quantize"] = {
                "status": "failed", "error": str(e),
            }
            raise PipelineError(
                f"Stage 'quantizer' failed: {e}", stage_name="quantizer"
            ) from e

    def get_execution_log(self) -> Dict[str, Any]:
        return self._execution_log.copy()
