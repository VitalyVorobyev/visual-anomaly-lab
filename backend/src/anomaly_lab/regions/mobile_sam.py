"""Automatic dominant-object extraction with the verified MobileSAM asset."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.regions.base import (
    RegionAvailability,
    RegionExtraction,
    RegionExtractionError,
    RegionExtractor,
)
from anomaly_lab.regions.transform import PixelBounds
from anomaly_lab.schemas import API_MODEL_CONFIG

ASSET_KEY = "mobile-sam-vit-t"


class MobileSamRegionConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    points_per_side: int = Field(
        default=8,
        ge=4,
        le=24,
        description="Automatic prompt grid width; cost grows quadratically.",
    )
    points_per_batch: int = Field(
        default=16,
        ge=1,
        le=64,
        description="Prompt batch size; lower it if accelerator memory is constrained.",
    )
    predicted_iou_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    stability_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    min_area_fraction: float = Field(default=0.02, gt=0.0, le=0.5)
    max_area_fraction: float = Field(default=0.98, ge=0.5, le=1.0)


class MobileSamRegionExtractor(RegionExtractor):
    title = "MobileSAM automatic region"
    summary = (
        "Generate prompt-grid masks with MobileSAM and choose the largest credible dominant object."
    )
    required_assets = (ASSET_KEY,)

    def __init__(self, config: BaseModel, *, assets: dict[str, Path] | None = None) -> None:
        super().__init__(config, assets=assets)
        self._generator: Any | None = None
        self._device: str | None = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return MobileSamRegionConfig

    @classmethod
    def availability(cls) -> RegionAvailability:
        available = (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("mobile_sam") is not None
        )
        return RegionAvailability(
            available=available,
            reason=None if available else "Install the backend's 'dl' extra to enable MobileSAM.",
        )

    def extract(self, image: np.ndarray) -> RegionExtraction:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise RegionExtractionError("region extractors require an RGB uint8 HxWx3 source image")
        generator = self._load_generator()
        try:
            records: list[dict[str, Any]] = generator.generate(image)
        except RuntimeError:
            if self._device != "mps":
                raise
            # The first real image is the only meaningful end-to-end smoke test. Rebuild
            # on CPU rather than move a generator that may retain failed device tensors.
            self._generator = None
            generator = self._load_generator(force_cpu=True)
            records = generator.generate(image)
        height, width = image.shape[:2]
        config = MobileSamRegionConfig.model_validate(self.config)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            fraction = float(record["area"]) / (height * width)
            if config.min_area_fraction <= fraction <= config.max_area_fraction:
                candidates.append((fraction, record))
        if not candidates:
            raise RegionExtractionError("MobileSAM found no mask within the configured area range")
        area_fraction, chosen = max(
            candidates,
            key=lambda item: (
                item[0],
                float(item[1]["predicted_iou"]),
                float(item[1]["stability_score"]),
            ),
        )
        x, y, box_width, box_height = (float(value) for value in chosen["bbox"])
        return RegionExtraction(
            bounds=PixelBounds(left=x, top=y, right=x + box_width, bottom=y + box_height),
            confidence=float(chosen["predicted_iou"]),
            metadata={
                "area_pixels": int(chosen["area"]),
                "coverage_fraction": area_fraction,
                "stability_score": float(chosen["stability_score"]),
                "device": self._device,
            },
        )

    def _load_generator(self, *, force_cpu: bool = False) -> Any:
        if self._generator is not None:
            return self._generator
        checkpoint = self.assets.get(ASSET_KEY)
        if checkpoint is None:
            raise RegionExtractionError(f"required model asset {ASSET_KEY!r} was not provided")
        package = importlib.import_module("mobile_sam")
        torch = importlib.import_module("torch")
        device = "mps" if not force_cpu and torch.backends.mps.is_available() else "cpu"
        model = package.sam_model_registry["vit_t"](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        config = MobileSamRegionConfig.model_validate(self.config)
        self._generator = package.SamAutomaticMaskGenerator(
            model,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.predicted_iou_threshold,
            stability_score_thresh=config.stability_threshold,
            crop_n_layers=0,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )
        self._device = device
        return self._generator
