"""Fixed centred crop for repeatably fixtured acquisitions."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.regions.base import RegionExtraction, RegionExtractionError, RegionExtractor
from anomaly_lab.regions.transform import PixelBounds
from anomaly_lab.schemas import API_MODEL_CONFIG


class CenterCropConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    width_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Crop width as a fraction of the source width.",
    )
    height_fraction: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Crop height as a fraction of the source height.",
    )
    center_x_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Horizontal crop centre as a fraction of the source width.",
    )
    center_y_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Vertical crop centre as a fraction of the source height.",
    )


class CenterCropExtractor(RegionExtractor):
    title = "Centred crop"
    summary = (
        "A fixed fractional window around a configured centre. For repeatably fixtured "
        "acquisitions where content-based detection disagrees between channels of one "
        "part, a deterministic window is the only extractor that is identical across "
        "them by construction."
    )

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return CenterCropConfig

    def extract(self, image: np.ndarray) -> RegionExtraction:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RegionExtractionError("region extractors require an RGB HxWx3 source image")
        config = CenterCropConfig.model_validate(self.config)
        height, width = image.shape[:2]
        crop_width = max(1, round(width * config.width_fraction))
        crop_height = max(1, round(height * config.height_fraction))
        center_x = width * config.center_x_fraction
        center_y = height * config.center_y_fraction
        # Clamp the window into the frame rather than failing at the border: a centre of
        # 0.0 with a half-frame window is a request for the left half, not an error.
        left = round(min(max(center_x - crop_width / 2, 0), width - crop_width))
        top = round(min(max(center_y - crop_height / 2, 0), height - crop_height))
        return RegionExtraction(
            bounds=PixelBounds(
                left=left, top=top, right=left + crop_width, bottom=top + crop_height
            ),
            metadata={"coverage_fraction": (crop_width * crop_height) / (width * height)},
        )
