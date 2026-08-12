"""Full-source control extractor."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from anomaly_lab.regions.base import RegionExtraction, RegionExtractionError, RegionExtractor
from anomaly_lab.regions.transform import PixelBounds
from anomaly_lab.schemas import API_MODEL_CONFIG


class IdentityConfig(BaseModel):
    model_config = API_MODEL_CONFIG


class IdentityExtractor(RegionExtractor):
    title = "Full source frame"
    summary = "Control profile: preserve the whole image before contain-resize and padding."

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return IdentityConfig

    def extract(self, image: np.ndarray) -> RegionExtraction:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RegionExtractionError("region extractors require an RGB HxWx3 source image")
        height, width = image.shape[:2]
        return RegionExtraction(
            bounds=PixelBounds(left=0, top=0, right=width, bottom=height),
            metadata={"coverage_fraction": 1.0},
        )
