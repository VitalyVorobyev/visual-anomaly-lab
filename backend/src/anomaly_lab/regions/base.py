"""The dataset-agnostic region-extractor plugin interface (ADR-0033)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.regions.transform import PixelBounds
from anomaly_lab.schemas import API_MODEL_CONFIG


class RegionAvailability(BaseModel):
    model_config = API_MODEL_CONFIG

    available: bool = True
    reason: str | None = None


class RegionExtraction(BaseModel):
    """One extractor's dominant-object proposal in the source frame."""

    model_config = API_MODEL_CONFIG

    bounds: PixelBounds
    confidence: float | None = Field(
        default=None,
        description="Extractor-specific quality signal; never compared across extractor types.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class RegionExtractorDescription(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str
    title: str
    summary: str
    availability: RegionAvailability
    required_assets: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any]


class RegionExtractionError(RuntimeError):
    """An image could not be localised under this extractor's frozen configuration."""


class RegionExtractor(ABC):
    """One lazily loaded source-frame dominant-object localiser."""

    title: str
    summary: str
    required_assets: tuple[str, ...] = ()

    def __init__(self, config: BaseModel, *, assets: dict[str, Path] | None = None) -> None:
        self.config = config
        self.assets = assets or {}

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[BaseModel]: ...

    @classmethod
    def availability(cls) -> RegionAvailability:
        return RegionAvailability()

    @abstractmethod
    def extract(self, image: np.ndarray) -> RegionExtraction:
        """Return one region for an RGB uint8 source image, or fail explicitly."""
