"""A deterministic foreground-from-border baseline with no optional dependencies."""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.regions.base import RegionExtraction, RegionExtractionError, RegionExtractor
from anomaly_lab.regions.transform import PixelBounds
from anomaly_lab.schemas import API_MODEL_CONFIG


class ForegroundThresholdConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    border_fraction: float = Field(
        default=0.08,
        gt=0.0,
        le=0.25,
        description="Image-edge fraction used to estimate a uniform background level.",
    )
    min_contrast: int = Field(
        default=12,
        ge=1,
        le=255,
        description="Minimum absolute 8-bit luminance distance from the border background.",
    )
    min_area_fraction: float = Field(
        default=0.01,
        gt=0.0,
        le=0.5,
        description="Fail when the largest connected foreground region is smaller than this.",
    )
    connectivity: Literal[4, 8] = Field(
        default=8,
        description="Pixel connectivity used for the dominant component.",
    )
    max_analysis_side: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Bound the classical analysis grid; the returned box remains in source pixels.",
    )


class ForegroundThresholdExtractor(RegionExtractor):
    title = "Foreground threshold"
    summary = (
        "Classical baseline: estimate the border background, threshold luminance distance, "
        "and keep the largest connected component."
    )

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return ForegroundThresholdConfig

    def extract(self, image: np.ndarray) -> RegionExtraction:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise RegionExtractionError("region extractors require an RGB uint8 HxWx3 source image")
        config = ForegroundThresholdConfig.model_validate(self.config)
        source_height, source_width = image.shape[:2]
        scale = min(1.0, config.max_analysis_side / max(source_height, source_width))
        width = max(1, round(source_width * scale))
        height = max(1, round(source_height * scale))
        analysis_image = image
        if (width, height) != (source_width, source_height):
            analysis_image = np.asarray(
                Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        gray = np.rint(
            analysis_image[:, :, 0].astype(np.float32) * 0.299
            + analysis_image[:, :, 1].astype(np.float32) * 0.587
            + analysis_image[:, :, 2].astype(np.float32) * 0.114
        ).astype(np.uint8)
        border = max(1, round(min(height, width) * config.border_fraction))
        border_values = np.concatenate(
            (
                gray[:border].ravel(),
                gray[-border:].ravel(),
                gray[:, :border].ravel(),
                gray[:, -border:].ravel(),
            )
        )
        background = float(np.median(border_values))
        distance = np.abs(gray.astype(np.float32) - background).astype(np.uint8)
        threshold = max(config.min_contrast, _otsu_threshold(distance))
        foreground = distance >= threshold
        component = _largest_component(foreground, connectivity=config.connectivity)
        area = int(component.sum())
        min_area = max(1, round(height * width * config.min_area_fraction))
        if area < min_area:
            raise RegionExtractionError(
                f"largest foreground component has {area} pixels; at least {min_area} are required"
            )
        ys, xs = np.nonzero(component)
        scale_x = source_width / width
        scale_y = source_height / height
        return RegionExtraction(
            bounds=PixelBounds(
                left=float(xs.min()) * scale_x,
                top=float(ys.min()) * scale_y,
                right=(float(xs.max()) + 1.0) * scale_x,
                bottom=(float(ys.max()) + 1.0) * scale_y,
            ),
            confidence=area / (height * width),
            metadata={
                "background_luminance": background,
                "threshold": threshold,
                "area_pixels": area,
                "coverage_fraction": area / (height * width),
                "analysis_width": width,
                "analysis_height": height,
            },
        )


def _otsu_threshold(values: np.ndarray) -> int:
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = float(histogram.sum())
    if total == 0:
        return 0
    levels = np.arange(256, dtype=np.float64)
    total_mean = float(np.dot(levels, histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 0
    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (total_mean - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return best_threshold


def _largest_component(mask: np.ndarray, *, connectivity: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    neighbours = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if connectivity == 8:
        neighbours += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif connectivity != 4:
        raise RegionExtractionError("connectivity must be 4 or 8")

    for y, x in zip(*np.nonzero(mask), strict=True):
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue: deque[tuple[int, int]] = deque([(int(y), int(x))])
        component: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            component.append((current_y, current_x))
            for dy, dx in neighbours:
                next_y, next_x = current_y + dy, current_x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(component) > len(best):
            best = component

    result = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best, strict=True)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result
