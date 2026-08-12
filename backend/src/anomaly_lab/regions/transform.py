"""The invertible source-image to prepared-image geometry contract.

Points are expressed in pixel-centre coordinates: integer ``(x, y)`` names the centre
of source pixel ``[y, x]``. Crop bounds are half-open pixel-edge coordinates, matching
numpy and Pillow crops. Keeping both conventions explicit prevents the half-pixel drift
that otherwise appears when a contained resize is projected back onto the source.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from anomaly_lab.schemas import API_MODEL_CONFIG


class PixelBounds(BaseModel):
    """A half-open rectangle in source pixel-edge coordinates."""

    model_config = API_MODEL_CONFIG

    left: float
    top: float
    right: float
    bottom: float

    @model_validator(mode="after")
    def _is_finite_and_non_empty(self) -> PixelBounds:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("region bounds must be finite")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("region bounds must have positive width and height")
        return self


class SpatialTransform(BaseModel):
    """A fully resolved crop, contain-resize and symmetric edge-pad transform.

    Only integers that were actually used by the image operation are persisted. Derived
    scales therefore describe Pillow's realised resize rather than an ideal scale that
    may differ after integer rounding.
    """

    model_config = API_MODEL_CONFIG

    schema_version: int = Field(default=1, frozen=True)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    crop_left: int = Field(ge=0)
    crop_top: int = Field(ge=0)
    crop_right: int = Field(gt=0)
    crop_bottom: int = Field(gt=0)
    prepared_width: int = Field(gt=0)
    prepared_height: int = Field(gt=0)
    resized_width: int = Field(gt=0)
    resized_height: int = Field(gt=0)
    pad_left: int = Field(ge=0)
    pad_top: int = Field(ge=0)
    pad_right: int = Field(ge=0)
    pad_bottom: int = Field(ge=0)

    @model_validator(mode="after")
    def _is_consistent(self) -> SpatialTransform:
        if self.schema_version != 1:
            raise ValueError("unsupported spatial transform schema version")
        if not (self.crop_left < self.crop_right <= self.source_width):
            raise ValueError("horizontal crop bounds must lie inside the source image")
        if not (self.crop_top < self.crop_bottom <= self.source_height):
            raise ValueError("vertical crop bounds must lie inside the source image")
        if self.pad_left + self.resized_width + self.pad_right != self.prepared_width:
            raise ValueError("horizontal resize and padding do not fill the prepared image")
        if self.pad_top + self.resized_height + self.pad_bottom != self.prepared_height:
            raise ValueError("vertical resize and padding do not fill the prepared image")
        if abs(self.pad_left - self.pad_right) > 1 or abs(self.pad_top - self.pad_bottom) > 1:
            raise ValueError("contain padding must be symmetric to within one pixel")
        return self

    @property
    def crop_width(self) -> int:
        return self.crop_right - self.crop_left

    @property
    def crop_height(self) -> int:
        return self.crop_bottom - self.crop_top

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.crop_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.crop_height

    @classmethod
    def resolve(
        cls,
        *,
        source_size: tuple[int, int],
        prepared_size: tuple[int, int],
        region: PixelBounds | None = None,
        padding_fraction: float = 0.05,
    ) -> SpatialTransform:
        """Resolve a proposed region to reproducible integer image operations.

        ``padding_fraction`` expands each side by that fraction of the detected region's
        corresponding dimension before clipping to the source. ``region=None`` is the
        identity extractor: it selects the full source frame without additional crop
        expansion, while still using the shared contain-resize/pad operation.
        """
        source_width, source_height = source_size
        prepared_width, prepared_height = prepared_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("source dimensions must be positive")
        if prepared_width <= 0 or prepared_height <= 0:
            raise ValueError("prepared dimensions must be positive")
        if not math.isfinite(padding_fraction) or not 0.0 <= padding_fraction <= 1.0:
            raise ValueError("padding_fraction must be finite and between 0 and 1")

        if region is None:
            left, top, right, bottom = 0, 0, source_width, source_height
        else:
            region_width = region.right - region.left
            region_height = region.bottom - region.top
            left = max(0, math.floor(region.left - region_width * padding_fraction))
            top = max(0, math.floor(region.top - region_height * padding_fraction))
            right = min(source_width, math.ceil(region.right + region_width * padding_fraction))
            bottom = min(source_height, math.ceil(region.bottom + region_height * padding_fraction))
            if left >= right or top >= bottom:
                raise ValueError("region does not intersect the source image")

        crop_width = right - left
        crop_height = bottom - top
        ideal_scale = min(prepared_width / crop_width, prepared_height / crop_height)
        resized_width = max(1, min(prepared_width, round(crop_width * ideal_scale)))
        resized_height = max(1, min(prepared_height, round(crop_height * ideal_scale)))
        horizontal_gap = prepared_width - resized_width
        vertical_gap = prepared_height - resized_height
        pad_left = horizontal_gap // 2
        pad_top = vertical_gap // 2

        return cls(
            source_width=source_width,
            source_height=source_height,
            crop_left=left,
            crop_top=top,
            crop_right=right,
            crop_bottom=bottom,
            prepared_width=prepared_width,
            prepared_height=prepared_height,
            resized_width=resized_width,
            resized_height=resized_height,
            pad_left=pad_left,
            pad_top=pad_top,
            pad_right=horizontal_gap - pad_left,
            pad_bottom=vertical_gap - pad_top,
        )

    def source_to_prepared(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map one pixel centre from source to prepared coordinates."""
        x, y = point
        return (
            (x - self.crop_left + 0.5) * self.scale_x - 0.5 + self.pad_left,
            (y - self.crop_top + 0.5) * self.scale_y - 0.5 + self.pad_top,
        )

    def prepared_to_source(self, point: tuple[float, float]) -> tuple[float, float]:
        """Map one pixel centre from prepared back to source coordinates."""
        x, y = point
        return (
            (x - self.pad_left + 0.5) / self.scale_x - 0.5 + self.crop_left,
            (y - self.pad_top + 0.5) / self.scale_y - 0.5 + self.crop_top,
        )

    def prepare_image(self, image: Image.Image, *, resample: Image.Resampling) -> Image.Image:
        """Apply the resolved transform, using edge pixels for letterbox padding."""
        self._require_source_size(image.size)
        cropped = image.crop((self.crop_left, self.crop_top, self.crop_right, self.crop_bottom))
        resized = cropped.resize((self.resized_width, self.resized_height), resample)
        array = np.asarray(resized)
        pad_width: tuple[tuple[int, int], ...] = (
            (self.pad_top, self.pad_bottom),
            (self.pad_left, self.pad_right),
        )
        if array.ndim == 3:
            pad_width += ((0, 0),)
        prepared = np.pad(array, pad_width, mode="edge")
        return Image.fromarray(prepared, mode=resized.mode)

    def prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        """Project a source-frame boolean mask into the prepared frame."""
        self._require_source_shape(mask.shape)
        cropped = mask[self.crop_top : self.crop_bottom, self.crop_left : self.crop_right]
        resized = Image.fromarray(cropped.astype(np.uint8) * 255, mode="L").resize(
            (self.resized_width, self.resized_height), Image.Resampling.NEAREST
        )
        prepared = np.zeros((self.prepared_height, self.prepared_width), dtype=bool)
        prepared[
            self.pad_top : self.pad_top + self.resized_height,
            self.pad_left : self.pad_left + self.resized_width,
        ] = np.asarray(resized) > 0
        return prepared

    def project_mask(self, prepared_mask: np.ndarray) -> np.ndarray:
        """Project a prepared-frame boolean mask back into its source crop."""
        self._require_prepared_shape(prepared_mask.shape)
        unpadded = prepared_mask[
            self.pad_top : self.pad_top + self.resized_height,
            self.pad_left : self.pad_left + self.resized_width,
        ]
        restored = Image.fromarray(unpadded.astype(np.uint8) * 255, mode="L").resize(
            (self.crop_width, self.crop_height), Image.Resampling.NEAREST
        )
        source = np.zeros((self.source_height, self.source_width), dtype=bool)
        source[self.crop_top : self.crop_bottom, self.crop_left : self.crop_right] = (
            np.asarray(restored) > 0
        )
        return source

    def project_map(self, prepared_map: np.ndarray) -> np.ndarray:
        """Project a float anomaly map to source pixels; uncovered pixels are NaN."""
        self._require_prepared_shape(prepared_map.shape)
        unpadded = np.asarray(
            prepared_map[
                self.pad_top : self.pad_top + self.resized_height,
                self.pad_left : self.pad_left + self.resized_width,
            ],
            dtype=np.float32,
        )
        restored = Image.fromarray(unpadded, mode="F").resize(
            (self.crop_width, self.crop_height), Image.Resampling.BILINEAR
        )
        source = np.full((self.source_height, self.source_width), np.nan, dtype=np.float32)
        source[self.crop_top : self.crop_bottom, self.crop_left : self.crop_right] = np.asarray(
            restored, dtype=np.float32
        )
        return source

    def _require_source_size(self, size: tuple[int, int]) -> None:
        if size != (self.source_width, self.source_height):
            raise ValueError(
                f"source size {size} does not match transform "
                f"{(self.source_width, self.source_height)}"
            )

    def _require_source_shape(self, shape: tuple[int, ...]) -> None:
        expected = (self.source_height, self.source_width)
        if shape != expected:
            raise ValueError(f"source mask shape {shape} does not match transform {expected}")

    def _require_prepared_shape(self, shape: tuple[int, ...]) -> None:
        expected = (self.prepared_height, self.prepared_width)
        if shape != expected:
            raise ValueError(f"prepared shape {shape} does not match transform {expected}")
