"""Automatic dominant-object extraction with the verified MobileSAM asset."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

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
    # The two bounds are a window over the whole interval, not a partition of it.
    #
    # They used to meet at 0.5 — `min` capped there, `max` floored there — which is a
    # reasonable-looking split that quietly decides what the extractor can be used for.
    # Because the chosen mask is the *largest* one inside the window, a floor of 0.5 on the
    # maximum means a near-full-frame mask always qualifies and therefore always wins. On a
    # dataset where the part is smaller than its background, that made the extractor
    # unable to select the part at all: the only setting that would have excluded the frame
    # was the one the form refused to accept.
    min_area_fraction: float = Field(
        default=0.02,
        gt=0.0,
        lt=1.0,
        description=(
            "Ignore candidate masks smaller than this fraction of the image; "
            "raises the floor when small speckle is being chosen."
        ),
    )
    max_area_fraction: float = Field(
        default=0.98,
        gt=0.0,
        le=1.0,
        description=(
            "Ignore candidate masks larger than this fraction of the image. "
            "The largest surviving mask is chosen, so lower this to reject a "
            "near-full-frame background mask and select the part instead."
        ),
    )

    max_border_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Ignore candidate masks that cover more than this fraction of the image's "
            "one-pixel border. A background segment wraps the frame; an object does not. "
            "1.0 disables the test."
        ),
    )
    selection: Literal["largest", "union"] = Field(
        default="largest",
        description=(
            "How surviving masks become one region: the largest mask, or the box around "
            "every surviving mask, so a part made of several pieces is kept whole."
        ),
    )

    @model_validator(mode="after")
    def _window_must_admit_something(self) -> MobileSamRegionConfig:
        """The only relationship between the two that is real.

        Both fields now span the interval independently, so nothing else stops a window
        from being written inside out — and an empty window fails every image with
        "found no mask within the configured area range", which describes the symptom and
        not the typo.
        """
        if self.min_area_fraction >= self.max_area_fraction:
            raise ValueError(
                "min_area_fraction must be below max_area_fraction; "
                f"got {self.min_area_fraction} and {self.max_area_fraction}"
            )
        return self


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
        except (RuntimeError, TypeError) as exc:
            if self._device != "mps" or not can_retry_on_cpu(exc):
                raise
            # The first real image is the only meaningful end-to-end smoke test. Rebuild
            # on CPU rather than move a generator that may retain failed device tensors.
            self._generator = None
            generator = self._load_generator(force_cpu=True)
            records = generator.generate(image)
        height, width = image.shape[:2]
        config = MobileSamRegionConfig.model_validate(self.config)
        extraction = select_region(records, height, width, config)
        extraction.metadata["device"] = self._device
        return extraction

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


def border_fraction(segmentation: np.ndarray) -> float:
    """The fraction of the image's one-pixel border ring that a mask covers."""
    mask = np.asarray(segmentation, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return float(mask.mean())
    ring = np.concatenate([mask[0, :], mask[-1, :], mask[1:-1, 0], mask[1:-1, -1]])
    return float(ring.mean())


def select_region(
    records: list[dict[str, Any]], height: int, width: int, config: MobileSamRegionConfig
) -> RegionExtraction:
    """Turn MobileSAM's candidate masks into one source-frame region, or fail explicitly.

    A candidate survives when its area lies in the configured window and, unless
    `max_border_fraction` is 1.0, when it covers no more than that fraction of the frame's
    border. `largest` returns the biggest survivor; `union` returns the box around all of
    them. The border test needs each record's binary `segmentation`.
    """
    frame = height * width
    survivors: list[tuple[float, float, dict[str, Any]]] = []
    for record in records:
        fraction = float(record["area"]) / frame
        if not config.min_area_fraction <= fraction <= config.max_area_fraction:
            continue
        border = 0.0
        if config.max_border_fraction < 1.0:
            border = border_fraction(record["segmentation"])
            if border > config.max_border_fraction:
                continue
        survivors.append((fraction, border, record))
    if not survivors:
        raise RegionExtractionError(
            "MobileSAM found no mask within the configured area range and border limit"
        )
    largest = max(
        survivors,
        key=lambda item: (
            item[0],
            float(item[2]["predicted_iou"]),
            float(item[2]["stability_score"]),
        ),
    )
    chosen_items = survivors if config.selection == "union" else [largest]
    fraction = largest[0]
    members = [item[2] for item in chosen_items]
    boxes = [tuple(float(value) for value in member["bbox"]) for member in members]
    bounds = PixelBounds(
        left=min(box[0] for box in boxes),
        top=min(box[1] for box in boxes),
        right=max(box[0] + box[2] for box in boxes),
        bottom=max(box[1] + box[3] for box in boxes),
    )
    if len(members) > 1:
        union = np.zeros((height, width), dtype=bool)
        for member in members:
            union |= np.asarray(member["segmentation"], dtype=bool)
        fraction = float(union.sum()) / frame
    return RegionExtraction(
        bounds=bounds,
        # A union is only as credible as its least credible member.
        confidence=min(float(member["predicted_iou"]) for member in members),
        metadata={
            "selection": config.selection,
            "mask_count": len(members),
            "area_pixels": round(fraction * frame),
            "coverage_fraction": fraction,
            # Unmeasured, not zero, when the border test is disabled.
            "border_fraction": (
                max(item[1] for item in chosen_items) if config.max_border_fraction < 1.0 else None
            ),
            "stability_score": min(float(member["stability_score"]) for member in members),
        },
    )


def can_retry_on_cpu(exc: RuntimeError | TypeError) -> bool:
    """Recognise accelerator execution failures without swallowing arbitrary type bugs."""
    if isinstance(exc, RuntimeError):
        return True
    message = str(exc)
    return "MPS" in message and "float64" in message
