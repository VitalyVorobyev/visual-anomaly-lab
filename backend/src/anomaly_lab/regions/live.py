"""One image through an unsaved region recipe, synchronously: the Prepare screen's live stage.

A build and a sampled check run as jobs; this is neither. It locates one image — or every
image of its sample when the recipe shares a crop — resolves the transform exactly as a
build would (`locate`, `unite_sample`), and returns the prepared frame inline. Nothing is
written. The extraction itself is pluggable: a classical extractor runs here, in the
caller's thread, and MobileSAM's box arrives from the resident worker (ADR-0026) and is
handed in as `Located` entries built by `located_from`.
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Image as ImageEntity
from anomaly_lab.domain.entities import SampleAlignment
from anomaly_lab.media import decode
from anomaly_lab.regions.base import RegionExtraction, RegionExtractionError, RegionExtractor
from anomaly_lab.regions.preparation import (
    RESAMPLE_FILTERS,
    Located,
    PreparationRecipe,
    Size,
    locate,
    unite_sample,
)
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform
from anomaly_lab.schemas import API_MODEL_CONFIG

# A live preview decodes whole source frames on the request path. Past this many pixels
# per image it is a batch job's work, not an interactive one's.
MAX_LIVE_SOURCE_PIXELS = 40_000_000


class RegionLivePreview(BaseModel):
    """What one image becomes under an unsaved recipe at one size."""

    model_config = API_MODEL_CONFIG

    image_id: int
    sample_id: int
    source_width: int
    source_height: int
    width: int = Field(description="Prepared frame width.")
    height: int = Field(description="Prepared frame height.")
    status: Literal["succeeded", "failed"]
    error: str | None = None
    region: PixelBounds | None = Field(
        default=None, description="The extractor's own box for this image, before padding."
    )
    transform: SpatialTransform | None = Field(
        default=None, description="The crop, resize and pad actually applied."
    )
    extractor_confidence: float | None = None
    extractor_metadata: dict[str, Any] = Field(default_factory=dict)
    united: int = Field(
        default=1, description="How many images of the sample the crop was united over."
    )
    prepared_png: str | None = Field(
        default=None,
        description="The prepared frame as a PNG data URL, exactly as a build writes it.",
    )
    elapsed_ms: float


class LiveSourceTooLargeError(ValueError):
    """An image too large to decode on the request path."""


def require_live_size(images: list[ImageEntity]) -> None:
    for image in images:
        if image.width * image.height > MAX_LIVE_SOURCE_PIXELS:
            raise LiveSourceTooLargeError(
                f"image {image.id} is {image.width}x{image.height}, more than the "
                f"{MAX_LIVE_SOURCE_PIXELS // 1_000_000} megapixels a live preview decodes; "
                "use Check 24"
            )


def members_for(
    recipe: PreparationRecipe, target: ImageEntity, sample_images: list[ImageEntity]
) -> list[ImageEntity]:
    """The images whose crops decide the target's: itself, or its whole sample when shared."""
    if recipe.sample_alignment is SampleAlignment.UNION:
        return sample_images
    return [target]


def locate_members(
    recipe: PreparationRecipe,
    size: Size,
    members: list[ImageEntity],
    extractor: RegionExtractor,
) -> list[Located]:
    """Locate every member in this thread: the classical extractors."""
    return [locate(recipe, size, record, extractor) for record in members]


def located_from(
    recipe: PreparationRecipe,
    size: Size,
    record: ImageEntity,
    extraction: RegionExtraction | None,
    error: str | None,
) -> Located:
    """A member located elsewhere — MobileSAM's resident — with its transform resolved here."""
    located = Located(record=record, started=time.perf_counter())
    if extraction is None:
        located.error = error or "the extractor returned no region"
        return located
    try:
        located.extraction = extraction
        located.transform = SpatialTransform.resolve(
            source_size=(record.width, record.height),
            prepared_size=size,
            region=extraction.bounds,
            padding_fraction=recipe.padding_fraction,
        )
    except Exception as exc:
        located.error = str(exc) or type(exc).__name__
    return located


def compose(
    recipe: PreparationRecipe,
    size: Size,
    target: ImageEntity,
    located: list[Located],
    *,
    started: float,
) -> RegionLivePreview:
    """Resolve the target's crop from its members and prepare its frame, as a build would."""
    own = next(item for item in located if item.record.id == target.id)
    if recipe.sample_alignment is SampleAlignment.UNION:
        entries = unite_sample(recipe, size, target.sample_id, located, None)
        entry = next(item for item in entries if item.image_id == target.id)
        transform, error = entry.transform, entry.error
    else:
        transform, error = own.transform, own.error
    extraction: RegionExtraction | None = own.extraction
    prepared_png: str | None = None
    if transform is not None:
        source = own.source if own.source is not None else _decode(target)
        prepared = transform.prepare_image(source, resample=RESAMPLE_FILTERS[recipe.resample])
        prepared_png = _data_url(prepared)
    return RegionLivePreview(
        image_id=target.id,
        sample_id=target.sample_id,
        source_width=target.width,
        source_height=target.height,
        width=size[0],
        height=size[1],
        status="failed" if transform is None else "succeeded",
        error=(error or "preparation failed") if transform is None else None,
        region=extraction.bounds if extraction is not None else None,
        transform=transform,
        extractor_confidence=extraction.confidence if extraction is not None else None,
        extractor_metadata=extraction.metadata if extraction is not None else {},
        united=len(located),
        prepared_png=prepared_png,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _decode(record: ImageEntity) -> Image.Image:
    source = decode.load(Path(record.path))
    if source.size != (record.width, record.height):
        raise RegionExtractionError(
            f"decoded size {source.size} differs from catalog {(record.width, record.height)}"
        )
    return source


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    # Lossless like the build, but fast: this is encoded on every keystroke's preview.
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
