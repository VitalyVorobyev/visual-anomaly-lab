"""Explore — what a frozen encoder sees in one image, asked by clicking on it, and the
instances SAM 3 finds for a phrase.

The HTTP edge only. The arithmetic is `explore/grid.py`, the scratch maps `explore/store.py`,
and the encoder lives in the resident worker (ADR-0026) under the same lock as every other
resident request, so an Explore click can never share the accelerator with a running job.

Every request is validated here, torch-free, before it reaches the resident: a request the
child refuses kills the process (any deviation does), and that would throw away a loaded
encoder over a typo.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from anomaly_lab.annotation_bitmap import tight_bitmap_shape
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.annotations import BitmapShape
from anomaly_lab.domain.entities import Image
from anomaly_lab.experiments.service import refuse_while_a_job_runs
from anomaly_lab.explore.grid import (
    DEFAULT_CLUSTERS,
    MAX_CLUSTERS,
    MAX_POINTS,
    MIN_CLUSTERS,
    ExploreMode,
)
from anomaly_lab.explore.store import (
    MapKind,
    StoredGrid,
    explore_dir,
    instance_mask,
    map_path,
    mask_of,
    read_grid,
    to_source,
)
from anomaly_lab.explore.text import (
    DEFAULT_THRESHOLD,
    MAX_INSTANCES,
    MAX_PHRASE_LENGTH,
    TextExploreError,
    clean_phrase,
)
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentError, ResidentWorker
from anomaly_lab.media.overlay import (
    parse_label_colours,
    render_anomaly_map,
    render_label_map,
    render_rgb_image,
)
from anomaly_lab.model_assets.catalog import SAM3_KEY, ModelAssetSpec, gated_message, get_spec
from anomaly_lab.model_assets.store import ResolvedAsset, resolve_asset
from anomaly_lab.models.dino_backbone import ACCESS_REQUEST_URLS, BACKBONES, DinoBackbone
from anomaly_lab.regions.transform import SpatialTransform
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["explore"])

DEFAULT_BACKBONE = DinoBackbone.DINOV2_VIT_B14


def backbone_title(backbone: DinoBackbone) -> str:
    """`dinov2_vit_b14` as `DINOv2 ViT-B/14`, `…_reg4` with its registers named."""
    family, _, rest = backbone.value.partition("_vit_")
    size, _, suffix = rest.partition("_")
    title = f"{family.replace('dino', 'DINO')} ViT-{size[0].upper()}/{size[1:]}"
    return f"{title} {suffix}" if suffix else title


class ExploreBackbone(BaseModel):
    model_config = API_MODEL_CONFIG

    key: DinoBackbone
    title: str
    patch_size: int
    gated: bool
    available: bool
    reason: str | None = Field(
        default=None, description="Why a gated encoder cannot be used here, in words."
    )


class ExploreTextCapability(BaseModel):
    model_config = API_MODEL_CONFIG

    available: bool = Field(description="A phrase can be asked now.")
    reason: str | None = Field(
        default=None, description="Why it cannot, in words the reader can act on."
    )
    asset_key: str = Field(description="The catalogued checkpoint (`/api/model-assets`).")
    installable: bool = Field(
        description="The runtime is present and the checkpoint only needs downloading."
    )
    gated: bool
    access_url: str | None = None
    max_phrase_length: int = MAX_PHRASE_LENGTH
    default_threshold: float = DEFAULT_THRESHOLD


class ExploreCapability(BaseModel):
    model_config = API_MODEL_CONFIG

    runtime_available: bool
    available: bool
    reason: str | None = None
    default_backbone: DinoBackbone = DEFAULT_BACKBONE
    backbones: list[ExploreBackbone]
    text: ExploreTextCapability
    min_clusters: int = MIN_CLUSTERS
    max_clusters: int = MAX_CLUSTERS
    default_clusters: int = DEFAULT_CLUSTERS


class ExplorePoint(BaseModel):
    model_config = API_MODEL_CONFIG

    x: float = Field(ge=0, description="Source-image pixel column.")
    y: float = Field(ge=0, description="Source-image pixel row.")


class ExploreRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    mode: ExploreMode
    backbone: DinoBackbone = DEFAULT_BACKBONE
    points: list[ExplorePoint] = Field(default_factory=list, max_length=MAX_POINTS)
    negatives: list[ExplorePoint] = Field(default_factory=list, max_length=MAX_POINTS)
    k: int = Field(default=DEFAULT_CLUSTERS, ge=MIN_CLUSTERS, le=MAX_CLUSTERS)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ExploreResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    mode: ExploreMode
    backbone: DinoBackbone
    device: Literal["mps", "cpu"]
    grid_rows: int
    grid_cols: int
    feature_dim: int
    patch_size: int
    transform: SpatialTransform = Field(
        description="The contain-resize from the source image into the encoder's frame."
    )
    cached: bool = Field(description="The image's features were already encoded.")
    warm: bool = Field(description="The encoder was already loaded.")
    encode_ms: float
    compute_ms: float
    elapsed_ms: float
    map_id: str
    map_kind: MapKind
    map_url: str
    cells: list[int] | None = Field(
        default=None,
        description="Clusters only: each grid cell's cluster, row-major; 0 is off the image.",
    )
    clusters: int | None = None
    value_low: float | None = Field(
        default=None,
        description=(
            "Similarity only: the cosine the heatmap's coldest colour stands for — this "
            "image's median. The heatmap is coloured from `value_low` to `value_high`, its top "
            "percent; a threshold stays in absolute cosine."
        ),
    )
    value_high: float | None = Field(
        default=None, description="Similarity only: the cosine of the hottest colour."
    )


class ExploreTextRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    phrase: str = Field(
        min_length=1,
        max_length=MAX_PHRASE_LENGTH,
        description="What to find, in a few words: `candle`, `the cap`.",
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        gt=0.0,
        lt=1.0,
        description="The score an instance must reach — SAM 3's own, not a probability.",
    )


class ExploreBox(BaseModel):
    model_config = API_MODEL_CONFIG

    x0: float
    y0: float
    x1: float
    y1: float


class ExploreInstance(BaseModel):
    model_config = API_MODEL_CONFIG

    index: int = Field(ge=1, description="1-based, best first; the label it is drawn as.")
    score: float
    box: ExploreBox = Field(description="Source-image pixels.")
    area: int = Field(ge=1, description="Mask area in source-image pixels.")


class ExploreTextResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    phrase: str
    threshold: float
    device: Literal["mps", "cpu"]
    cached: bool = Field(description="The image's vision features were already encoded.")
    warm: bool = Field(description="SAM 3 was already loaded.")
    encode_ms: float
    prompt_ms: float
    elapsed_ms: float
    map_id: str | None = Field(
        default=None, description="The instance map; `None` when nothing was found."
    )
    map_url: str | None = None
    instances: list[ExploreInstance]
    dropped: int = Field(
        ge=0,
        description=f"Instances past the {MAX_INSTANCES} kept, lowest scores first.",
    )


class ExploreShapeRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    cluster: int | None = Field(default=None, ge=1, le=MAX_CLUSTERS)
    instance: int | None = Field(default=None, ge=1, le=MAX_INSTANCES)
    label_key: str = Field(
        default="defect", min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$"
    )


class ExploreShape(BaseModel):
    model_config = API_MODEL_CONFIG

    shape: BitmapShape
    area: int = Field(ge=1, description="Mask area in source-image pixels.")


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _hub_cached(settings: Settings, backbone: DinoBackbone) -> bool:
    name = BACKBONES[backbone].timm_name.split(".")[0]
    tag = BACKBONES[backbone].timm_name
    hub = settings.model_cache_dir / "huggingface" / "hub"
    return (hub / f"models--timm--{tag}").is_dir() or (hub / f"models--timm--{name}").is_dir()


def runtime_available() -> bool:
    """Whether the `dl` extra is installed: checked without importing torch."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("timm") is not None
    )


def text_runtime_available() -> bool:
    """Whether SAM 3 can run: the `dl` extra, which carries transformers."""
    return runtime_available() and importlib.util.find_spec("transformers") is not None


def hub_token_present() -> bool:
    """An HF_TOKEN, or an account signed in with `hf auth login`. Nothing about the token
    is read beyond whether it exists."""
    if os.environ.get("HF_TOKEN"):
        return True
    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    from huggingface_hub import get_token

    return get_token() is not None


def _sam3() -> ModelAssetSpec:
    spec = get_spec(SAM3_KEY)
    assert spec is not None, "SAM 3 is catalogued"
    return spec


def _text_capability(settings: Settings) -> tuple[ExploreTextCapability, ResolvedAsset]:
    spec = _sam3()
    resolved = resolve_asset(settings, spec)
    runtime = text_runtime_available()
    reason: str | None = None
    if not runtime:
        reason = f"Install the backend's 'dl' extra to run {spec.title}."
    elif not resolved.ready and resolved.reason == "missing" and resolved.source == "managed":
        reason = (
            f"{spec.title} is not downloaded: {spec.total_size / 1e9:.1f} GB, once, under the "
            f"{spec.license_name}."
        )
        if spec.gated and not hub_token_present():
            reason = gated_message(spec)
    elif not resolved.ready:
        reason = f"{spec.title} is not usable: {resolved.reason}."
    return (
        ExploreTextCapability(
            available=runtime and resolved.ready,
            reason=reason,
            asset_key=spec.key,
            installable=runtime and not resolved.ready and resolved.source == "managed",
            gated=spec.gated,
            access_url=spec.hub.access_url if spec.hub is not None else None,
        ),
        resolved,
    )


def _backbone_entry(settings: Settings, backbone: DinoBackbone) -> ExploreBackbone:
    spec = BACKBONES[backbone]
    available = (
        not spec.gated or bool(os.environ.get("HF_TOKEN")) or _hub_cached(settings, backbone)
    )
    reason = None
    if not available:
        reason = f"{spec.license_note} Request access at {ACCESS_REQUEST_URLS[backbone]}."
    return ExploreBackbone(
        key=backbone,
        title=backbone_title(backbone),
        patch_size=spec.patch_size,
        gated=spec.gated,
        available=available,
        reason=reason,
    )


@router.get("/api/explore/capability", summary="Whether Explore can run, and on which encoders")
def explore_capability(request: Request) -> ExploreCapability:
    settings = _settings(request)
    runtime = runtime_available()
    backbones = [_backbone_entry(settings, backbone) for backbone in DinoBackbone]
    return ExploreCapability(
        runtime_available=runtime,
        available=runtime and any(entry.available for entry in backbones),
        reason=None if runtime else "Install the backend's 'dl' extra to enable Explore.",
        backbones=backbones,
        text=_text_capability(settings)[0],
    )


@router.post(
    "/api/images/{image_id}/explore",
    summary="Ask a frozen encoder what it sees in one image",
)
async def explore_image(request: Request, image_id: int, body: ExploreRequest) -> ExploreResponse:
    settings = _settings(request)
    queue: JobQueue = request.app.state.job_queue
    await asyncio.to_thread(refuse_while_a_job_runs, settings, queue)
    image = await asyncio.to_thread(_image, settings, image_id)
    _validate(image, body)
    if not runtime_available():
        raise HTTPException(
            status_code=409, detail="Install the backend's 'dl' extra to enable Explore."
        )
    entry = _backbone_entry(settings, body.backbone)
    if not entry.available:
        raise HTTPException(status_code=409, detail=entry.reason)

    resident: ResidentWorker = request.app.state.resident
    started = time.perf_counter()
    payload: dict[str, object] = {
        "image_id": image.id,
        "mode": body.mode.value,
        "points": [point.model_dump(mode="json") for point in body.points],
        "negatives": [point.model_dump(mode="json") for point in body.negatives],
        "k": body.k,
        "seed": body.seed,
    }
    try:
        result, warm = await resident.explore(backbone=body.backbone.value, payload=payload)
    except ResidentError as exc:
        tail = resident.stderr_tail()
        raise HTTPException(status_code=503, detail=f"{exc}\n{tail}" if tail else str(exc)) from exc

    try:
        return ExploreResponse.model_validate(
            {
                **result,
                "warm": warm,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "map_url": f"/api/explore/maps/{result.get('map_id')}.png",
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Explore returned an invalid response") from exc


@router.post(
    "/api/images/{image_id}/explore/text",
    summary="Ask SAM 3 for the instances a phrase names in one image",
)
async def explore_text(
    request: Request, image_id: int, body: ExploreTextRequest
) -> ExploreTextResponse:
    settings = _settings(request)
    queue: JobQueue = request.app.state.job_queue
    await asyncio.to_thread(refuse_while_a_job_runs, settings, queue)
    image = await asyncio.to_thread(_image, settings, image_id)
    try:
        phrase = clean_phrase(body.phrase)
    except TextExploreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    capability, resolved = await asyncio.to_thread(_text_capability, settings)
    if not capability.available:
        raise HTTPException(status_code=409, detail=capability.reason)

    resident: ResidentWorker = request.app.state.resident
    started = time.perf_counter()
    payload: dict[str, object] = {
        "image_id": image.id,
        "phrase": phrase,
        "threshold": body.threshold,
    }
    try:
        result, warm = await resident.segment_text(
            asset_key=capability.asset_key, asset_path=resolved.path, payload=payload
        )
    except ResidentError as exc:
        tail = resident.stderr_tail()
        raise HTTPException(status_code=503, detail=f"{exc}\n{tail}" if tail else str(exc)) from exc

    map_id = result.get("map_id")
    instances = result.get("instances")
    try:
        return ExploreTextResponse.model_validate(
            {
                **result,
                "instances": [
                    {**entry, "index": index}
                    for index, entry in enumerate(
                        instances if isinstance(instances, list) else [], start=1
                    )
                ],
                "warm": warm,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "map_url": f"/api/explore/maps/{map_id}.png" if map_id else None,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="SAM 3 returned an invalid response") from exc


def _image(settings: Settings, image_id: int) -> Image:
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
    return image


def _validate(image: Image, body: ExploreRequest) -> None:
    """400 for a prompt that cannot mean anything on this image."""
    for point in [*body.points, *body.negatives]:
        if point.x >= image.width or point.y >= image.height:
            raise HTTPException(status_code=400, detail="a point lies outside the source image")
    if body.mode is ExploreMode.SIMILAR and not body.points:
        raise HTTPException(status_code=400, detail="similarity needs at least one positive point")


def _stored(settings: Settings, map_id: str) -> StoredGrid:
    try:
        path: Path = map_path(explore_dir(settings), map_id)
        return read_grid(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="no such explore map") from exc


@router.get(
    "/api/explore/maps/{map_id}.png",
    summary="An explore map as an overlay at the source image's size",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def explore_map(
    request: Request,
    map_id: str,
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    cluster: int | None = Query(default=None, ge=1, le=MAX_CLUSTERS),
    instance: int | None = Query(default=None, ge=1, le=MAX_INSTANCES),
    colours: str | None = Query(
        default=None,
        description=(
            "Comma-separated `rrggbb`, one per cluster or instance, or one for a threshold mask."
        ),
    ),
) -> Response:
    """Similarity over this image's own range; a threshold turns it into a filled mask.

    The heatmap is coloured over the range the response reported (`value_low` to
    `value_high`: this image's median to its top percent), so it is legible on any image;
    the threshold is absolute cosine. Clusters are drawn in the colours the
    client names, one per cluster, as a supervised run's label map is; `cluster` keeps that
    one alone. Instances are drawn the same way, one colour each, the better-scoring one on
    top where two overlap; `instance` draws that one's whole mask alone. False colour is
    opaque RGB. Every overlay is drawn once per click, so none is
    zlib-optimised.
    """
    stored = _stored(_settings(request), map_id)
    palette = _palette(colours)
    if stored.kind is MapKind.INSTANCES:
        try:
            labels = (
                instance_mask(stored, instance).astype(np.float32) * float(instance)
                if instance is not None
                else to_source(stored).astype(np.float32)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = render_label_map(labels, palette, truth=False, optimize=False)
    elif stored.kind is MapKind.RGB:
        payload = render_rgb_image(to_source(stored), optimize=False)
    elif stored.kind is MapKind.VALUES and threshold is None:
        payload = render_anomaly_map(
            to_source(stored), value_range=stored.value_range or (0.0, 1.0), optimize=False
        )
    elif stored.kind is MapKind.VALUES:
        mask = mask_of(stored, threshold=threshold)
        payload = render_label_map(
            mask.astype(np.float32), palette[:1], truth=False, optimize=False
        )
    else:
        labels = to_source(stored).astype(np.float32)
        if cluster is not None:
            labels = np.where(labels == cluster, labels, 0.0).astype(np.float32)
        payload = render_label_map(labels, palette, truth=False, optimize=False)
    return Response(content=payload, media_type="image/png", headers={"Cache-Control": "no-store"})


def _palette(colours: str | None) -> list[tuple[int, int, int]]:
    if colours is None:
        # A neutral grey rather than a palette of this layer's own: colours are the
        # interface's, and a caller that names none gets a visible, unopinionated answer.
        return [(139, 148, 155)]
    try:
        return parse_label_colours(colours)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/api/explore/maps/{map_id}/shape",
    summary="Turn an explore mask into an annotation candidate",
)
def explore_shape(request: Request, map_id: str, body: ExploreShapeRequest) -> ExploreShape:
    """A thresholded similarity, one cluster or one instance as a tight source-frame bitmap.

    The same shape MobileSAM's candidates are (`tight_bitmap_shape`), so the editor takes it
    as an assist candidate the person accepts or rejects; nothing is written here.
    """
    stored = _stored(_settings(request), map_id)
    try:
        mask = mask_of(
            stored, threshold=body.threshold, cluster=body.cluster, instance=body.instance
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    shape = tight_bitmap_shape(
        mask, shape_id=f"explore-{uuid.uuid4().hex}", label_key=body.label_key
    )
    if shape is None:
        raise HTTPException(status_code=422, detail="the mask is empty at this setting")
    return ExploreShape(shape=shape, area=int(mask.sum()))
