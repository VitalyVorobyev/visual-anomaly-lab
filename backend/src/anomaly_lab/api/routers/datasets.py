"""Datasets, their channel dictionary, and their samples.

Every filter here is expressed over **samples**, even the ones a user thinks of in terms
of images: "channel = dark" means "samples having a dark-field image", never "these
images". The sample is the unit of identity and labelling (ADR-0005), and keeping the API
in those terms is what stops a screen from quietly reintroducing image-level thinking.

Nothing in this module counts channels. A dataset's channel list has whatever length it
has, and a two-channel sample is described by the same shape as a three-channel one.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.db.repositories.samples import SampleFilter
from anomaly_lab.domain.entities import Channel, Dataset, Image, Label, LabelSource, Sample, Subset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

MAX_PAGE_SIZE = 500


class ImageSummary(BaseModel):
    """One image of a sample, as the grid and the viewer need it.

    Carries no pixels: the client builds `/api/images/{id}/{tier}` URLs itself, so a page
    of thumbnails is that many cacheable image requests rather than one enormous JSON
    document.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    channel: str | None = None
    channel_id: int | None = None
    width: int
    height: int
    bit_depth: int
    file_size: int
    path: str


class SampleSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    dataset_id: int
    group_key: str
    external_id: str
    label: Label
    label_source: LabelSource
    notes: str | None = None
    images: list[ImageSummary] = Field(default_factory=list)


class SamplePage(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    limit: int
    offset: int
    items: list[SampleSummary] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    root_path: str
    adapter: str | None = None
    created_at: str
    notes: str | None = None
    samples: int
    images: int
    label_counts: dict[Label, int] = Field(default_factory=dict)


class DatasetDetail(DatasetSummary):
    manifest_path: str | None = None
    channels: list[Channel] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    splits: int = 0


class LabelUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: Label


def _dataset_summary(conn: sqlite3.Connection, dataset: Dataset) -> DatasetSummary:
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        root_path=dataset.root_path,
        adapter=dataset.adapter,
        created_at=dataset.created_at,
        notes=dataset.notes,
        samples=samples_repo.count_samples(conn, dataset.id),
        images=datasets_repo.count_images(conn, dataset.id),
        label_counts=datasets_repo.label_counts(conn, dataset.id),
    )


def _image_summary(image: Image, channel_names: dict[int, str]) -> ImageSummary:
    return ImageSummary(
        id=image.id,
        channel=channel_names.get(image.channel_id) if image.channel_id else None,
        channel_id=image.channel_id,
        width=image.width,
        height=image.height,
        bit_depth=image.bit_depth,
        file_size=image.file_size,
        path=image.path,
    )


def _sample_summary(
    sample: Sample, images: list[Image], channel_names: dict[int, str]
) -> SampleSummary:
    return SampleSummary(
        id=sample.id,
        dataset_id=sample.dataset_id,
        group_key=sample.group_key,
        external_id=sample.external_id,
        label=sample.label,
        label_source=sample.label_source,
        notes=sample.notes,
        images=[_image_summary(image, channel_names) for image in images],
    )


def _channel_names(conn: sqlite3.Connection, dataset_id: int) -> dict[int, str]:
    return {c.id: c.name for c in datasets_repo.list_channels(conn, dataset_id)}


def _require_dataset(conn: sqlite3.Connection, dataset_id: int) -> Dataset:
    dataset = datasets_repo.get_dataset(conn, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
    return dataset


@router.get("", summary="Every dataset, with its counts")
def list_datasets(request: Request) -> list[DatasetSummary]:
    """List the catalog."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        return [_dataset_summary(conn, dataset) for dataset in datasets_repo.list_datasets(conn)]


@router.get("/{dataset_id}", summary="One dataset, with its channels and capture groups")
def get_dataset(request: Request, dataset_id: int) -> DatasetDetail:
    """Everything the browser needs to build its filters."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset = _require_dataset(conn, dataset_id)
        return DatasetDetail(
            **_dataset_summary(conn, dataset).model_dump(),
            manifest_path=dataset.manifest_path,
            channels=datasets_repo.list_channels(conn, dataset_id),
            group_keys=samples_repo.list_group_keys(conn, dataset_id),
            splits=len(splits_repo.list_splits(conn, dataset_id)),
        )


@router.delete("/{dataset_id}", status_code=204, summary="Delete a dataset and its rows")
def delete_dataset(request: Request, dataset_id: int) -> None:
    """Remove the catalog entry. **Source images are never touched** (ADR-0001).

    Refused while an experiment references the dataset: an experiment freezes its
    configuration at creation, so orphaning one would leave results that cannot be
    explained (ADR-0005).
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        try:
            datasets_repo.delete_dataset(conn, dataset_id)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=(f"dataset {dataset_id} still has experiments attached; delete those first"),
            ) from exc


@router.get("/{dataset_id}/samples", summary="A page of samples, filtered")
def list_samples(
    request: Request,
    dataset_id: int,
    label: Label | None = None,
    channel_id: int | None = Query(
        default=None, description="Samples that have an image in this channel."
    ),
    split_id: int | None = None,
    subset: Subset | None = Query(
        default=None, description="Only meaningful together with `split_id`."
    ),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> SamplePage:
    """Page through a dataset's samples in a stable order.

    Ordered by capture group then by natural numeric identity, so `10` follows `9` rather
    than `1`, and so a scroll position still means the same thing after a re-import has
    inserted rows in the middle.
    """
    settings: Settings = request.app.state.settings
    filters = SampleFilter(label=label, channel_id=channel_id, split_id=split_id, subset=subset)

    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        total = samples_repo.count_samples(conn, dataset_id, filters)
        page = samples_repo.list_samples(conn, dataset_id, filters, limit=limit, offset=offset)
        names = _channel_names(conn, dataset_id)
        # One query for the whole page rather than one per sample.
        grouped = images_repo.list_images_for_samples(conn, [s.id for s in page])

    return SamplePage(
        total=total,
        limit=limit,
        offset=offset,
        items=[_sample_summary(s, grouped.get(s.id, []), names) for s in page],
    )


@router.get("/{dataset_id}/samples/{sample_id}", summary="One sample and all of its images")
def get_sample(request: Request, dataset_id: int, sample_id: int) -> SampleSummary:
    """One grouped sample. Its channel tabs are built from this list, whatever its length."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        sample = samples_repo.get_sample(conn, sample_id)
        if sample is None or sample.dataset_id != dataset_id:
            raise HTTPException(
                status_code=404, detail=f"no sample {sample_id} in dataset {dataset_id}"
            )
        names = _channel_names(conn, dataset_id)
        images = images_repo.list_images_for_sample(conn, sample.id)

    return _sample_summary(sample, images, names)


@router.patch("/{dataset_id}/samples/{sample_id}", summary="Set a sample's label")
def update_label(
    request: Request, dataset_id: int, sample_id: int, body: LabelUpdate
) -> SampleSummary:
    """Record an operator's label.

    Marks the sample `manual`, which is what makes the correction survive the next import
    of the same tree (ADR-0013).
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        existing = samples_repo.get_sample(conn, sample_id)
        if existing is None or existing.dataset_id != dataset_id:
            raise HTTPException(
                status_code=404, detail=f"no sample {sample_id} in dataset {dataset_id}"
            )
        updated = samples_repo.set_label(conn, sample_id, body.label)
        if updated is None:  # pragma: no cover - the row was read a moment ago
            raise HTTPException(status_code=404, detail=f"no sample {sample_id}")
        names = _channel_names(conn, dataset_id)
        images = images_repo.list_images_for_sample(conn, sample_id)

    return _sample_summary(updated, images, names)
