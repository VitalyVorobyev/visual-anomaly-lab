"""Datasets, their channel dictionary, and their samples.

Every filter here is expressed over **samples**, even the ones a user thinks of in terms
of images: "channel = dark" means "samples having a dark-field image", never "these
images". The sample is the unit of identity and labelling (ADR-0005), and keeping the API
in those terms is what stops a screen from quietly reintroducing image-level thinking.

Nothing in this module counts channels. A dataset's channel list has whatever length it
has, and a two-channel sample is described by the same shape as a three-channel one.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.datasets.reference_packs import PackMembership, pack_membership
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.db.repositories.samples import SampleFilter
from anomaly_lab.domain.entities import (
    AnnotationScope,
    AnnotationState,
    Channel,
    Dataset,
    Image,
    Label,
    LabelSource,
    Sample,
    Subset,
)
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker
from anomaly_lab.media.cache import TIERS, cache_path
from anomaly_lab.owned_storage import StorageUsage, experiment_artifact_path, path_usage
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

MAX_PAGE_SIZE = 500


class ImageSummary(BaseModel):
    """One image of a sample, as the grid and the viewer need it.

    Carries no pixels: the client builds `/api/images/{id}/{tier}` URLs itself, so a page
    of thumbnails is that many cacheable image requests rather than one enormous JSON
    document.
    """

    model_config = API_MODEL_CONFIG

    id: int
    channel: str | None = None
    channel_id: int | None = None
    width: int
    height: int
    bit_depth: int
    file_size: int
    path: str


class SampleSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    group_key: str
    external_id: str
    label: Label
    label_source: LabelSource
    notes: str | None = None
    images: list[ImageSummary] = Field(default_factory=list)
    annotation: AnnotationState = AnnotationState.NONE


class SamplePage(BaseModel):
    model_config = API_MODEL_CONFIG

    total: int
    limit: int
    offset: int
    items: list[SampleSummary] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    name: str
    root_path: str
    adapter: str | None = None
    created_at: str
    notes: str | None = None
    samples: int
    images: int
    label_counts: dict[Label, int] = Field(default_factory=dict)
    # The three fields the catalogue reads. `collection` and `description` are *effective*
    # values -- the stored override if there is one, otherwise what the dataset's reference
    # pack supplies -- while `notes` above stays the raw override, because an editor has to
    # be able to tell "nothing written here" from "written, and identical to the default".
    collection: str | None = None
    description: str | None = None
    cover_image_id: int | None = None


class DatasetDetail(DatasetSummary):
    manifest_path: str | None = None
    channels: list[Channel] = Field(default_factory=list)
    group_keys: list[str] = Field(default_factory=list)
    splits: int = 0
    # Carried on the detail rather than the summary: the annotation editor already reads
    # the detail, and the catalogue grid has no use for it.
    annotation_scope: AnnotationScope = AnnotationScope.IMAGE


class DatasetUpdate(BaseModel):
    """The editable part of a dataset: what it is, and what it belongs with.

    Both fields are optional in the strong sense -- omitting one leaves the stored value
    alone, and sending `null` clears it. Name and root path are deliberately absent: they
    are the dataset's identity, unique in the schema, and the key a re-import resolves
    against (handbook import.md).
    """

    model_config = API_MODEL_CONFIG

    notes: str | None = Field(
        default=None,
        description="A sentence or two describing the dataset. Blank restores the default.",
    )
    collection: str | None = Field(
        default=None,
        description="The group this dataset is filed under. Blank restores the default.",
    )


class LabelUpdate(BaseModel):
    model_config = API_MODEL_CONFIG

    label: Label


class BulkLabelFilter(BaseModel):
    """The browser's filters, as a request body rather than a query string."""

    model_config = API_MODEL_CONFIG

    label: Label | None = None
    channel_id: int | None = None
    split_id: int | None = None
    subset: Subset | None = Field(
        default=None, description="Only meaningful together with `split_id`."
    )

    def to_filter(self) -> SampleFilter:
        return SampleFilter(
            label=self.label,
            channel_id=self.channel_id,
            split_id=self.split_id,
            subset=self.subset,
        )


class BulkLabelRequest(BaseModel):
    """Label a selection, or everything matching a filter.

    The two ways of naming the target are deliberately exclusive. A selection is what the
    grid's checkboxes produce; a filter is what "label everything I am currently looking
    at" means, and it is evaluated *server-side* from the same clause the grid pages with,
    so it cannot label a different set than the one whose count was shown.
    """

    model_config = API_MODEL_CONFIG

    label: Label
    sample_ids: list[int] | None = Field(
        default=None, description="An explicit selection. Ids outside this dataset are ignored."
    )
    filters: BulkLabelFilter | None = Field(
        default=None,
        description="Every sample matching these filters. An empty object means the whole dataset.",
    )

    @model_validator(mode="after")
    def _exactly_one_target(self) -> BulkLabelRequest:
        if (self.sample_ids is None) == (self.filters is None):
            msg = "provide exactly one of `sample_ids` or `filters`"
            raise ValueError(msg)
        return self


class BulkLabelResult(BaseModel):
    model_config = API_MODEL_CONFIG

    updated: int


class DatasetDeletionPreview(BaseModel):
    """Every app-owned consequence of deleting a dataset."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    name: str
    samples: int
    images: int
    splits: int
    experiments: int
    jobs: int
    region_profiles: int
    manual_labels: int
    generated_files: int
    generated_bytes: int
    active_jobs: list[JobSummary] = Field(default_factory=list)
    resident_loaded: bool = False
    storage_locations_safe: bool = True
    can_delete: bool
    blocker: str | None = None


class DatasetDeletionResult(BaseModel):
    """Completed row cascade and best-effort app-owned storage cleanup."""

    model_config = API_MODEL_CONFIG

    deleted: bool
    freed_files: int
    freed_bytes: int
    cleanup_errors: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _DeletionInventory:
    dataset: Dataset
    image_ids: list[int]
    experiment_ids: list[int]
    jobs: int
    owned_paths: list[Path]
    usage: StorageUsage
    storage_safe: bool
    samples: int
    splits: int
    region_profiles: int
    manual_labels: int


def _dataset_summary(
    conn: sqlite3.Connection,
    dataset: Dataset,
    membership: PackMembership | None = None,
) -> DatasetSummary:
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
        collection=_effective(dataset.collection, membership.collection if membership else None),
        description=_effective(dataset.notes, membership.description if membership else None),
        cover_image_id=datasets_repo.cover_image_id(conn, dataset.id),
    )


def _effective(override: str | None, derived: str | None) -> str | None:
    """The override if it says anything, else the derived value, else nothing.

    Blank is treated as absent so that clearing a field in the editor returns the derived
    value rather than leaving a dataset with an empty description where a real one exists.
    """
    if override is not None and override.strip():
        return override
    return derived or None


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
    sample: Sample,
    images: list[Image],
    channel_names: dict[int, str],
    annotation: AnnotationState = AnnotationState.NONE,
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
        annotation=annotation,
    )


def _channel_names(conn: sqlite3.Connection, dataset_id: int) -> dict[int, str]:
    return {c.id: c.name for c in datasets_repo.list_channels(conn, dataset_id)}


def _require_dataset(conn: sqlite3.Connection, dataset_id: int) -> Dataset:
    dataset = datasets_repo.get_dataset(conn, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
    return dataset


def _owned_manifest_path(settings: Settings, dataset: Dataset) -> Path | None:
    if not dataset.manifest_path:
        return None
    stored = Path(dataset.manifest_path).expanduser()
    expected_parent = settings.manifests_dir
    expected_prefix = f"dataset-{dataset.id}-"
    if (
        not stored.is_absolute()
        or ".." in stored.parts
        or settings.manifests_dir.is_symlink()
        or stored.parent != expected_parent
        or not stored.name.startswith(expected_prefix)
        or not stored.name.endswith(".json")
        or stored.is_symlink()
    ):
        return None
    return stored


def _deletion_inventory(
    conn: sqlite3.Connection, settings: Settings, dataset_id: int
) -> _DeletionInventory:
    dataset = _require_dataset(conn, dataset_id)
    images = images_repo.list_images_for_dataset(conn, dataset_id)
    experiments = experiments_repo.list_experiments_for_dataset(conn, dataset_id)
    region_profiles = region_profiles_repo.list_profiles(conn, dataset_id)
    jobs = jobs_repo.list_jobs_for_dataset(conn, dataset_id)

    paths: list[Path] = []
    safe = True
    for experiment in experiments:
        artifact_path = experiment_artifact_path(settings, experiment)
        if experiment.artifact_dir and artifact_path is None:
            safe = False
        elif artifact_path is not None:
            paths.append(artifact_path)
    for image in images:
        paths.extend(
            cache_path(settings, image.id, tier) for tier, spec in TIERS.items() if spec.cached
        )
        annotation_path = settings.annotation_image_dir(image.id)
        if settings.annotations_dir.is_symlink() or annotation_path.is_symlink():
            safe = False
        paths.append(annotation_path)
    if settings.region_profiles_dir.is_symlink():
        safe = False
    for profile in region_profiles:
        profile_path = settings.region_profile_dir(profile.id)
        if profile_path.is_symlink():
            safe = False
        paths.append(profile_path)
    paths.extend(
        settings.jobs_log_dir / f"{job.id}.log" for job in jobs if job.experiment_id is None
    )

    owned_manifest = _owned_manifest_path(settings, dataset)
    if dataset.manifest_path and owned_manifest is None:
        safe = False
    elif owned_manifest is not None:
        paths.append(owned_manifest)

    unique_paths = list(dict.fromkeys(paths))
    usage = sum((path_usage(path) for path in unique_paths), start=StorageUsage())
    manual_labels = int(
        conn.execute(
            "SELECT COUNT(*) FROM sample WHERE dataset_id = ? AND label_source = 'manual'",
            (dataset_id,),
        ).fetchone()[0]
    )
    return _DeletionInventory(
        dataset=dataset,
        image_ids=[image.id for image in images],
        experiment_ids=[experiment.id for experiment in experiments],
        jobs=len(jobs),
        owned_paths=unique_paths,
        usage=usage,
        storage_safe=safe,
        samples=samples_repo.count_samples(conn, dataset_id),
        splits=len(splits_repo.list_splits(conn, dataset_id)),
        region_profiles=len(region_profiles),
        manual_labels=manual_labels,
    )


def _remove_owned_path(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


@router.get("", summary="Every dataset, with its counts")
def list_datasets(request: Request) -> list[DatasetSummary]:
    """List the catalog."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        datasets = datasets_repo.list_datasets(conn)
        # Resolved once for the whole list rather than once per dataset: the matcher walks
        # every pack spec and resolves its roots on disk, which is cheap once and silly
        # thirteen times.
        membership = pack_membership(settings, datasets)
        return [_dataset_summary(conn, dataset, membership.get(dataset.id)) for dataset in datasets]


@router.get("/{dataset_id}", summary="One dataset, with its channels and capture groups")
def get_dataset(request: Request, dataset_id: int) -> DatasetDetail:
    """Everything the browser needs to build its filters."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        return _dataset_detail(conn, settings, _require_dataset(conn, dataset_id))


@router.patch("/{dataset_id}", summary="Edit a dataset's description and collection")
def update_dataset(request: Request, dataset_id: int, body: DatasetUpdate) -> DatasetDetail:
    """Write the fields the request actually named.

    A field left out of the body is untouched; a field sent as `null` or blank clears the
    override, which is how a dataset returns to the description and collection its
    reference pack supplies. `exclude_unset` is what keeps those two cases apart -- with a
    plain `is None` test, "do not change this" and "clear this" would arrive identically.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        fields: dict[str, str | None] = {
            name: (value.strip() or None) if isinstance(value, str) else None
            for name, value in body.model_dump(exclude_unset=True).items()
        }
        updated = datasets_repo.update_dataset(conn, dataset_id, fields=fields)
        if updated is None:  # pragma: no cover - existence was checked above
            raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
        return _dataset_detail(conn, settings, updated)


def _dataset_detail(
    conn: sqlite3.Connection, settings: Settings, dataset: Dataset
) -> DatasetDetail:
    membership = pack_membership(settings, [dataset]).get(dataset.id)
    return DatasetDetail(
        **_dataset_summary(conn, dataset, membership).model_dump(),
        manifest_path=dataset.manifest_path,
        channels=datasets_repo.list_channels(conn, dataset.id),
        group_keys=samples_repo.list_group_keys(conn, dataset.id),
        splits=len(splits_repo.list_splits(conn, dataset.id)),
        annotation_scope=dataset.annotation_scope,
    )


@router.get(
    "/{dataset_id}/deletion-preview",
    summary="Preview every app-owned consequence of deleting a dataset",
)
def preview_dataset_deletion(request: Request, dataset_id: int) -> DatasetDeletionPreview:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        inventory = _deletion_inventory(conn, settings, dataset_id)
        active = jobs_repo.active_jobs_for_dataset(conn, dataset_id)
    resident: ResidentWorker = request.app.state.resident
    snapshot = resident.snapshot()
    resident_loaded = snapshot is not None and snapshot.experiment_id in inventory.experiment_ids

    blocker: str | None = None
    if active:
        blocker = "Cancel or wait for active dataset work before deleting it."
    elif not inventory.storage_safe:
        blocker = "Stored app-owned paths do not match this dataset's expected locations."

    return DatasetDeletionPreview(
        dataset_id=dataset_id,
        name=inventory.dataset.name,
        samples=inventory.samples,
        images=len(inventory.image_ids),
        splits=inventory.splits,
        experiments=len(inventory.experiment_ids),
        jobs=inventory.jobs,
        region_profiles=inventory.region_profiles,
        manual_labels=inventory.manual_labels,
        generated_files=inventory.usage.files,
        generated_bytes=inventory.usage.bytes,
        active_jobs=[summary_of(job) for job in active],
        resident_loaded=resident_loaded,
        storage_locations_safe=inventory.storage_safe,
        can_delete=blocker is None,
        blocker=blocker,
    )


@router.delete("/{dataset_id}", summary="Delete a dataset and all app-owned descendants")
async def delete_dataset(request: Request, dataset_id: int) -> DatasetDeletionResult:
    """Delete catalog state and generated storage, never source images or source masks."""
    settings: Settings = request.app.state.settings
    queue: JobQueue = request.app.state.job_queue
    resident: ResidentWorker = request.app.state.resident

    async with queue.lifecycle_guard(), resident.eviction_guard():
        with connection(settings.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                # The write transaction begins before the inventory: a simultaneous
                # re-import or manual-label request cannot add rows after we have named
                # the consequences but before the cascade commits.
                inventory = _deletion_inventory(conn, settings, dataset_id)
                if not inventory.storage_safe:
                    raise HTTPException(
                        status_code=409,
                        detail="refusing dataset deletion because an app-owned path is unsafe",
                    )
                active = jobs_repo.active_jobs_for_dataset(conn, dataset_id)
                if active:
                    raise HTTPException(
                        status_code=409,
                        detail="cancel or wait for active dataset work before deleting it",
                    )
                deleted = datasets_repo.delete_dataset_rows(
                    conn, dataset_id, include_experiments=True
                )
                conn.execute("COMMIT")
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

        freed = StorageUsage()
        cleanup_errors: list[str] = []
        for path in inventory.owned_paths:
            usage = path_usage(path)
            try:
                await asyncio.to_thread(_remove_owned_path, path)
            except OSError as exc:
                cleanup_errors.append(f"{path.name}: {exc}")
            else:
                freed += usage

    return DatasetDeletionResult(
        deleted=deleted,
        freed_files=freed.files,
        freed_bytes=freed.bytes,
        cleanup_errors=cleanup_errors,
    )


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
    annotated: bool | None = Query(
        default=None,
        description=(
            "`false` lists samples with at least one image still lacking ground truth — "
            "what an annotation queue asks for. Omit for both."
        ),
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
    filters = SampleFilter(
        label=label,
        channel_id=channel_id,
        split_id=split_id,
        subset=subset,
        annotated=annotated,
    )

    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        total = samples_repo.count_samples(conn, dataset_id, filters)
        page = samples_repo.list_samples(conn, dataset_id, filters, limit=limit, offset=offset)
        names = _channel_names(conn, dataset_id)
        # One query for the whole page rather than one per sample, for both reads.
        sample_ids = [s.id for s in page]
        grouped = images_repo.list_images_for_samples(conn, sample_ids)
        annotation = annotations_repo.annotation_state_for_samples(conn, sample_ids)

    return SamplePage(
        total=total,
        limit=limit,
        offset=offset,
        items=[
            _sample_summary(
                s, grouped.get(s.id, []), names, annotation.get(s.id, AnnotationState.NONE)
            )
            for s in page
        ],
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
        annotation = annotations_repo.annotation_state_for_samples(conn, [sample.id])

    return _sample_summary(sample, images, names, annotation.get(sample.id, AnnotationState.NONE))


@router.patch("/{dataset_id}/samples/{sample_id}", summary="Set a sample's label")
def update_label(
    request: Request, dataset_id: int, sample_id: int, body: LabelUpdate
) -> SampleSummary:
    """Record an operator's label.

    Marks the sample `manual`, which is what makes the correction survive the next import
    of the same tree (handbook import.md).
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
        annotation = annotations_repo.annotation_state_for_samples(conn, [sample_id])

    return _sample_summary(updated, images, names, annotation.get(sample_id, AnnotationState.NONE))


@router.patch("/{dataset_id}/samples", summary="Label many samples at once")
def update_labels(request: Request, dataset_id: int, body: BulkLabelRequest) -> BulkLabelResult:
    """Label a selection, or everything matching a filter, in one request.

    Labelling one sample at a time is fine for correcting a handful and hopeless for a
    directory that is entirely one class — which is the common case on import, and the
    reason this exists. Like the single-sample route it marks every row it touches
    `manual`, so a re-import of the same tree cannot undo the work (handbook import.md).

    The filter form is resolved server-side from the grid's own `_where` clause, so the
    set that gets labelled is provably the set whose count the UI displayed.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        if body.sample_ids is not None:
            updated = samples_repo.set_labels(conn, dataset_id, body.sample_ids, body.label)
        elif body.filters is not None:
            updated = samples_repo.set_labels_matching(
                conn, dataset_id, body.filters.to_filter(), body.label
            )
        else:  # pragma: no cover - the request validator rejects this shape
            raise HTTPException(
                status_code=422, detail="provide exactly one of `sample_ids` or `filters`"
            )

    return BulkLabelResult(updated=updated)
