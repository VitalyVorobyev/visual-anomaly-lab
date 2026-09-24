"""Bounded preview and atomic full preparation for one region profile revision."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.domain.entities import Image as ImageEntity
from anomaly_lab.domain.entities import RegionProfileRevision, SampleAlignment, SpatialResample
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.media import decode
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset, sha256_file
from anomaly_lab.models.base import evenly_spaced
from anomaly_lab.regions.base import RegionExtractionError
from anomaly_lab.regions.registry import build as build_extractor
from anomaly_lab.regions.registry import get_extractor_class
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform
from anomaly_lab.schemas import API_MODEL_CONFIG

PREVIEW_LIMIT = 24
SUMMARY_FILENAME = "summary.json"
MANIFEST_FILENAME = "transforms.jsonl"

_RESAMPLE_FILTERS = {
    SpatialResample.NEAREST: Image.Resampling.NEAREST,
    SpatialResample.BILINEAR: Image.Resampling.BILINEAR,
    SpatialResample.BICUBIC: Image.Resampling.BICUBIC,
    SpatialResample.LANCZOS: Image.Resampling.LANCZOS,
}


class RegionPreparationEntry(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    source_sha256: str
    status: Literal["succeeded", "failed"]
    transform: SpatialTransform | None = None
    prepared_sha256: str | None = None
    extractor_confidence: float | None = None
    extractor_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    elapsed_ms: float | None = None


class RegionBuildSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    schema_version: int = 1
    profile_id: int
    dataset_id: int
    total: int
    succeeded: int
    failed: int
    manifest_sha256: str
    config_sha256: str
    storage_files: int
    storage_bytes: int
    elapsed_ms: float
    preview_entries: list[RegionPreparationEntry] = Field(default_factory=list, max_length=24)
    failure_examples: list[RegionPreparationEntry] = Field(default_factory=list, max_length=24)


@dataclass(frozen=True)
class PreparedRegionBuild:
    """A verified immutable materialization ready to be handed to an experiment."""

    root: Path
    profile: RegionProfileRevision
    summary: RegionBuildSummary
    entries: dict[int, RegionPreparationEntry]

    def image_path(self, image_id: int) -> Path:
        entry = self.entries.get(image_id)
        if entry is None or entry.prepared_sha256 is None:
            raise ValueError(f"region build has no prepared image {image_id}")
        path = self.root / "images" / f"{image_id}.png"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"prepared image {image_id} is missing from the immutable build")
        actual_digest = sha256_file(path)
        if actual_digest != entry.prepared_sha256:
            raise ValueError(
                f"prepared image {image_id} changed: "
                f"expected {entry.prepared_sha256}, found {actual_digest}"
            )
        return path

    def transform_for(self, image_id: int) -> SpatialTransform:
        entry = self.entries.get(image_id)
        if entry is None or entry.transform is None:
            raise ValueError(f"region build has no transform for image {image_id}")
        return entry.transform


def preview_indices(channels: Sequence[int | None], limit: int = PREVIEW_LIMIT) -> list[int]:
    """Spread a preview over the whole image list, never its first folder — and every channel.

    `channels` is each image's channel in list order. One stride over an interleaved
    multi-channel list can resonate with the interleave and land on only some channels, so
    the budget is shared between channels first and spread within each one.
    """
    groups: dict[int | None, list[int]] = {}
    for index, channel in enumerate(channels):
        groups.setdefault(channel, []).append(index)
    if len(groups) <= 1 or len(channels) <= limit:
        return evenly_spaced(len(channels), limit)
    members = list(groups.values())
    # Equal shares, and what a small channel cannot use goes to the others.
    shares = [0] * len(members)
    while sum(shares) < limit:
        open_groups = [i for i, group in enumerate(members) if shares[i] < len(group)]
        for index in open_groups[: limit - sum(shares)]:
            shares[index] += 1
    chosen: list[int] = []
    for group, share in zip(members, shares, strict=True):
        chosen.extend(group[position] for position in evenly_spaced(len(group), share))
    return sorted(chosen)


def preview_sample_indices(sample_ids: Sequence[int], limit: int = PREVIEW_LIMIT) -> list[int]:
    """Spread a preview over whole samples, for a profile that unites a sample's crops.

    A union crop is computed from every image of a sample, so a preview that showed one
    channel of a part would show a crop it did not compute. The budget is spent on complete
    samples instead, evenly spaced over the dataset: as many as fit in `limit` images. Every
    channel of each chosen sample is included by construction, which is the stratification
    `preview_indices` has to arrange explicitly when it picks single images. One sample is
    always chosen, even if it alone holds more than `limit` images; the caller reports at
    most `limit`.
    """
    groups: dict[int, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(sample_id, []).append(index)
    members = list(groups.values())
    if not members:
        return []
    for count in range(min(len(members), limit), 0, -1):
        picked = [members[position] for position in evenly_spaced(len(members), count)]
        if count == 1 or sum(len(group) for group in picked) <= limit:
            return sorted(index for group in picked for index in group)
    raise AssertionError("unreachable")  # pragma: no cover


def preview_selection(
    images: Sequence[ImageEntity], alignment: SampleAlignment, limit: int = PREVIEW_LIMIT
) -> list[int]:
    """The images a preview (and a build summary's preview) shows under this alignment."""
    if alignment is SampleAlignment.UNION:
        return preview_sample_indices([image.sample_id for image in images], limit)
    return preview_indices([image.channel_id for image in images], limit)


def read_build_summary(settings: Settings, profile_id: int) -> RegionBuildSummary | None:
    root = settings.region_profile_dir(profile_id)
    path = root / SUMMARY_FILENAME
    if settings.region_profiles_dir.is_symlink() or root.is_symlink() or path.is_symlink():
        return None
    try:
        return RegionBuildSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None


def has_published_build(settings: Settings, profile_id: int) -> bool:
    """Whether a profile path was published, even when its report is now corrupted."""
    root = settings.region_profile_dir(profile_id)
    return root.exists() or root.is_symlink()


def load_prepared_build(
    settings: Settings,
    profile: RegionProfileRevision,
    *,
    manifest_sha256: str,
) -> PreparedRegionBuild:
    """Verify and load the complete materialization pinned by an experiment."""
    root = settings.region_profile_dir(profile.id)
    summary = read_build_summary(settings, profile.id)
    if summary is None:
        raise ValueError(f"region profile {profile.id} has no completed build")
    if summary.profile_id != profile.id or summary.dataset_id != profile.dataset_id:
        raise ValueError(
            f"region profile {profile.id} build identity does not match its database revision"
        )
    if summary.manifest_sha256 != manifest_sha256:
        raise ValueError(
            f"region profile {profile.id} now names build {summary.manifest_sha256}, "
            f"but the experiment pins {manifest_sha256}"
        )
    if summary.config_sha256 != _config_digest(profile):
        raise ValueError(f"region profile {profile.id} no longer matches its build configuration")
    if summary.failed or summary.succeeded != summary.total:
        raise ValueError(
            f"region profile {profile.id} is incomplete: "
            f"{summary.succeeded}/{summary.total} images prepared"
        )

    manifest = root / MANIFEST_FILENAME
    if root.is_symlink() or manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"region profile {profile.id} has no safe transform manifest")
    actual_digest = sha256_file(manifest)
    if actual_digest != manifest_sha256:
        raise ValueError(
            f"region profile {profile.id} manifest changed: "
            f"expected {manifest_sha256}, found {actual_digest}"
        )

    entries: dict[int, RegionPreparationEntry] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
        for line in lines:
            entry = RegionPreparationEntry.model_validate_json(line)
            if entry.status != "succeeded" or entry.transform is None or not entry.prepared_sha256:
                raise ValueError(f"image {entry.image_id} is not a complete prepared entry")
            if entry.image_id in entries:
                raise ValueError(f"image {entry.image_id} occurs twice in the transform manifest")
            entries[entry.image_id] = entry
    except (OSError, ValueError) as exc:
        raise ValueError(f"region profile {profile.id} has an invalid transform manifest") from exc
    if len(entries) != summary.total:
        raise ValueError(
            f"region profile {profile.id} manifest has {len(entries)} entries, "
            f"expected {summary.total}"
        )
    return PreparedRegionBuild(root=root, profile=profile, summary=summary, entries=entries)


def run_region_prepare_job(ctx: JobContext) -> dict[str, Any]:
    profile_id = _integer_param(ctx, "profile_id")
    dataset_id = _integer_param(ctx, "dataset_id")
    mode = str(ctx.params.get("mode", ""))
    if mode not in {"preview", "build"}:
        raise ValueError("region preparation mode must be 'preview' or 'build'")

    with connection(ctx.settings.db_path) as conn:
        profile = profiles_repo.get_profile(conn, profile_id)
        if profile is None:
            raise ValueError(f"no region profile with id {profile_id}")
        if profile.dataset_id != dataset_id:
            raise ValueError("profile and job dataset do not match")
        images = images_repo.list_images_for_dataset(conn, dataset_id)
    selected = (
        images
        if mode == "build"
        else [images[index] for index in preview_selection(images, profile.sample_alignment)]
    )
    assets = _resolve_assets(ctx.settings, profile)
    extractor = build_extractor(
        profile.extractor_type,
        profile.extractor_config,
        assets=assets,
    )
    ctx.log(
        f"{mode.title()} profile {profile.id} ({profile.extractor_type}) on "
        f"{len(selected)} of {len(images)} images."
    )

    if mode == "preview":
        # Bounded: a union preview can exceed the budget only through one sample larger
        # than it, whose crop needs every image — so the result, not the work, is cut.
        entries = _process_images(ctx, profile, selected, extractor=extractor, output_dir=None)[
            :PREVIEW_LIMIT
        ]
        succeeded = sum(entry.status == "succeeded" for entry in entries)
        return {
            "mode": "preview",
            "profile_id": profile.id,
            "dataset_id": profile.dataset_id,
            "sampled": len(entries),
            "dataset_images": len(images),
            "succeeded": succeeded,
            "failed": len(entries) - succeeded,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }

    return _build_all(ctx, profile, selected, extractor=extractor).model_dump(mode="json")


def _build_all(
    ctx: JobContext,
    profile: RegionProfileRevision,
    images: list[ImageEntity],
    *,
    extractor: Any,
) -> RegionBuildSummary:
    if has_published_build(ctx.settings, profile.id):
        raise ValueError(
            f"region profile {profile.id} already has an immutable completed build; "
            "create a new profile revision to rebuild it"
        )
    destination = ctx.settings.region_profile_dir(profile.id)
    staging = ctx.settings.region_profiles_dir / f".profile-{profile.id}-job-{ctx.job_id}.staging"
    backup = ctx.settings.region_profiles_dir / f".profile-{profile.id}-job-{ctx.job_id}.previous"
    _require_managed_build_paths(ctx.settings, destination, staging, backup)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    (staging / "images").mkdir(parents=True)
    started = time.perf_counter()
    try:
        entries = _process_images(ctx, profile, images, extractor=extractor, output_dir=staging)
        manifest = staging / MANIFEST_FILENAME
        with manifest.open("w", encoding="utf-8") as handle:
            for entry in entries:
                stable = entry.model_dump(mode="json", exclude={"elapsed_ms"}, exclude_none=True)
                handle.write(json.dumps(stable, sort_keys=True, separators=(",", ":")) + "\n")
        manifest_digest = sha256_file(manifest)
        files = [path for path in staging.rglob("*") if path.is_file()]
        succeeded = sum(entry.status == "succeeded" for entry in entries)
        summary = RegionBuildSummary(
            profile_id=profile.id,
            dataset_id=profile.dataset_id,
            total=len(entries),
            succeeded=succeeded,
            failed=len(entries) - succeeded,
            manifest_sha256=manifest_digest,
            config_sha256=_config_digest(profile),
            storage_files=len(files) + 1,
            storage_bytes=sum(path.stat().st_size for path in files),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            preview_entries=[
                entries[index] for index in preview_selection(images, profile.sample_alignment)
            ][:PREVIEW_LIMIT],
            failure_examples=[entry for entry in entries if entry.status == "failed"][:24],
        )
        summary_path = staging / SUMMARY_FILENAME
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        final_summary = summary
        # The decimal width of `storage_bytes` can change the summary's own size. Iterate
        # to the tiny fixed point so the report is exact rather than one serialization old.
        for _ in range(4):
            actual_bytes = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
            if actual_bytes == final_summary.storage_bytes:
                break
            final_summary = final_summary.model_copy(update={"storage_bytes": actual_bytes})
            summary_path.write_text(final_summary.model_dump_json(indent=2), encoding="utf-8")

        if destination.exists() or destination.is_symlink():
            destination.replace(backup)
        staging.replace(destination)
        shutil.rmtree(backup, ignore_errors=True)
        ctx.progress(1.0, f"Prepared {succeeded}/{len(entries)} images")
        return final_summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _process_images(
    ctx: JobContext,
    profile: RegionProfileRevision,
    images: list[ImageEntity],
    *,
    extractor: Any,
    output_dir: Path | None,
) -> list[RegionPreparationEntry]:
    if profile.sample_alignment is SampleAlignment.UNION:
        return _process_samples(ctx, profile, images, extractor=extractor, output_dir=output_dir)
    return _process_each_image(ctx, profile, images, extractor=extractor, output_dir=output_dir)


def _process_each_image(
    ctx: JobContext,
    profile: RegionProfileRevision,
    images: list[ImageEntity],
    *,
    extractor: Any,
    output_dir: Path | None,
) -> list[RegionPreparationEntry]:
    entries: list[RegionPreparationEntry] = []
    total = len(images)
    for offset, image_record in enumerate(images):
        ctx.raise_if_cancelled()
        started = time.perf_counter()
        try:
            source = decode.load(Path(image_record.path))
            if source.size != (image_record.width, image_record.height):
                raise RegionExtractionError(
                    f"decoded size {source.size} differs from catalog "
                    f"{(image_record.width, image_record.height)}"
                )
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            extraction = extractor.extract(rgb)
            transform = SpatialTransform.resolve(
                source_size=source.size,
                prepared_size=(profile.prepared_width, profile.prepared_height),
                region=extraction.bounds,
                padding_fraction=profile.padding_fraction,
            )
            prepared_digest = None
            if output_dir is not None:
                prepared = transform.prepare_image(
                    source, resample=_RESAMPLE_FILTERS[profile.resample]
                )
                image_path = output_dir / "images" / f"{image_record.id}.png"
                prepared.save(image_path, format="PNG", optimize=False)
                prepared_digest = sha256_file(image_path)
            entry = RegionPreparationEntry(
                image_id=image_record.id,
                source_sha256=image_record.sha256,
                status="succeeded",
                transform=transform,
                prepared_sha256=prepared_digest,
                extractor_confidence=extraction.confidence,
                extractor_metadata=extraction.metadata,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            entry = RegionPreparationEntry(
                image_id=image_record.id,
                source_sha256=image_record.sha256,
                status="failed",
                error=str(exc) or type(exc).__name__,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
            ctx.log(f"Image {image_record.id} failed: {entry.error}", level="warning")
        entries.append(entry)
        ctx.progress(
            (offset + 1) / max(total, 1),
            f"{profile.extractor_type}: {offset + 1}/{total}",
        )
    return entries


@dataclass
class _Located:
    """One decoded image and its own extraction, held only while its sample is processed."""

    record: ImageEntity
    started: float
    source: Image.Image | None = None
    extraction: Any = None
    transform: SpatialTransform | None = None
    error: str | None = None


def _process_samples(
    ctx: JobContext,
    profile: RegionProfileRevision,
    images: list[ImageEntity],
    *,
    extractor: Any,
    output_dir: Path | None,
) -> list[RegionPreparationEntry]:
    """Prepare one sample at a time, giving all of its images the union of their crops.

    Memory is one sample's decoded images, never the dataset's. Entries come back in the
    order `images` was given, so the manifest is ordered exactly as a per-image build's.
    """
    groups: dict[int, list[int]] = {}
    for index, record in enumerate(images):
        groups.setdefault(record.sample_id, []).append(index)
    by_index: dict[int, RegionPreparationEntry] = {}
    done = 0
    total = len(images)
    for sample_id, indices in groups.items():
        located: list[_Located] = []
        for index in indices:
            ctx.raise_if_cancelled()
            located.append(_locate(profile, images[index], extractor))
        for index, entry in zip(
            indices, _unite_sample(profile, sample_id, located, output_dir), strict=True
        ):
            by_index[index] = entry
            if entry.status == "failed":
                ctx.log(f"Image {entry.image_id} failed: {entry.error}", level="warning")
        done += len(indices)
        ctx.progress(done / max(total, 1), f"{profile.extractor_type}: {done}/{total}")
    return [by_index[index] for index in range(total)]


def _locate(profile: RegionProfileRevision, record: ImageEntity, extractor: Any) -> _Located:
    located = _Located(record=record, started=time.perf_counter())
    try:
        source = decode.load(Path(record.path))
        if source.size != (record.width, record.height):
            raise RegionExtractionError(
                f"decoded size {source.size} differs from catalog {(record.width, record.height)}"
            )
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        located.extraction = extractor.extract(rgb)
        located.transform = SpatialTransform.resolve(
            source_size=source.size,
            prepared_size=(profile.prepared_width, profile.prepared_height),
            region=located.extraction.bounds,
            padding_fraction=profile.padding_fraction,
        )
        located.source = source
    except Exception as exc:
        located.error = str(exc) or type(exc).__name__
    return located


def _unite_sample(
    profile: RegionProfileRevision,
    sample_id: int,
    located: list[_Located],
    output_dir: Path | None,
) -> list[RegionPreparationEntry]:
    """Replace each image's crop with the union of its sample's crops, or fail the sample.

    A sample is united only when every image was located and all share one source size:
    cropping the surviving images on their own would be exactly the misregistration this
    mode exists to prevent, and a box has no meaning across two frames of different sizes.
    """
    failed = [item for item in located if item.error is not None]
    if failed:
        named = ", ".join(f"image {item.record.id}" for item in failed)
        return [
            _failed_entry(
                item,
                item.error
                if item.error is not None
                else f"sample {sample_id} was not united because {named} failed",
            )
            for item in located
        ]
    sizes = {(item.record.width, item.record.height) for item in located}
    if len(sizes) > 1:
        described = ", ".join(
            f"image {item.record.id} is {item.record.width}x{item.record.height}"
            for item in located
        )
        reason = (
            f"sample {sample_id} cannot share one crop: its images differ in size ({described})"
        )
        return [_failed_entry(item, reason) for item in located]

    transforms = [item.transform for item in located if item.transform is not None]
    union = PixelBounds(
        left=min(transform.crop_left for transform in transforms),
        top=min(transform.crop_top for transform in transforms),
        right=max(transform.crop_right for transform in transforms),
        bottom=max(transform.crop_bottom for transform in transforms),
    )
    shared = SpatialTransform.resolve(
        source_size=sizes.pop(),
        prepared_size=(profile.prepared_width, profile.prepared_height),
        # The members are already padded and clipped; the union is taken as it is.
        region=union,
        padding_fraction=0.0,
    )
    entries: list[RegionPreparationEntry] = []
    for item in located:
        try:
            prepared_digest = None
            if output_dir is not None and item.source is not None:
                prepared = shared.prepare_image(
                    item.source, resample=_RESAMPLE_FILTERS[profile.resample]
                )
                image_path = output_dir / "images" / f"{item.record.id}.png"
                prepared.save(image_path, format="PNG", optimize=False)
                prepared_digest = sha256_file(image_path)
        except Exception as exc:
            # Same rule as a failed extraction: the sample fails as a whole.
            error = str(exc) or type(exc).__name__
            sibling = f"sample {sample_id} was not united because image {item.record.id} failed"
            return [_failed_entry(other, error if other is item else sibling) for other in located]
        entries.append(
            RegionPreparationEntry(
                image_id=item.record.id,
                source_sha256=item.record.sha256,
                status="succeeded",
                transform=shared,
                prepared_sha256=prepared_digest,
                extractor_confidence=item.extraction.confidence,
                extractor_metadata=item.extraction.metadata,
                elapsed_ms=(time.perf_counter() - item.started) * 1000.0,
            )
        )
    return entries


def _failed_entry(item: _Located, error: str) -> RegionPreparationEntry:
    return RegionPreparationEntry(
        image_id=item.record.id,
        source_sha256=item.record.sha256,
        status="failed",
        error=error,
        elapsed_ms=(time.perf_counter() - item.started) * 1000.0,
    )


def _resolve_assets(settings: Settings, profile: RegionProfileRevision) -> dict[str, Path]:
    extractor_type = get_extractor_class(profile.extractor_type)
    assets: dict[str, Path] = {}
    for key in extractor_type.required_assets:
        spec = get_spec(key)
        if spec is None:
            raise RegionExtractionError(f"extractor requires unknown model asset {key!r}")
        resolved = resolve_asset(settings, spec)
        if not resolved.ready:
            raise RegionExtractionError(f"required model asset {key!r} is {resolved.reason}")
        assets[key] = resolved.path
    return assets


def _config_digest(profile: RegionProfileRevision) -> str:
    # A field added after builds were published enters the digest only at a non-default
    # value, so every revision authored before it keeps the digest its build recorded.
    exclude = {"id", "created_at"}
    if profile.sample_alignment is SampleAlignment.PER_IMAGE:
        exclude.add("sample_alignment")
    payload = profile.model_dump(mode="json", exclude=exclude)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _integer_param(ctx: JobContext, key: str) -> int:
    try:
        return int(ctx.params[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"region preparation job has malformed {key}") from exc


def _require_managed_build_paths(settings: Settings, *paths: Path) -> None:
    root = settings.region_profiles_dir
    if root.is_symlink():
        raise ValueError("region profile storage root may not be a symlink")
    for path in paths:
        if path.parent != root or path.is_symlink():
            raise ValueError(f"unsafe region profile build path: {path}")
