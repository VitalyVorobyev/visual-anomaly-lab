"""Bounded preview and atomic full preparation for one region profile revision."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
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
from anomaly_lab.domain.entities import RegionProfileRevision, SpatialResample
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.media import decode
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset, sha256_file
from anomaly_lab.models.base import evenly_spaced
from anomaly_lab.regions.base import RegionExtractionError
from anomaly_lab.regions.registry import build as build_extractor
from anomaly_lab.regions.registry import get_extractor_class
from anomaly_lab.regions.transform import SpatialTransform
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


def preview_indices(total: int, limit: int = PREVIEW_LIMIT) -> list[int]:
    """Spread a preview over the whole image list, never its first folder."""
    return evenly_spaced(total, limit)


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
        images if mode == "build" else [images[index] for index in preview_indices(len(images))]
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
        entries = _process_images(ctx, profile, selected, extractor=extractor, output_dir=None)
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
            preview_entries=[entries[index] for index in preview_indices(len(entries))],
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
    payload = profile.model_dump(
        mode="json",
        exclude={"id", "created_at"},
    )
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
