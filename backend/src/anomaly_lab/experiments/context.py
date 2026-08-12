"""Assembling a model's context from a job's context.

The two handlers need the same five things — the experiment, its plugin, a resolved
device, an artifact directory and a diagnostics writer — so the assembly lives here once
rather than being copied into both.

This module is also where the injection boundary of ADR-0014 is actually enforced: a
`JobContext` goes in, a `ModelContext` comes out, and the plugin on the far side has no
route back to settings, to SQLite, or to the queue.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories.images import SplitImage
from anomaly_lab.domain.entities import Experiment
from anomaly_lab.models.base import AnomalyModel, ImageRecord, evenly_spaced
from anomaly_lab.models.device import ResolvedDevice, resolve_device
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.regions.preparation import PreparedRegionBuild, load_prepared_build


class ExperimentJobError(Exception):
    """The job cannot proceed, for a reason the operator can act on."""


@dataclass(frozen=True)
class LoadedExperiment:
    experiment: Experiment
    model: AnomalyModel
    model_class: type[AnomalyModel]
    preprocessing: PreprocessingConfig
    region_build: PreparedRegionBuild
    device: ResolvedDevice
    artifact_dir: Path
    cache_dir: Path


def load_experiment(
    conn: sqlite3.Connection,
    settings: Settings,
    experiment_id: int,
) -> LoadedExperiment:
    """Read an experiment and build its plugin, or fail with a readable reason."""
    experiment = experiments_repo.get_experiment(conn, experiment_id)
    if experiment is None:
        msg = f"no experiment with id {experiment_id}"
        raise ExperimentJobError(msg)

    model_class = get_model_class(experiment.model_type)
    availability = model_class.availability()
    if not availability.available:
        raise ExperimentJobError(availability.reason or f"{experiment.model_type} is unavailable")

    artifact_dir = settings.experiment_dir(experiment.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = settings.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    profile = region_profiles_repo.get_profile(conn, experiment.region_profile_id)
    if profile is None:
        raise ExperimentJobError(
            f"experiment {experiment.id} references missing region profile "
            f"{experiment.region_profile_id}"
        )
    try:
        region_build = load_prepared_build(
            settings,
            profile,
            manifest_sha256=experiment.region_manifest_sha256,
        )
    except ValueError as exc:
        raise ExperimentJobError(str(exc)) from exc

    return LoadedExperiment(
        experiment=experiment,
        model=model_class.build(experiment.model_config_),
        model_class=model_class,
        preprocessing=PreprocessingConfig.model_validate(experiment.preprocessing_config),
        region_build=region_build,
        device=resolve_device(model_class.capabilities().preferred_device),
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
    )


def diagnostics_writer(
    loaded: LoadedExperiment,
    *,
    enabled: bool,
    image_budget: int | None = None,
    over_images: Sequence[int] | None = None,
) -> DiagnosticWriter:
    """A writer that is enabled only when the model actually produces diagnostics.

    `over_images` is the full set the run is about to touch, in order. When given, the
    budget is spent on an **evenly spaced** selection of it rather than on whichever
    images the model reaches first — otherwise the kept set is N neighbours from one
    corner of the dataset presented as a sample of the run.
    """
    keep = None
    if over_images is not None and image_budget is not None:
        keep = [over_images[index] for index in evenly_spaced(len(over_images), image_budget)]
    return DiagnosticWriter(
        loaded.artifact_dir / "diagnostics",
        enabled=enabled and loaded.model_class.capabilities().produces_diagnostics,
        image_budget=image_budget,
        keep_images=keep,
    )


def to_records(images: list[SplitImage], region_build: PreparedRegionBuild) -> list[ImageRecord]:
    """Narrow catalog rows to what a model is allowed to see.

    Notably absent: the label and the mask. A model is not told which images are
    defective, so a plugin cannot quietly use the answer it is being asked to find.
    """
    records: list[ImageRecord] = []
    for image in images:
        entry = region_build.entries.get(image.image_id)
        if entry is None:
            raise ExperimentJobError(
                f"region build {region_build.summary.manifest_sha256} has no image {image.image_id}"
            )
        if entry.source_sha256 != image.sha256:
            raise ExperimentJobError(
                f"source image {image.image_id} changed after region preparation; "
                "create and build a new profile revision"
            )
        try:
            path = region_build.image_path(image.image_id)
        except ValueError as exc:
            raise ExperimentJobError(str(exc)) from exc
        records.append(
            ImageRecord(
                image_id=image.image_id,
                sample_id=image.sample_id,
                channel=image.channel,
                path=path,
            )
        )
    return records
