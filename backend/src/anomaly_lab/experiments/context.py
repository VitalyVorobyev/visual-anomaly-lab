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
from dataclasses import dataclass
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories.images import SplitImage
from anomaly_lab.domain.entities import Experiment
from anomaly_lab.models.base import AnomalyModel, ImageRecord
from anomaly_lab.models.device import ResolvedDevice, resolve_device
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class


class ExperimentJobError(Exception):
    """The job cannot proceed, for a reason the operator can act on."""


@dataclass(frozen=True)
class LoadedExperiment:
    experiment: Experiment
    model: AnomalyModel
    model_class: type[AnomalyModel]
    preprocessing: PreprocessingConfig
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

    return LoadedExperiment(
        experiment=experiment,
        model=model_class.build(experiment.model_config_),
        model_class=model_class,
        preprocessing=PreprocessingConfig.model_validate(experiment.preprocessing_config),
        device=resolve_device(model_class.capabilities().preferred_device),
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
    )


def diagnostics_writer(
    loaded: LoadedExperiment,
    *,
    enabled: bool,
    image_budget: int | None = None,
) -> DiagnosticWriter:
    """A writer that is enabled only when the model actually produces diagnostics."""
    return DiagnosticWriter(
        loaded.artifact_dir / "diagnostics",
        enabled=enabled and loaded.model_class.capabilities().produces_diagnostics,
        image_budget=image_budget,
    )


def to_records(images: list[SplitImage]) -> list[ImageRecord]:
    """Narrow catalog rows to what a model is allowed to see.

    Notably absent: the label and the mask. A model is not told which images are
    defective, so a plugin cannot quietly use the answer it is being asked to find.
    """
    return [
        ImageRecord(
            image_id=image.image_id,
            sample_id=image.sample_id,
            channel=image.channel,
            path=Path(image.path),
        )
        for image in images
    ]
