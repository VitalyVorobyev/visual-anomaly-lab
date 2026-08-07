"""The `train` job handler.

One entry in `jobs/handlers.py` and this function. The queue, the JSON-lines protocol,
cancellation, log tee-ing and WebSocket fan-out were built in M2 and are kind-agnostic;
nothing in any of them knows that training exists (ADR-0009).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.entities import ExperimentStatus, Label, Subset
from anomaly_lab.experiments.context import (
    ExperimentJobError,
    diagnostics_writer,
    load_experiment,
    to_records,
)
from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.models.base import ModelCancelledError, TrainContext
from anomaly_lab.schemas import API_MODEL_CONFIG

MODEL_SUBDIR = "model"


class TrainParams(BaseModel):
    """What a train job is given. `experiment_id` carries everything else."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    diagnostics: bool = Field(
        default=True,
        description="Record what the model shows about itself. Costs disk, not accuracy.",
    )


def run_train_job(ctx: JobContext) -> dict[str, Any]:
    """Fit a method on its split's training normals and persist the fitted model."""
    params = TrainParams.model_validate(dict(ctx.params))

    with connection(ctx.settings.db_path) as conn:
        loaded = load_experiment(conn, ctx.settings, params.experiment_id)
        experiment = loaded.experiment

        # Normals only, and by *sample* label rather than by image, because the label
        # lives on the sample (ADR-0005). Anomaly detection learns what normal looks
        # like; a defect in the training set teaches the model that defects are normal.
        train_images = images_repo.list_images_for_split(
            conn,
            experiment.split_id,
            subsets=[Subset.TRAIN],
            labels=[Label.NORMAL],
        )
        # Held-out normals, where the split has any. EfficientAD fits its score
        # normalization on these; VisA's official one-class protocol has no `val` subset
        # at all, so this is routinely empty and must not be treated as an error.
        val_images = images_repo.list_images_for_split(
            conn,
            experiment.split_id,
            subsets=[Subset.VAL],
            labels=[Label.NORMAL],
        )
        all_train = images_repo.list_images_for_split(
            conn, experiment.split_id, subsets=[Subset.TRAIN]
        )
        experiments_repo.set_status(conn, experiment.id, ExperimentStatus.TRAINING)

    if not train_images:
        msg = (
            f"split {experiment.split_id} has no normal-labelled samples in its train "
            "subset, so there is nothing to learn from. Label some samples normal, or "
            "create a split whose train subset holds them."
        )
        raise ExperimentJobError(msg)

    excluded = len(all_train) - len(train_images)
    if excluded:
        ctx.log(
            f"{excluded} image(s) in the train subset are not labelled normal and were "
            "excluded from fitting",
            level="warning",
        )

    ctx.log(f"method {experiment.model_type} on device {loaded.device.device.value}")
    ctx.log(f"device: {loaded.device.reason}")
    ctx.log(
        f"{len(train_images)} training image(s), {len(val_images)} held-out normal(s), "
        f"input {loaded.preprocessing.width}x{loaded.preprocessing.height} "
        f"{loaded.preprocessing.color.value}"
    )
    if not val_images:
        ctx.log(
            "this split has no val subset; a method that calibrates on held-out normals "
            "will say what it does instead",
            level="warning",
        )

    writer = diagnostics_writer(loaded, enabled=params.diagnostics)
    train_ctx = TrainContext(
        artifact_dir=loaded.artifact_dir,
        cache_dir=loaded.cache_dir,
        preprocessing=loaded.preprocessing,
        device=loaded.device.device,
        reporter=ctx,
        diagnostics=writer,
        val=to_records(val_images),
    )

    try:
        loaded.model.fit(to_records(train_images), train_ctx)
        model_dir = loaded.artifact_dir / MODEL_SUBDIR
        model_dir.mkdir(parents=True, exist_ok=True)
        loaded.model.save(model_dir)
    except ModelCancelledError as exc:
        # The plugin's cancellation and the job system's are the same event wearing two
        # names; the boundary translates rather than leaking either one across.
        with connection(ctx.settings.db_path) as conn:
            experiments_repo.set_status(conn, experiment.id, ExperimentStatus.DRAFT)
        raise JobCancelledError from exc
    except Exception:
        with connection(ctx.settings.db_path) as conn:
            experiments_repo.set_status(conn, experiment.id, ExperimentStatus.FAILED)
        raise

    index = writer.flush()
    with connection(ctx.settings.db_path) as conn:
        experiments_repo.set_status(conn, experiment.id, ExperimentStatus.TRAINED)

    ctx.progress(1.0, "trained")
    return {
        "experiment_id": experiment.id,
        "model_type": experiment.model_type,
        "device": loaded.device.device.value,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "excluded_images": excluded,
        "artifact_dir": str(loaded.artifact_dir),
        "diagnostics": len(index.entries),
    }
