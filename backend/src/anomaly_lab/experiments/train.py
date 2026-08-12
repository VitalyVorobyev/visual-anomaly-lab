"""The `train` job handler.

One entry in `jobs/handlers.py` and this function. The queue, the JSON-lines protocol,
cancellation, log tee-ing and WebSocket fan-out were built in M2 and are kind-agnostic;
nothing in any of them knows that training exists (ADR-0009).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
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
from anomaly_lab.models.base import ModelCancelledError, SupportsResume, TrainContext
from anomaly_lab.schemas import API_MODEL_CONFIG

MODEL_SUBDIR = "model"
TRAIN_STATE_FILENAME = "train_state.json"
TRAIN_STATE_FORMAT = 1


class TrainParams(BaseModel):
    """What a train job is given. `experiment_id` carries everything else."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    diagnostics: bool = Field(
        default=True,
        description="Record what the model shows about itself. Costs disk, not accuracy.",
    )
    additional_steps: int | None = Field(
        # `default_factory`, not `= None`. A literal default emits `"default": null` into
        # the schema, and `openapi-typescript` then makes the property **required** — the
        # trap that silently pinned a value for two milestones (see the M4.6 summary).
        default_factory=lambda: None,
        ge=1,
        le=200_000,
        description=(
            "Continue the existing model for this many further steps instead of training "
            "from scratch. Only for a method that declares `supports_resume`."
        ),
    )


class TrainingState(BaseModel):
    """How much training an experiment's stored model has actually had.

    Written beside the checkpoint as JSON, and read by the API, because a `.pt` cannot be
    opened in a process that has no torch — which the API process deliberately does not.
    Without this the configuration panel shows `max_steps: 4000` beside an 8000-step
    model, which is a lie in the one record that is supposed to make a run reproducible.
    """

    model_config = API_MODEL_CONFIG

    format: int = TRAIN_STATE_FORMAT
    completed_steps: int
    runs: int = 1
    last_run_steps: int = 0
    model_type: str = ""
    written_at: str = ""
    resumable: bool = True
    """False for a checkpoint written before optimizer state was saved."""


def read_training_state(model_dir: Path) -> TrainingState | None:
    """The sidecar, or `None` when nothing has trained or it predates this."""
    path = model_dir / TRAIN_STATE_FILENAME
    if not path.is_file():
        return None
    try:
        return TrainingState.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


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
        val=to_records(val_images, loaded.region_build),
    )

    model_dir = loaded.artifact_dir / MODEL_SUBDIR
    previous = read_training_state(model_dir)
    resuming = params.additional_steps is not None

    if resuming:
        _check_resumable(loaded.model, experiment.model_type, model_dir)
        ctx.log(
            f"continuing the stored model for {params.additional_steps} more steps "
            f"(it has {previous.completed_steps if previous else 0} so far)"
        )

    try:
        records = to_records(train_images, loaded.region_build)
        if resuming:
            model = loaded.model
            model.load(model_dir)
            # `_check_resumable` already proved this; narrowing again keeps mypy honest
            # without an assert, which ruff correctly refuses in shipped code.
            if not isinstance(model, SupportsResume):  # pragma: no cover - checked above
                raise ExperimentJobError(f"{experiment.model_type} cannot resume")
            model.fit_more(records, train_ctx, additional_steps=params.additional_steps or 0)
        else:
            loaded.model.fit(records, train_ctx)
        model_dir.mkdir(parents=True, exist_ok=True)
        loaded.model.save(model_dir)
        _write_training_state(
            model_dir,
            model=loaded.model,
            model_type=experiment.model_type,
            previous=previous if resuming else None,
        )
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

    state = read_training_state(model_dir)
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
        "completed_steps": None if state is None else state.completed_steps,
        "resumed": resuming,
    }


def _check_resumable(model: Any, model_type: str, model_dir: Path) -> None:
    """Refuse a continuation the method cannot honour, before anything is spawned.

    The flag and the protocol are checked against each other rather than the flag being
    trusted: a method that declares `supports_resume` without satisfying `SupportsResume`
    is a plugin bug, and saying so beats a `AttributeError` deep inside the handler.
    """
    if not type(model).capabilities().supports_resume:
        msg = (
            f"method {model_type} cannot continue a finished run — it declares no "
            "`supports_resume`. Train it from scratch instead."
        )
        raise ExperimentJobError(msg)
    if not isinstance(model, SupportsResume):
        msg = (
            f"method {model_type} declares `supports_resume` but does not implement "
            "`completed_steps` and `fit_more`. That is a bug in the plugin."
        )
        raise ExperimentJobError(msg)
    if not (model_dir / "").exists() or not any(model_dir.iterdir()):
        msg = (
            f"experiment has no stored model to continue from in {model_dir}. Train it once first."
        )
        raise ExperimentJobError(msg)


def _write_training_state(
    model_dir: Path,
    *,
    model: Any,
    model_type: str,
    previous: TrainingState | None,
) -> None:
    """Record how much training the stored checkpoint has had, in plain JSON.

    Written by the *handler* rather than by the plugin, from `completed_steps()`, so the
    API can report progress without importing torch. A method that does not support
    resume writes nothing: there is no step count to report, and an invented one would be
    worse than the absence.
    """
    if not isinstance(model, SupportsResume):
        return
    completed = model.completed_steps()
    state = TrainingState(
        completed_steps=completed,
        runs=1 if previous is None else previous.runs + 1,
        last_run_steps=completed - (previous.completed_steps if previous else 0),
        model_type=model_type,
        written_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (model_dir / TRAIN_STATE_FILENAME).write_text(state.model_dump_json(indent=2), encoding="utf-8")
