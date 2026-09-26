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
from anomaly_lab.domain.entities import ExperimentStatus, JobKind, Task
from anomaly_lab.experiments.context import (
    ExperimentJobError,
    diagnostics_writer,
    load_experiment,
    to_records,
)
from anomaly_lab.experiments.policy import NoTrainingPolicyError, training_set
from anomaly_lab.experiments.targets import (
    PreparedBoxTargets,
    PreparedClassTargets,
    PreparedLabelTargets,
)
from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.jobs.protocol import FOLLOW_UP_KEY
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
    then_score: bool = Field(
        # A factory for the reason `additional_steps` gives: optional in the generated client.
        default_factory=lambda: False,
        description=(
            "Once training succeeds, queue scoring and evaluation of the default subsets. "
            "Nothing is queued after a failed or cancelled run."
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
    """Fit a method on what its task trains on (`experiments/policy.py`), and persist it."""
    params = TrainParams.model_validate(dict(ctx.params))

    # What the run trains on is settled before its region build is resolved, so a split
    # with nothing to fit is refused before a dataset's worth of pixels is prepared.
    with connection(ctx.settings.db_path) as conn:
        stored = experiments_repo.get_experiment(conn, params.experiment_id)
        if stored is None:
            raise ExperimentJobError(f"no experiment with id {params.experiment_id}")
        try:
            chosen = training_set(conn, stored)
        except NoTrainingPolicyError as exc:
            raise ExperimentJobError(str(exc)) from exc

    train_images, val_images, excluded = chosen.train, chosen.val, chosen.excluded
    if not train_images:
        raise ExperimentJobError(chosen.empty_because)

    with connection(ctx.settings.db_path) as conn:
        loaded = load_experiment(conn, ctx.settings, params.experiment_id, job=ctx)
        experiment = loaded.experiment
        experiments_repo.set_status(conn, experiment.id, ExperimentStatus.TRAINING)
    if excluded:
        ctx.log(
            f"{excluded} image(s) in the train subset {chosen.excluded_because} and were "
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
    if experiment.target_label is not None:
        ctx.log(f"fitting on references of {experiment.target_label!r}")
    elif experiment.classes:
        ctx.log(f"fitting on annotated images of {', '.join(experiment.classes)}")
    elif not val_images:
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
        targets=None
        if experiment.target_label is None
        else PreparedClassTargets(experiment.target_label, chosen.truths, loaded.region_build),
        label_targets=PreparedLabelTargets(
            tuple(experiment.classes), chosen.label_truths, loaded.region_build
        )
        if experiment.task is Task.SEMANTIC_SEGMENTATION
        else None,
        box_targets=PreparedBoxTargets(
            tuple(experiment.classes), chosen.box_truths, loaded.region_build
        )
        if experiment.task is Task.OBJECT_DETECTION
        else None,
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
    result: dict[str, Any] = {
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
    if params.then_score:
        # Named here, queued by the queue: only it knows the run succeeded rather than being
        # cancelled after this line, and it enqueues whatever a job names without knowing
        # what an `infer` is.
        result[FOLLOW_UP_KEY] = {
            "kind": JobKind.INFER.value,
            "params": {"experiment_id": experiment.id},
        }
    return result


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
