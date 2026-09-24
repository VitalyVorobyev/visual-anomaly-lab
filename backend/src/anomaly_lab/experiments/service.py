"""Experiment orchestration: the decisions between the routes and the repositories.

What lives here is everything that *decides* — whether a request can succeed, in what
order rows and directories are written and removed, which process may touch an
experiment's artifacts while it does. The routes under `api/routers/experiments/` read
parameters, call one function here and shape the answer; the read models they answer with
are built in `api/routers/experiments/views.py`.

Refusals are `anomaly_lab.errors` categories. Nothing here imports FastAPI.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_lab.annotations.class_truth import MAX_SEGMENTATION_CLASSES
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection, transaction
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.deployment.export import ExportParams
from anomaly_lab.domain.entities import (
    TARGETED_TASKS,
    ClassPresence,
    Experiment,
    ExperimentStatus,
    Job,
    Split,
    Subset,
    Task,
)
from anomaly_lab.errors import (
    ConflictError,
    GoneError,
    InvalidInputError,
    NotFoundError,
    UnavailableError,
    UnsupportedRequestError,
)
from anomaly_lab.eval.evaluators import evaluator_for, has_evaluator
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.experiments.train import MODEL_SUBDIR, read_training_state
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentError, ResidentWorker
from anomaly_lab.media.overlay import render_anomaly_map, render_rgb_image
from anomaly_lab.models.base import Capabilities
from anomaly_lab.models.diagnostics import (
    DiagnosticEntry,
    DiagnosticKind,
    DisplayRange,
    PruneResult,
    PruneScope,
    load_index,
    prune,
)
from anomaly_lab.models.preprocessing import (
    PreprocessingConfig,
    PreprocessingOptions,
    load_array,
)
from anomaly_lab.models.registry import UnknownModelError, get_model_class
from anomaly_lab.owned_storage import StorageUsage, experiment_artifact_path, path_usage
from anomaly_lab.regions.preparation import (
    PreparedRegionBuild,
    load_prepared_build,
    read_build_summary,
)

# --- Lookups and preconditions -------------------------------------------------------------


def load_experiment(settings: Settings, experiment_id: int) -> Experiment:
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
    if experiment is None:
        raise NotFoundError(f"no experiment with id {experiment_id}")
    return experiment


def registered_capabilities(model_type: str) -> Capabilities:
    """The method's capabilities, or a refusal naming it when it is no longer registered."""
    try:
        return get_model_class(model_type).capabilities()
    except UnknownModelError as exc:
        raise InvalidInputError(f"method {model_type} is not registered") from exc


def refuse_while_a_job_runs(settings: Settings, queue: JobQueue) -> None:
    """409 with the job's name, for the operations that must not race a worker.

    Any running job, not only this experiment's: one machine, one device, one FIFO queue
    (ADR-0009), and an inference job's `flush()` merges the index with whatever is on
    disk — so a delete landing between the model returning and the flush would be quietly
    undone. Refusing while naming what to wait for is the whole difference between "not
    now" and "that button does nothing sometimes".

    Both halves are needed and neither is redundant. The queue's claim covers the window
    between the pre-spawn hook finishing and the row being written; the row covers a
    worker this process did not launch — and is the durable answer, reconciled at startup.
    """
    with connection(settings.db_path) as conn:
        job = jobs_repo.running_job(conn)
        if job is None and queue.current_job_id is not None:
            job = jobs_repo.get_job(conn, queue.current_job_id)
    if job is None:
        return
    raise ConflictError(f"a {job.kind.value} job (id {job.id}) is running; try again when it ends")


def refuse_impossible_resume(experiment: Experiment) -> None:
    """Refuse a continuation that cannot work, naming which of the three reasons it is.

    Refused before enqueueing rather than ten minutes later inside a worker: a form that
    cannot succeed should fail as a form.
    """
    capabilities = registered_capabilities(experiment.model_type)

    if not capabilities.supports_resume:
        raise InvalidInputError(
            f"method {experiment.model_type} cannot continue a finished run; it has "
            "no notion of a training step. Train it from scratch instead."
        )

    state = read_training_state(Path(experiment.artifact_dir) / MODEL_SUBDIR)
    if state is None:
        raise InvalidInputError(
            "this experiment has not been trained yet, so there is nothing to continue"
        )
    if not state.resumable:
        raise InvalidInputError(
            "this checkpoint was written before optimizer state was saved, so it "
            "cannot be continued exactly. Train from scratch once, and that run can "
            "then be continued."
        )


def refuse_impossible_export(experiment: Experiment, params: ExportParams) -> None:
    capabilities = registered_capabilities(experiment.model_type)
    if params.format not in capabilities.portable_formats:
        raise InvalidInputError(
            f"method {experiment.model_type} does not support {params.format.value} export"
        )
    if experiment.status is not ExperimentStatus.TRAINED:
        raise InvalidInputError("train the experiment before exporting it")


def refuse_impossible_diagnose(
    settings: Settings, queue: JobQueue, experiment: Experiment, image_id: int
) -> None:
    """Fail a request that cannot succeed as a *request*, not as a 503 ten seconds later.

    The same reasoning `create_experiment` applies to a config that fails its method's
    schema. Every check here is cheap and needs no torch, which is the point: the process
    that would discover them holds an accelerator and costs a model load to start.
    """
    refuse_while_a_job_runs(settings, queue)

    capabilities = registered_capabilities(experiment.model_type)
    if not capabilities.produces_diagnostics:
        raise InvalidInputError(
            f"method {experiment.model_type} records no diagnostics about an image"
        )
    if capabilities.requires_training and experiment.status is not ExperimentStatus.TRAINED:
        raise InvalidInputError(
            f"experiment {experiment.id} is {experiment.status.value}; train it before "
            "asking what it saw in an image"
        )

    with connection(settings.db_path) as conn:
        in_split = images_repo.list_images_for_split(
            conn,
            experiment.split_id,
            subsets=list(Subset),
            channels=experiment.channels,
        )
    if not any(image.image_id == image_id for image in in_split):
        # Not merely "no such image": an image of another dataset exists and is still the
        # wrong thing to ask this experiment about — and so is a dark-field image of the
        # right part, when this run only ever read bright-field.
        detail = f"image {image_id} is not in this experiment's split"
        if experiment.channels:
            detail += f" and channel selection ({', '.join(experiment.channels)})"
        raise NotFoundError(detail)


# --- Creation ------------------------------------------------------------------------------


def resolve_channels(
    conn: sqlite3.Connection, dataset_id: int, requested: Sequence[str]
) -> list[str]:
    """Validate a requested channel selection and freeze it in the dataset's own order.

    Stored in `Channel.position` order rather than the order the client happened to send,
    so two identical requests produce identical frozen records and a diff between two
    experiments never shows a difference that is only a permutation.

    An unknown name is a 422 naming what *is* available. Accepting it silently would
    produce a run that read no images and reported nothing wrong, which looks exactly like
    a split with no data in it.
    """
    if not requested:
        return []
    available = [channel.name for channel in datasets_repo.list_channels(conn, dataset_id)]
    if not available:
        raise InvalidInputError(
            f"dataset {dataset_id} has no channels, so there is nothing to select; "
            "leave the selection empty"
        )
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise InvalidInputError(
            f"dataset {dataset_id} has no channel named {', '.join(unknown)}; "
            f"it has {', '.join(available)}"
        )
    chosen = set(requested)
    return [name for name in available if name in chosen]


def validate_target(
    conn: sqlite3.Connection, task: Task, target_label: str | None, split: Split
) -> None:
    """Refuse a target class that the task does not take, or that the split cannot serve.

    A targeted task's references are the split's `train` subset (ADR-0040), so every one of
    them has to show the class: a reference without a region of it has nothing to teach.
    A `few_shot` split was drawn for one class and serves no other.
    """
    if task not in TARGETED_TASKS:
        if target_label is not None:
            raise InvalidInputError(f"the task {task.value!r} does not take a target class")
        return
    if target_label is None:
        raise InvalidInputError(f"the task {task.value!r} needs a target class")
    known = {label.key for label in annotations_repo.list_labels(conn, split.dataset_id)}
    if target_label not in known:
        raise InvalidInputError(
            f"dataset {split.dataset_id} has no annotation class {target_label!r}"
        )
    drawn_for = split.params.get("label_key")
    if split.strategy == "few_shot" and drawn_for != target_label:
        raise InvalidInputError(
            f"split {split.id} draws references of {drawn_for!r}, not {target_label!r}"
        )
    references = splits_repo.list_sample_ids(conn, split.id, Subset.TRAIN)
    if not references:
        raise InvalidInputError(f"split {split.id} has no references in its train subset")
    presence = annotations_repo.class_presence(conn, split.dataset_id, target_label)
    blank = sorted(
        sample_id
        for sample_id in references
        if presence.get(sample_id) is not ClassPresence.PRESENT
    )
    if blank:
        raise InvalidInputError(
            f"references {blank} of split {split.id} show no completed region of {target_label!r}"
        )


def pin_classes(conn: sqlite3.Connection, task: Task, dataset_id: int) -> list[str]:
    """The class list a supervised segmentation run is frozen with; empty for any other task.

    Every class of the dataset, in taxonomy order, at creation (ADR-0039). Class `i` of the
    list is label index `i + 1` for the life of the run, so a class added or reordered later
    cannot renumber a stored model's output. The class-index PNGs are 8-bit, and 255 marks a
    pixel no pinned class answers for, which bounds the list at 254.
    """
    if task is not Task.SEMANTIC_SEGMENTATION:
        return []
    classes = [label.key for label in annotations_repo.list_labels(conn, dataset_id)]
    if not classes:
        raise InvalidInputError(f"dataset {dataset_id} has no annotation class to segment")
    if len(classes) > MAX_SEGMENTATION_CLASSES:
        raise InvalidInputError(
            f"dataset {dataset_id} has {len(classes)} annotation classes; a segmentation run "
            f"can segment at most {MAX_SEGMENTATION_CLASSES}"
        )
    return classes


def create_experiment(
    settings: Settings,
    *,
    name: str,
    dataset_id: int,
    split_id: int,
    region_profile_id: int,
    model_type: str,
    config: dict[str, Any],
    preprocessing: dict[str, Any],
    task: Task = Task.ANOMALY,
    target_label: str | None = None,
    evaluation: dict[str, Any],
    channels: Sequence[str],
    notes: str | None,
) -> Experiment:
    """Validate a configuration against its method's schema and record it, frozen.

    Validation happens here rather than at job time so a typo is refused on the create
    screen instead of as a failed job discovered ten minutes later.
    """
    try:
        model_class = get_model_class(model_type)
    except UnknownModelError as exc:
        raise InvalidInputError(str(exc)) from exc

    supported = model_class.capabilities().tasks
    if task not in supported:
        listed = ", ".join(str(entry) for entry in supported)
        raise InvalidInputError(
            f"method {model_type!r} does not support the task {task.value!r}; it supports {listed}"
        )
    if not has_evaluator(task):
        raise InvalidInputError(f"no evaluator is registered for the task {task.value!r} yet")

    try:
        frozen_config = model_class.config_model().model_validate(config).model_dump(mode="json")
        preprocessing_options = PreprocessingOptions.model_validate(preprocessing)
        frozen_evaluation = EvalConfig.model_validate(evaluation).model_dump(mode="json")
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc

    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise NotFoundError(f"no dataset with id {dataset_id}")
        frozen_channels = resolve_channels(conn, dataset_id, channels)
        split = splits_repo.get_split(conn, split_id)
        if split is None:
            raise NotFoundError(f"no split with id {split_id}")
        if split.dataset_id != dataset_id:
            raise InvalidInputError(f"split {split_id} belongs to dataset {split.dataset_id}")
        validate_target(conn, task, target_label, split)
        classes = pin_classes(conn, task, dataset_id)
        profile = region_profiles_repo.get_profile(conn, region_profile_id)
        if profile is None:
            raise NotFoundError(f"no region profile with id {region_profile_id}")
        if profile.dataset_id != dataset_id:
            raise InvalidInputError(
                f"region profile {region_profile_id} belongs to dataset {profile.dataset_id}"
            )
        build_summary = read_build_summary(settings, profile.id)
        if build_summary is None:
            raise InvalidInputError(f"region profile {profile.id} has not been built")
        try:
            load_prepared_build(settings, profile, manifest_sha256=build_summary.manifest_sha256)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        prepared = PreprocessingConfig(
            width=profile.prepared_width,
            height=profile.prepared_height,
            color=preprocessing_options.color,
        )
        # What only the method knows it cannot read — a patch size the frame does not divide —
        # is refused here, by the method, rather than minutes into a job.
        try:
            model_class.check_input(model_class.config_model().model_validate(config), prepared)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        frozen_preprocessing = prepared.model_dump(mode="json")

        # The directory is named after the row, so it cannot be built until the row
        # exists; created first, then recorded, then made.
        experiment = experiments_repo.create_experiment(
            conn,
            name=name,
            dataset_id=dataset_id,
            split_id=split_id,
            region_profile_id=profile.id,
            region_manifest_sha256=build_summary.manifest_sha256,
            model_type=model_type,
            task=task.value,
            target_label=target_label,
            classes=classes,
            model_config=frozen_config,
            preprocessing_config=frozen_preprocessing,
            eval_config=frozen_evaluation,
            channels=frozen_channels,
            artifact_dir="",
            notes=notes,
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        stored = experiments_repo.get_experiment(conn, experiment.id)
    if stored is None:  # pragma: no cover - inserted a moment ago
        raise RuntimeError("the experiment vanished after creation")
    return stored


# --- Deletion ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeletionPreview:
    usage: StorageUsage
    active_jobs: list[Job]
    resident_loaded: bool
    artifact_location_safe: bool
    blocker: str | None


@dataclass(frozen=True)
class DeletionOutcome:
    deleted: bool
    artifacts_removed: bool
    freed: StorageUsage
    artifact_error: str | None


def preview_deletion(
    settings: Settings, resident: ResidentWorker, experiment: Experiment
) -> DeletionPreview:
    artifact_path = experiment_artifact_path(settings, experiment)
    usage = path_usage(artifact_path)
    with connection(settings.db_path) as conn:
        active = jobs_repo.active_jobs_for_experiment(conn, experiment.id)
    snapshot = resident.snapshot()
    location_safe = not experiment.artifact_dir or artifact_path is not None

    blocker: str | None = None
    if active:
        blocker = "Cancel or wait for the active job before deleting this experiment."
    elif not location_safe:
        blocker = "The stored artifact path is outside this experiment's app-owned directory."

    return DeletionPreview(
        usage=usage,
        active_jobs=active,
        resident_loaded=snapshot is not None and snapshot.experiment_id == experiment.id,
        artifact_location_safe=location_safe,
        blocker=blocker,
    )


async def delete_experiment(
    settings: Settings, queue: JobQueue, resident: ResidentWorker, experiment: Experiment
) -> DeletionOutcome:
    """Remove the rows, then the directory — in that order, and never the other way.

    The filesystem cannot join a database transaction, so the deletion that can be rolled
    back goes first. A leftover directory is inert; a row pointing at deleted artifacts
    is a broken screen.
    """
    artifact_path = experiment_artifact_path(settings, experiment)
    if experiment.artifact_dir and artifact_path is None:
        raise ConflictError(
            "refusing to remove an artifact path outside this experiment's app-owned directory"
        )

    async with queue.lifecycle_guard(), resident.eviction_guard():
        # Stable now: no worker or resident can write into the directory while it is
        # measured and removed, so the result reports what was actually reclaimed.
        usage = await asyncio.to_thread(path_usage, artifact_path)
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            active = jobs_repo.active_jobs_for_experiment(conn, experiment.id)
            if active:
                raise ConflictError(
                    "cancel or wait for the active job before deleting this experiment"
                )
            deleted = experiments_repo.delete_experiment(conn, experiment.id)

        removed = True
        artifact_error: str | None = None
        if deleted and artifact_path is not None and artifact_path.exists():
            try:
                await asyncio.to_thread(shutil.rmtree, artifact_path)
            except OSError as exc:
                removed = False
                artifact_error = str(exc)

    return DeletionOutcome(
        deleted=deleted,
        artifacts_removed=removed,
        freed=usage if removed else StorageUsage(),
        artifact_error=artifact_error,
    )


# --- Evaluation ----------------------------------------------------------------------------


def reevaluate(settings: Settings, experiment: Experiment) -> None:
    """Recompute every stored `SampleResult` and `MetricSet` from the stored scores.

    A drifted truth file raises `GroundTruthDriftError`, which is a `ConflictError`.
    """
    with connection(settings.db_path) as conn:
        evaluator_for(experiment.task).evaluate_and_store(conn, experiment)


# --- Diagnostics ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnoseOutcome:
    keys: list[str]
    elapsed_ms: float
    warm: bool


async def diagnose(
    settings: Settings,
    queue: JobQueue,
    resident: ResidentWorker,
    experiment_id: int,
    image_id: int,
) -> DiagnoseOutcome:
    """Ask the resident worker what the method saw in one image (ADR-0026)."""
    experiment = await asyncio.to_thread(load_experiment, settings, experiment_id)
    await asyncio.to_thread(refuse_impossible_diagnose, settings, queue, experiment, image_id)

    started = time.perf_counter()
    try:
        keys, warm = await resident.request(experiment.id, image_id)
    except ResidentError as exc:
        # 503, not 500: the request was well formed and the model exists — the process
        # that answers it did not survive. The stderr tail is here because the reason is
        # nearly always a library's, and "the resident stopped" alone is unactionable.
        detail = str(exc)
        tail = resident.stderr_tail()
        raise UnavailableError(f"{detail}\n{tail}" if tail else detail) from exc

    return DiagnoseOutcome(
        keys=keys, elapsed_ms=(time.perf_counter() - started) * 1000.0, warm=warm
    )


async def clear_diagnostics(
    settings: Settings,
    queue: JobQueue,
    resident: ResidentWorker,
    experiment_id: int,
    scope: PruneScope,
) -> PruneResult:
    """Prune stored diagnostics, never while a job or the resident could write them back.

    The resident is evicted rather than raced: the delete waits for any request in
    flight, which is bounded and is the same guarantee the queue gets.
    """
    experiment = await asyncio.to_thread(load_experiment, settings, experiment_id)
    await asyncio.to_thread(refuse_while_a_job_runs, settings, queue)

    await resident.evict()

    root = settings.experiment_dir(experiment.id) / "diagnostics"
    return await asyncio.to_thread(prune, root, scope=scope)


@dataclass(frozen=True)
class SourceValues:
    """The pinned prepared pixels of one image, resolved but not yet read.

    Split so a caller can answer a conditional request from `etag` alone: the digest is
    over the prepared artifact and the frozen model-input config, both immutable for one
    experiment, so a match means the bytes need not be read at all.
    """

    image_id: int
    etag: str
    prepared_path: Path
    config: PreprocessingConfig
    build: PreparedRegionBuild

    def load(self) -> np.ndarray:
        """Every colour plane as float32, projected back into the source frame."""
        try:
            array = load_array(self.prepared_path, self.config)
        except (OSError, ValueError) as exc:
            # The catalog references files in place, so a source file can disappear between
            # import and now — the same 410 the image tiers give.
            raise GoneError(
                f"the source file for image {self.image_id} is no longer readable"
            ) from exc

        transform = self.build.transform_for(self.image_id)
        return np.stack(
            [transform.project_map(array[..., channel]) for channel in range(array.shape[-1])],
            axis=-1,
        )


def source_values(settings: Settings, experiment: Experiment, image_id: int) -> SourceValues:
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, image_id)
        profile = region_profiles_repo.get_profile(conn, experiment.region_profile_id)
    if image is None:
        raise NotFoundError(f"no image {image_id}")
    if profile is None:
        raise GoneError("the experiment's region profile is missing")

    try:
        build = load_prepared_build(
            settings, profile, manifest_sha256=experiment.region_manifest_sha256
        )
        entry = build.entries.get(image_id)
        if entry is None or entry.source_sha256 != image.sha256:
            raise ValueError(f"image {image_id} is not part of the pinned region build")
        prepared_path = build.image_path(image_id)
    except ValueError as exc:
        raise GoneError(str(exc)) from exc

    config = PreprocessingConfig.model_validate(experiment.preprocessing_config)
    digest = hashlib.sha256(
        f"{entry.prepared_sha256}|{config.model_dump_json()}".encode()
    ).hexdigest()[:16]
    return SourceValues(
        image_id=image_id,
        etag=f'W/"source-{digest}"',
        prepared_path=prepared_path,
        config=config,
        build=build,
    )


@dataclass(frozen=True)
class StoredDiagnostic:
    """One diagnostic, resolved through the index the model wrote — never through a path.

    The caller names a `(key, image_id)` pair and this resolves it; there is no request
    that can name a file, so path traversal is impossible by construction.
    """

    experiment_id: int
    entry: DiagnosticEntry
    target: Path
    recorded_range: DisplayRange | None

    def etag(self, frame: int, fmt: str) -> str | None:
        """A validator over what actually decides the bytes.

        Diagnostics are **not** immutable the way an imported image is: re-running
        inference overwrites an image's error maps in place, so the file's size and
        modification time are part of the identity. Without them a browser would keep
        showing the previous run's picture under the current run's caption.

        **The range is part of the identity too**, and used not to be. The same array drawn
        over a different `(low, high)` is a different picture, so anything that widened a
        key's range — re-inference, and now a diagnostic computed on demand
        (handbook diagnostics.md) — left every
        already-fetched PNG cached at the old scale, with nothing on screen to say so.
        """
        try:
            stat = self.target.stat()
        except OSError:
            return None
        value_range = self.recorded_range
        span = "none" if value_range is None else f"{value_range.low:.9g}:{value_range.high:.9g}"
        return (
            f'W/"diag-{self.experiment_id}-{self.entry.key}-{self.entry.image_id}-{frame}-{fmt}'
            f'-{span}-{stat.st_size}-{stat.st_mtime_ns}"'
        )

    def load(self) -> np.ndarray:
        try:
            array: np.ndarray = np.load(self.target, allow_pickle=False)
        except (OSError, ValueError) as exc:
            # The artifact directory is deletable by design, so a referenced file that is
            # gone is an expected state rather than corruption — the same 410 a missing
            # source file or a missing anomaly map gets.
            raise GoneError(
                f"the payload for diagnostic {self.entry.key!r} is no longer readable"
            ) from exc
        return array

    def _frame_of(self, array: np.ndarray, frame: int) -> np.ndarray:
        if frame >= array.shape[0]:
            raise NotFoundError(
                f"diagnostic {self.entry.key!r} has {array.shape[0]} frame(s); "
                f"there is no frame {frame}"
            )
        plane: np.ndarray = array[frame]
        return plane

    def raw_plane(self, array: np.ndarray, frame: int) -> np.ndarray:
        """The float32 field behind the picture, for the hover readout."""
        if self.entry.kind is DiagnosticKind.IMAGE:
            raise UnsupportedRequestError(
                f"diagnostic {self.entry.key!r} is of kind image — an (H, W, 3) picture in "
                "[0, 1] rather than a field of values, so there is no number to read from it"
            )
        if self.entry.kind is DiagnosticKind.GRID:
            return self._frame_of(array, frame)
        return array

    def render_png(self, array: np.ndarray, frame: int) -> bytes:
        """Colormapped kinds are stretched over the run-wide range the writer recorded."""
        recorded = self.recorded_range
        value_range = None if recorded is None else (recorded.low, recorded.high)
        if self.entry.kind is DiagnosticKind.IMAGE:
            return render_rgb_image(array)
        if self.entry.kind is DiagnosticKind.GRID:
            return render_anomaly_map(
                self._frame_of(array, frame), value_range=value_range, alpha_follows_score=False
            )
        return render_anomaly_map(array, value_range=value_range, alpha_follows_score=False)


def stored_diagnostic(
    settings: Settings, experiment: Experiment, key: str, image_id: int | None
) -> StoredDiagnostic:
    root = settings.experiment_dir(experiment.id) / "diagnostics"
    index = load_index(root)

    entry = next(
        (item for item in index.entries if item.key == key and item.image_id == image_id),
        None,
    )
    if entry is None:
        scope = "run-scoped" if image_id is None else f"image {image_id}"
        raise NotFoundError(f"experiment {experiment.id} recorded no {scope} diagnostic {key!r}")
    if entry.path is None:
        raise UnsupportedRequestError(
            f"diagnostic {key!r} is of kind {entry.kind.value}, whose payload is "
            "already inline in the index; fetching it here would be a second source "
            "of truth for the same data"
        )

    # The range is resolved *before* any validator is computed, because it is part of what
    # decides the bytes: the same array over a different span is a different picture.
    return StoredDiagnostic(
        experiment_id=experiment.id,
        entry=entry,
        target=root / entry.path,
        recorded_range=index.ranges.get(key),
    )
