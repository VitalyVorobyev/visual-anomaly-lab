"""The model plugin interface (ADR-0007, §5).

Every anomaly-detection method is a plugin behind this one interface. The rest of the
application knows the interface and the registry key, and nothing else — no route, no
schema, no UI screen mentions a method by name.

Three seams carry the weight:

  * **Scores are per-image.** Aggregating a part's channels into one verdict is the
    evaluation layer's job (§8), not the model's. A `channel_aware` model may consult
    `ImageRecord.channel` internally and still returns one `Prediction` per input image.
  * **A model touches nothing but its context.** No SQLite, no `Settings`, no writing
    outside `artifact_dir`. Everything a long-running plugin needs — progress,
    cancellation, logging, diagnostics — is injected (ADR-0014), so a plugin is testable
    with a fake reporter and no job system at all.
  * **Configuration is a pydantic model.** Its JSON Schema is served to the frontend,
    which generates the form. Adding a hyperparameter is a Python field and nothing else.

Ground truth is deliberately absent from `ImageRecord`. These methods train on normal
data; handing a model the masks would make it possible to write a plugin that quietly
cheats, and the evaluation layer reads masks from the catalog anyway.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.models.diagnostics import DiagnosticKind, DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.schemas import API_MODEL_CONFIG


class Device(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class Capabilities(BaseModel):
    """What a method can do, declared rather than inferred.

    The UI and the job layer branch on these flags. They never branch on a registry key —
    the moment they do, adding a method stops being a file plus an entry.
    """

    model_config = API_MODEL_CONFIG

    requires_training: bool = True
    produces_anomaly_map: bool = True
    produces_diagnostics: bool = False
    channel_aware: bool = False
    dataset_specific: bool = False
    supports_resume: bool = False
    """Whether a finished run can be continued for further steps (ADR-0025).

    A capability rather than a special case, because most methods have no notion of a
    step: `pixel_reference` builds a median over the training set and is either fitted or
    not. A method that declares this must also satisfy `SupportsResume`, and the train
    handler checks that the two agree rather than trusting the flag.
    """
    preferred_device: Device = Device.CPU


class Availability(BaseModel):
    """Whether this method can run here, and if not, what is missing.

    The deep methods live behind an optional dependency group, so a checkout that never
    installed torch should show them greyed out with the reason, rather than offering
    them and failing minutes later inside a worker.
    """

    model_config = API_MODEL_CONFIG

    available: bool = True
    reason: str | None = None


class ImageRecord(BaseModel):
    """One image a model is asked to look at."""

    model_config = API_MODEL_CONFIG

    image_id: int
    sample_id: int
    channel: str | None = None
    path: Path


class Prediction(BaseModel):
    """A model's verdict on one image. Higher `score` means more anomalous."""

    model_config = API_MODEL_CONFIG

    image_id: int
    score: float
    anomaly_map: Path | None = None
    inference_ms: float = 0.0


class ProgressReporter(Protocol):
    """The subset of the job context a model is allowed to see.

    `JobContext` satisfies this structurally, so the job layer passes itself and the
    model never learns that a job system exists. Tests pass `NullReporter`.
    """

    def progress(self, fraction: float, message: str | None = None) -> None: ...
    def log(self, message: str, level: str = "info") -> None: ...
    def metric(self, name: str, value: float, step: int | None = None) -> None: ...
    def should_cancel(self) -> bool: ...


class NullReporter:
    """A reporter that discards everything — the default for unit tests."""

    def progress(self, fraction: float, message: str | None = None) -> None:
        return

    def log(self, message: str, level: str = "info") -> None:
        return

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        return

    def should_cancel(self) -> bool:
        return False


class ModelCancelledError(Exception):
    """Raised by `raise_if_cancelled` so a plugin unwinds through its own cleanup."""


@dataclass
class ModelContext:
    """Everything a plugin needs and must not invent for itself."""

    artifact_dir: Path
    """The experiment's directory — the only place a model may write its own outputs."""

    cache_dir: Path
    """Shared, app-managed storage for downloaded assets: pretrained weights, penalty
    sets. Separate from `artifact_dir` because these files belong to the *method* and are
    reused across every experiment that runs it — copying ImageNette per run would be
    absurd. A model may read and write here; it still may not touch anything else."""

    preprocessing: PreprocessingConfig
    device: Device
    reporter: ProgressReporter
    diagnostics: DiagnosticWriter

    def progress(self, fraction: float, message: str | None = None) -> None:
        self.reporter.progress(fraction, message)

    def log(self, message: str, level: str = "info") -> None:
        self.reporter.log(message, level)

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        self.reporter.metric(name, value, step)

    def should_cancel(self) -> bool:
        return self.reporter.should_cancel()

    def raise_if_cancelled(self) -> None:
        if self.should_cancel():
            raise ModelCancelledError

    def emit_diagnostic(
        self,
        key: str,
        title: str,
        kind: DiagnosticKind,
        payload: np.ndarray | dict[str, Any],
        *,
        image_id: int | None = None,
        description: str | None = None,
    ) -> None:
        """Record something about how the model reached its answer.

        A no-op when the run has diagnostics disabled, so a plugin calls it
        unconditionally rather than guarding every call site.
        """
        self.diagnostics.emit(key, title, kind, payload, image_id=image_id, description=description)


@dataclass
class TrainContext(ModelContext):
    """`fit`'s view of the world.

    `val` carries the validation records when the split has any. EfficientAD needs
    held-out normals to fit its score-normalization quantiles; a split with no `val`
    subset — VisA's official one-class protocol, for instance — passes an empty sequence,
    and a model that needs them must fall back visibly rather than silently.
    """

    val: Sequence[ImageRecord] = ()


@dataclass
class InferContext(ModelContext):
    """`predict`'s view of the world."""

    maps_subdir: str = "maps"
    """Which directory under `artifact_dir` receives this run's maps.

    Set by the caller, never by a plugin, and the default is what every job uses. It
    exists because `predict` writes a map unconditionally, and a caller that is *not* a
    run must not be able to overwrite one: an on-demand diagnostic request scores a single
    image to see what its branches did, and dropping the result into `maps/{image_id}.npy`
    would replace a stored map under a `range.json` fitted by a different run — leaving
    that image's map a generation ahead of its own score row, with nothing saying so
    (ADR-0026).
    """

    _map_extremes: list[tuple[float, float]] = field(default_factory=list)

    @property
    def maps_dir(self) -> Path:
        """Where anomaly maps are written. Created on first use, not at construction."""
        path = self.artifact_dir / self.maps_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def map_path(self, image_id: int) -> Path:
        """The canonical float32 `.npy` path for one image's map (ADR-0007)."""
        return self.maps_dir / f"{image_id}.npy"

    def write_map(self, image_id: int, array: np.ndarray) -> Path:
        """Persist one anomaly map as float32 and return where it went.

        Raw and unnormalized on purpose: colormap, range and opacity are view decisions,
        applied when the map is looked at, so they stay changeable after the expensive
        computation is done.

        A robust display range is accumulated as a side effect. Every map in a run has to
        be drawn on the same scale or two images cannot be compared by eye, and rendering
        each one against its own min and max would make a clean part look as alarming as
        a defective one. The high end is each map's 99.9th percentile rather than its
        maximum, so a single hot pixel cannot black out every other map in the run.

        An anomaly map is **2-D**, and that is enforced here rather than trusted. Torch
        models hand back `(1, H, W)` or `(1, 1, H, W)` depending on how many leading
        dimensions the caller happened to index away, and a stored map with a stray axis
        is not rejected by anything until the evaluation layer tries to resample it —
        several minutes of inference later, with an error naming a dtype rather than the
        plugin that produced it. Singleton axes are dropped; anything else is a bug in
        the plugin and is reported as one, immediately.
        """
        path = self.map_path(image_id)
        values = np.ascontiguousarray(np.squeeze(array), dtype=np.float32)
        if values.ndim != 2:
            msg = (
                f"an anomaly map must be 2-D; image {image_id} produced an array of "
                f"shape {np.shape(array)}, which is not a 2-D map with singleton axes"
            )
            raise ValueError(msg)
        np.save(path, values)
        self._map_extremes.append((float(values.min()), float(np.percentile(values, 99.9))))
        return path

    def display_range(self) -> tuple[float, float] | None:
        """`(low, high)` for rendering this run's maps, or `None` if none were written."""
        if not self._map_extremes:
            return None
        low = min(low for low, _ in self._map_extremes)
        high = max(high for _, high in self._map_extremes)
        return (low, high if high > low else low + 1.0)


class AnomalyModel(ABC):
    """One anomaly-detection method.

    Subclasses narrow `__init__`'s parameter to the type their own `config_model()`
    returns; mypy exempts `__init__` from override checks, so this stays honest without
    a generic parameter running through every call site.
    """

    title: ClassVar[str] = ""
    """Human-readable name for the method picker."""

    summary: ClassVar[str] = ""
    """One sentence explaining what the method does, shown under the name."""

    def __init__(self, config: BaseModel) -> None:
        self._base_config = config

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[BaseModel]:
        """The pydantic model whose JSON Schema becomes this method's config form."""

    @classmethod
    @abstractmethod
    def capabilities(cls) -> Capabilities: ...

    @classmethod
    def availability(cls) -> Availability:
        """Whether this method's dependencies are installed. Available by default."""
        return Availability()

    @classmethod
    def build(cls, config: dict[str, Any]) -> AnomalyModel:
        """Validate a stored config mapping into this method's own config type."""
        return cls(cls.config_model().model_validate(config))

    @abstractmethod
    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        """Learn from normal images. Called once, in a `train` job."""

    @abstractmethod
    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        """Score images. One `Prediction` per input, in the same order."""

    @abstractmethod
    def save(self, artifact_dir: Path) -> None:
        """Persist whatever `load` needs to reproduce this fitted model."""

    @abstractmethod
    def load(self, artifact_dir: Path) -> None:
        """Restore a fitted model written by `save`."""


@runtime_checkable
class SupportsResume(Protocol):
    """A method whose training can be continued from where it stopped (ADR-0025).

    A **Protocol**, not two more abstract methods on `AnomalyModel`. Most methods have no
    steps to continue — `pixel_reference` computes a median and is done — and making them
    carry `raise NotImplementedError` stubs would put a lie in the interface: the ABC is
    meant to say what every method does. The train handler asks `isinstance`, which works
    because this is `runtime_checkable`.

    The pair must agree with `Capabilities.supports_resume`. A method that declares the
    flag without satisfying the protocol is a plugin bug, and the handler says so by name
    rather than failing later inside a checkpoint load.
    """

    def completed_steps(self) -> int:
        """Total steps this fitted model has trained for, across every run.

        Absolute, not per-run. It is what makes a continued run's loss curve a
        continuation rather than a second curve starting at zero (ADR-0020).
        """
        ...

    def fit_more(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        additional_steps: int,
    ) -> None:
        """Continue a loaded model for `additional_steps` further steps.

        Called after `load`, never instead of `fit`. An implementation that cannot resume
        exactly — an older checkpoint holding weights but no optimizer state — must refuse
        by name rather than silently restarting the optimizer, because a continuation that
        resets Adam's moments is not a continuation.
        """
        ...


def evenly_spaced(count: int, limit: int) -> list[int]:
    """Indices sampled across the whole range, not the first `limit` of it.

    Every plugin here has at least one pass whose cost is linear in the training set and
    whose accuracy is not — building a per-pixel reference, fitting a quantile. Capping
    such a pass by taking the first N is the tempting shortcut and the wrong one:
    datasets arrive in acquisition order often enough that the first 128 images are one
    production batch, one lighting condition, one shift.
    """
    if count <= limit:
        return list(range(count))
    return [round(index) for index in np.linspace(0, count - 1, limit)]


def module_available(module: str, extra: str, purpose: str) -> Availability:
    """Report whether an optional dependency is importable, without importing it.

    `find_spec` rather than a try/import: importing torch to discover that torch is
    installed costs seconds in the API process, and this is called every time the method
    picker is opened.
    """
    if importlib.util.find_spec(module) is not None:
        return Availability()
    return Availability(
        available=False,
        reason=(
            f"{purpose} needs the optional '{extra}' dependency group. "
            f"Install it with: uv sync --directory backend --extra {extra}"
        ),
    )


class ModelDescription(BaseModel):
    """What the method picker needs, without importing the method's dependencies."""

    model_config = API_MODEL_CONFIG

    key: str
    title: str
    summary: str
    capabilities: Capabilities
    availability: Availability
    config_schema: dict[str, Any] = Field(default_factory=dict)
