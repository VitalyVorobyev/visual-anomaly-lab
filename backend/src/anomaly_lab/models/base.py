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
    cancellation, logging, diagnostics — is injected (handbook frontend.md), so a plugin is testable
    with a fake reporter and no job system at all.
  * **Configuration is a pydantic model.** Its JSON Schema is served to the frontend,
    which generates the form. Adding a hyperparameter is a Python field and nothing else.

Ground truth is deliberately absent from `ImageRecord`. These methods train on normal
data; handing a model the masks would make it possible to write a plugin that quietly
cheats, and the evaluation layer reads masks from the catalog anyway.
"""

from __future__ import annotations

import importlib.util
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from PIL import Image as PILImage
from pydantic import BaseModel, Field

# One numpy primitive, no model knowledge and no torch, imported so that "where the map
# peaks" is defined once. The evaluation layer must never import a *model* module; nothing
# forbids the reverse, and `models/preprocessing.py` is already read from `eval/` for the
# same reason — one rule, one implementation.
from anomaly_lab.domain.entities import Task
from anomaly_lab.eval.localization import peak_of
from anomaly_lab.map_files import map_file, write_map_file
from anomaly_lab.models.diagnostics import DiagnosticKind, DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.regions.transform import SpatialTransform
from anomaly_lab.schemas import API_MODEL_CONFIG


class Device(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"


class PortableFormat(StrEnum):
    """A deployment representation a fitted method can prove it emits."""

    ONNX = "onnx"


class Capabilities(BaseModel):
    """What a method can do, declared rather than inferred.

    The UI and the job layer branch on these flags. They never branch on a registry key —
    the moment they do, adding a method stops being a file plus an entry.
    """

    model_config = API_MODEL_CONFIG

    tasks: list[Task] = Field(default_factory=lambda: [Task.ANOMALY])
    """The tasks this method can be run as (ADR-0039).

    An experiment is created for one task and refused for a method that does not list it.
    The default is what every method written before tasks existed does, so none of them
    had to change.
    """
    requires_training: bool = True
    produces_anomaly_map: bool = True
    produces_diagnostics: bool = False
    channel_aware: bool = False
    dataset_specific: bool = False
    supports_resume: bool = False
    """Whether a finished run can be continued for further steps (handbook jobs.md).

    A capability rather than a special case, because most methods have no notion of a
    step: `pixel_reference` builds a median over the training set and is either fitted or
    not. A method that declares this must also satisfy `SupportsResume`, and the train
    handler checks that the two agree rather than trusting the flag.
    """
    portable_formats: list[PortableFormat] = Field(default_factory=list)
    """Formats this plugin can export with numerical parity (ADR-0034).

    Empty means unsupported, not "unknown". The API and UI may offer an export only
    when the requested value is present here and the instance satisfies the matching
    structural protocol.
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
    """A model's verdict on one image. Higher `score` means more anomalous.

    Every task keeps the image-level `score`, so ranking and the gallery work for all of
    them (ADR-0039). A supervised segmentation method also writes a label map through
    `InferContext.write_label_map` and returns its path here; its `score` is the share of
    the image it assigned to a class other than background. A detection method writes its
    boxes through `InferContext.write_instances`; its `score` is its highest confidence, 0
    when it found nothing.
    """

    model_config = API_MODEL_CONFIG

    image_id: int
    score: float
    anomaly_map: Path | None = None
    label_map: Path | None = None
    instances: Path | None = None
    """A detection method's boxes, written through `InferContext.write_instances`."""
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


class TargetProvider(Protocol):
    """The ground truth a targeted task fits on (ADR-0039, ADR-0040).

    The only way a plugin ever sees ground truth. An `anomaly` run gets `None` instead, so a
    method ranking by unlikeness to normal cannot read a defect mask by construction.
    """

    @property
    def label_key(self) -> str:
        """The class this run segments."""
        ...

    def mask(self, image_id: int) -> np.ndarray:
        """Where the class is in one training image: boolean, in the prepared frame."""
        ...


IGNORE_INDEX = 255
"""A label-map pixel no pinned class answers for — letterbox padding, or a class the run was
not created with. Nothing fits on it and nothing counts it. Label maps are 8-bit and 0 is
background, which leaves 254 classes."""


class LabelTargetProvider(Protocol):
    """The ground truth a supervised segmentation task fits on (ADR-0039).

    A sibling of `TargetProvider` rather than a method on it: a targeted run is about one
    class and answers with a boolean mask, a supervised run is about a pinned class list and
    answers with a label map. Absent for every other task.
    """

    @property
    def classes(self) -> tuple[str, ...]:
        """The pinned classes: `classes[i]` is label index `i + 1`, and 0 is background."""
        ...

    def labels(self, image_id: int) -> np.ndarray:
        """One training image's label map: `uint8`, in the prepared frame. `IGNORE_INDEX`
        (255) marks a pixel no pinned class answers for — the letterbox padding, or a class
        the run was not created with — and is fitted on by nobody."""
        ...


Box = tuple[float, float, float, float]
"""`(x0, y0, x1, y1)` in pixel-edge coordinates: `x0 <= x < x1`, so a box covering exactly the
pixel at column 3 is `(3, y0, 4, y1)` — the convention of an instances file and of a crop."""

MAX_INSTANCES_PER_IMAGE = 100
"""At most this many detections of one image are stored and counted — COCO's cap. A method
keeps its most confident ones; the write seam refuses more rather than choosing for it."""


@dataclass(frozen=True)
class TargetBox:
    """One true object instance: its class and its box."""

    label_key: str
    box: Box


@dataclass(frozen=True)
class PredictedInstance:
    """One detection: its class, its box and how confident the method is in it.

    A confidence ranks detections within one run and means nothing across runs (ADR-0028).
    """

    label_key: str
    box: Box
    confidence: float


class BoxTargetProvider(Protocol):
    """The ground truth a detection task fits on (ADR-0039).

    The third sibling: a detection run is about a pinned class list, like segmentation, and
    answers with boxes. Absent for every other task.
    """

    @property
    def classes(self) -> tuple[str, ...]:
        """The pinned classes, in the run's order."""
        ...

    def boxes(self, image_id: int) -> list[TargetBox]:
        """One training image's object instances of the pinned classes, in the prepared
        frame. An empty list is a confirmed absence of every one of them."""
        ...


@dataclass
class TrainContext(ModelContext):
    """`fit`'s view of the world.

    `val` carries the validation records when the split has any. EfficientAD needs
    held-out normals to fit its score-normalization quantiles; a split with no `val`
    subset — VisA's official one-class protocol, for instance — passes an empty sequence,
    and a model that needs them must fall back visibly rather than silently.

    `targets` is `None` for `anomaly`. For a targeted task it answers for every record
    `fit` is given. `label_targets` is the same for a supervised segmentation task and
    `box_targets` for a detection task, each `None` for every other: these three fields are
    the only way ground truth reaches a plugin.
    """

    val: Sequence[ImageRecord] = ()
    targets: TargetProvider | None = None
    label_targets: LabelTargetProvider | None = None
    box_targets: BoxTargetProvider | None = None


@dataclass
class InferContext(ModelContext):
    """`predict`'s view of the world."""

    maps_subdir: str = "maps"
    """Which directory under `artifact_dir` receives this run's maps.

    Set by the caller, never by a plugin, and the default is what every job uses. It
    exists because `predict` writes a map unconditionally, and a caller that is *not* a
    run must not be able to overwrite one: an on-demand diagnostic request scores a single
    image to see what its branches did, and dropping the result into `maps/{image_id}.npz`
    would replace a stored map under a `range.json` fitted by a different run — leaving
    that image's map a generation ahead of its own score row, with nothing saying so
    (ADR-0026).
    """

    map_transform: Callable[[int], SpatialTransform] | None = None
    """Injected: the pinned spatial transform that places an image's map in its source.

    Plugins emit maps in the prepared frame. The experiment boundary owns the pinned
    transform, so the map is stored beside it here and projected on read (`read_map`), and
    no method needs to know that region profiles exist.
    """

    mask_projector: Callable[[int, np.ndarray], np.ndarray] | None = None
    """The same projection for a boolean mask, which is resampled nearest, never blended."""

    label_projector: Callable[[int, np.ndarray], np.ndarray] | None = None
    """The same projection for a `uint8` label map: nearest, and background outside the crop."""

    box_projector: Callable[[int, Box], Box | None] | None = None
    """The same projection for a box: clipped to the crop, `None` when nothing of it is left."""

    _map_extremes: list[tuple[float, float]] = field(default_factory=list)

    _map_peaks: dict[int, tuple[int, int]] = field(default_factory=dict)
    """Where each written map's largest value landed, in the frame it was stored in.

    Collected here, at the one boundary every method's map already passes through, so that
    no plugin has to know the localization verdict exists — and so that the peak is taken
    from the *projected* array, which is the array the reviewer will look at.
    """

    @property
    def maps_dir(self) -> Path:
        """Where anomaly maps are written. Created on first use, not at construction."""
        path = self.artifact_dir / self.maps_subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def map_path(self, image_id: int) -> Path:
        """Where one image's map is written (ADR-0007); `anomaly_lab.map_files` owns the format."""
        return map_file(self.maps_dir, image_id)

    def write_map(self, image_id: int, array: np.ndarray) -> Path:
        """Persist one anomaly map, raw float32, and return where it went.

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
        transform = self.map_transform(image_id) if self.map_transform is not None else None
        # Range and peak are taken on the source-frame array a reader will get back, which
        # `read_map` reproduces bit for bit from what is stored.
        projected = transform.project_map(values) if transform is not None else values
        finite = projected[np.isfinite(projected)]
        if finite.size == 0:
            raise ValueError(f"anomaly map for image {image_id} has no covered finite pixels")
        write_map_file(path, values, transform)
        self._map_extremes.append((float(finite.min()), float(np.percentile(finite, 99.9))))
        peak = peak_of(projected)
        if peak is not None:
            self._map_peaks[image_id] = peak
        return path

    def mask_path(self, image_id: int) -> Path:
        """Where one image's predicted mask is written, beside its map."""
        return self.maps_dir / f"{image_id}.mask.png"

    def write_mask(self, image_id: int, mask: np.ndarray) -> Path:
        """Persist a method's own foreground decision for one image, as a 0/255 PNG.

        For a method whose mask is more than its map above a threshold — a refined or
        post-processed region. The map stays the foreground probability, and a run that
        writes no mask is read by thresholding it. The mask arrives in the prepared frame
        and is stored in the source frame, like the map.
        """
        values = np.squeeze(np.asarray(mask)).astype(bool)
        if values.ndim != 2:
            msg = (
                f"a mask must be 2-D; image {image_id} produced an array of shape {np.shape(mask)}"
            )
            raise ValueError(msg)
        if self.mask_projector is not None:
            values = np.asarray(self.mask_projector(image_id, values), dtype=bool)
        path = self.mask_path(image_id)
        temporary = path.with_suffix(".tmp.png")
        try:
            PILImage.fromarray(values.astype(np.uint8) * 255, mode="L").save(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def label_map_path(self, image_id: int) -> Path:
        """Where one image's predicted label map is written, beside its map."""
        return self.maps_dir / f"{image_id}.labels.png"

    def write_label_map(self, image_id: int, labels: np.ndarray, *, classes: int) -> Path:
        """Persist a supervised segmentation method's class per pixel, as an 8-bit PNG.

        0 is background and `i + 1` is the run's `classes[i]`; `classes` is how many the run
        pinned, and a value above it is a plugin bug, reported here rather than as a confusing
        confusion matrix later. The map arrives in the prepared frame and is stored in the
        source frame, resampled nearest: a class index is a name, never blended. Pixels a
        region crop left uncovered are background.
        """
        values = np.squeeze(np.asarray(labels))
        if values.ndim != 2:
            msg = (
                f"a label map must be 2-D; image {image_id} produced an array of shape "
                f"{np.shape(labels)}"
            )
            raise ValueError(msg)
        if not np.issubdtype(values.dtype, np.integer):
            msg = f"a label map holds class indices; image {image_id} gave {values.dtype}"
            raise ValueError(msg)
        if values.size and (int(values.min()) < 0 or int(values.max()) > classes):
            msg = (
                f"image {image_id}'s label map holds indices {int(values.min())}.."
                f"{int(values.max())}, outside 0..{classes}"
            )
            raise ValueError(msg)
        values = values.astype(np.uint8)
        if self.label_projector is not None:
            values = np.asarray(self.label_projector(image_id, values), dtype=np.uint8)
        path = self.label_map_path(image_id)
        temporary = path.with_suffix(".tmp.png")
        try:
            PILImage.fromarray(values, mode="L").save(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def instances_path(self, image_id: int) -> Path:
        """Where one image's detections are written, beside its map."""
        return self.maps_dir / f"{image_id}.instances.json"

    def write_instances(
        self, image_id: int, instances: Sequence[PredictedInstance], *, classes: Sequence[str]
    ) -> Path:
        """Persist a detection method's boxes for one image, as JSON, most confident first.

        `classes` is the run's pinned list; a class outside it, a box with no area, a
        confidence that is not finite, or more than `MAX_INSTANCES_PER_IMAGE` detections is
        a plugin bug, reported here rather than as a confusing AP later. Boxes arrive in the
        prepared frame and are stored in the source frame; one that falls wholly outside the
        region crop is dropped, because no source pixel is under it. An empty list is written
        too: it says the method looked and found nothing.
        """
        if len(instances) > MAX_INSTANCES_PER_IMAGE:
            msg = (
                f"image {image_id} has {len(instances)} detections; at most "
                f"{MAX_INSTANCES_PER_IMAGE} are stored, so keep the most confident"
            )
            raise ValueError(msg)
        known = set(classes)
        kept: list[PredictedInstance] = []
        for instance in instances:
            if instance.label_key not in known:
                msg = (
                    f"image {image_id} has a detection of {instance.label_key!r}, which the run "
                    f"does not pin ({', '.join(classes)})"
                )
                raise ValueError(msg)
            if not np.isfinite(instance.confidence):
                msg = f"image {image_id} has a detection with confidence {instance.confidence}"
                raise ValueError(msg)
            x0, y0, x1, y1 = (float(value) for value in instance.box)
            if not np.all(np.isfinite((x0, y0, x1, y1))) or x1 <= x0 or y1 <= y0:
                raise ValueError(f"image {image_id} has a box with no area: {instance.box}")
            box: Box | None = (x0, y0, x1, y1)
            if self.box_projector is not None:
                box = self.box_projector(image_id, (x0, y0, x1, y1))
            if box is not None:
                kept.append(PredictedInstance(instance.label_key, box, float(instance.confidence)))
        kept.sort(key=lambda instance: -instance.confidence)
        stored = [
            {
                "label_key": instance.label_key,
                "box": [float(value) for value in instance.box],
                "confidence": instance.confidence,
            }
            for instance in kept
        ]
        path = self.instances_path(image_id)
        temporary = path.with_suffix(".tmp.json")
        try:
            temporary.write_text(json.dumps({"instances": stored}), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def peak_for(self, image_id: int) -> tuple[int, int] | None:
        """`(x, y)` of this image's map peak, or `None` if no map was written for it."""
        return self._map_peaks.get(image_id)

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
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        """Raise `ValueError`, with a reason a reader can act on, for input this method cannot read.

        Called when an experiment is created (and when the reference studio previews), so a
        combination that could only fail at fit — a patch size the prepared frame does not
        divide — is refused on the create screen by name rather than minutes into a job. Any
        input by default.
        """
        return

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
    """A method whose training can be continued from where it stopped (handbook jobs.md).

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
        continuation rather than a second curve starting at zero (handbook jobs.md).
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
