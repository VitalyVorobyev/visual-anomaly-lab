"""Domain entities (ADR-0005).

Entities are added as the milestones that use them arrive. The numbered SQL migrations
remain the authoritative description of the data model (ADR-0004); these models are how
the rest of the application reads it.

Two invariants from ADR-0005 are visible in the shapes below and must stay that way:

  * `Sample` owns `label` and, through `SplitAssignment`, subset membership. There is no
    image-level label and no image-level assignment, so every view of a part necessarily
    shares a subset.
  * `Channel` is a per-dataset row and `Image.channel_id` is optional. Nothing here
    encodes how many channels a dataset has.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from anomaly_lab.schemas import API_MODEL_CONFIG


def _decode_json_object(value: object) -> object:
    """Accept either the stored JSON string or an already-decoded mapping."""
    if isinstance(value, str):
        return json.loads(value)
    return value


class JobKind(StrEnum):
    IMPORT = "import"
    REFERENCE_IMPORT = "reference_import"
    VERIFY = "verify"
    PREWARM = "prewarm"
    TRAIN = "train"
    INFER = "infer"
    DISTILL = "distill"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


class Label(StrEnum):
    NORMAL = "normal"
    DEFECT = "defect"
    UNLABELED = "unlabeled"


class LabelSource(StrEnum):
    """Where a label came from, so hand corrections survive a re-import (ADR-0013)."""

    IMPORT = "import"
    MANUAL = "manual"


class Subset(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Job(BaseModel):
    """An asynchronous execution record (§6)."""

    model_config = API_MODEL_CONFIG

    id: int
    kind: JobKind
    experiment_id: int | None = None
    status: JobStatus
    progress: float = 0.0
    message: str | None = None
    log_path: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @field_validator("params", "result", mode="before")
    @classmethod
    def _decode_json_columns(cls, value: object) -> object:
        """Both columns are stored as JSON strings; callers work with dicts."""
        return _decode_json_object(value)


class Dataset(BaseModel):
    """A named collection of samples rooted at a path on disk."""

    model_config = API_MODEL_CONFIG

    id: int
    name: str
    root_path: str
    adapter: str | None = None
    manifest_path: str | None = None
    created_at: str
    notes: str | None = None


class Channel(BaseModel):
    """One entry of a dataset's acquisition-channel dictionary."""

    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    name: str
    position: int = 0


class Sample(BaseModel):
    """One logical physical part, identified by `(dataset_id, group_key, external_id)`."""

    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    group_key: str
    external_id: str
    label: Label = Label.UNLABELED
    label_source: LabelSource = LabelSource.IMPORT
    notes: str | None = None


class Image(BaseModel):
    """One file on disk, referenced in place and never copied (ADR-0001)."""

    model_config = API_MODEL_CONFIG

    id: int
    sample_id: int
    channel_id: int | None = None
    path: str
    width: int
    height: int
    bit_depth: int
    file_size: int
    sha256: str
    imported_at: str


class MaskKind(StrEnum):
    """What a mask asserts. Only ground truth exists today; predictions are artifacts."""

    GROUND_TRUTH = "ground_truth"


class Mask(BaseModel):
    """Pixel-level ground truth for one image, referenced in place like the image itself.

    A mask belongs to an `Image` rather than a `Sample` because a multi-view sample can be
    annotated in one view and not another, and because the mask has to align with a
    specific image's pixel grid to mean anything.

    Imported masks remain source provenance. Migration 005 added an optional digest:
    old rows are hashed when first used as an annotation base, while newly discovered
    masks may carry one immediately.
    """

    model_config = API_MODEL_CONFIG

    id: int
    image_id: int
    path: str
    kind: MaskKind = MaskKind.GROUND_TRUTH
    sha256: str | None = None


class Split(BaseModel):
    """A named, seeded partition of a dataset's samples. Immutable once created."""

    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    name: str
    strategy: str
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    @field_validator("params", mode="before")
    @classmethod
    def _decode_split_params(cls, value: object) -> object:
        return _decode_json_object(value)


class SplitAssignment(BaseModel):
    """Sample-level subset membership. There is deliberately no image-level equivalent."""

    model_config = API_MODEL_CONFIG

    split_id: int
    sample_id: int
    subset: Subset


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class Aggregation(StrEnum):
    """How a sample's per-image scores become one number (ADR-0011)."""

    MAX = "max"
    MEAN = "mean"


class Experiment(BaseModel):
    """One method, on one split, with the configuration frozen at creation.

    The three config columns are kept apart because they answer different questions:
    `model_config_` is the plugin's own hyperparameters, `preprocessing_config` is what
    every method is made to see identically, and `eval_config` is how the results are
    read. Only the last may be reinterpreted without re-running anything.
    """

    model_config = API_MODEL_CONFIG

    id: int
    name: str
    dataset_id: int
    split_id: int
    model_type: str
    # `model_config` is taken by pydantic itself, so the field carries a trailing
    # underscore in Python and its database and wire name through the alias.
    model_config_: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    preprocessing_config: dict[str, Any] = Field(default_factory=dict)
    eval_config: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    artifact_dir: str
    created_at: str
    notes: str | None = None

    @field_validator("model_config_", "preprocessing_config", "eval_config", mode="before")
    @classmethod
    def _decode_config_columns(cls, value: object) -> object:
        return _decode_json_object(value)


class ImageResult(BaseModel):
    """One model's verdict on one image. Higher `score` means more anomalous."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    image_id: int
    score: float
    map_path: str | None = None
    inference_ms: float


class SampleResult(BaseModel):
    """The aggregated verdict on one physical part, with the aggregation recorded."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    sample_id: int
    agg_score: float
    aggregation: Aggregation


class MetricSet(BaseModel):
    """Threshold-independent metrics for one subset of one experiment (ADR-0011)."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    subset: Subset
    metrics: dict[str, Any] = Field(default_factory=dict)
    ground_truth_digest: str | None = None
    computed_at: str

    @field_validator("metrics", mode="before")
    @classmethod
    def _decode_metrics(cls, value: object) -> object:
        return _decode_json_object(value)
