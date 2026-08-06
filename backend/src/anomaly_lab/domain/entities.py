"""Domain entities (ADR-0005).

Entities are added as the milestones that use them arrive. `001_initial.sql` remains the
authoritative description of the data model (ADR-0004); these models are how the rest of
the application reads it.

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

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _decode_json_object(value: object) -> object:
    """Accept either the stored JSON string or an already-decoded mapping."""
    if isinstance(value, str):
        return json.loads(value)
    return value


class JobKind(StrEnum):
    IMPORT = "import"
    VERIFY = "verify"
    PREWARM = "prewarm"
    TRAIN = "train"
    INFER = "infer"


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

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

    id: int
    name: str
    root_path: str
    adapter: str | None = None
    manifest_path: str | None = None
    created_at: str
    notes: str | None = None


class Channel(BaseModel):
    """One entry of a dataset's acquisition-channel dictionary."""

    model_config = ConfigDict(frozen=True)

    id: int
    dataset_id: int
    name: str
    position: int = 0


class Sample(BaseModel):
    """One logical physical part, identified by `(dataset_id, group_key, external_id)`."""

    model_config = ConfigDict(frozen=True)

    id: int
    dataset_id: int
    group_key: str
    external_id: str
    label: Label = Label.UNLABELED
    label_source: LabelSource = LabelSource.IMPORT
    notes: str | None = None


class Image(BaseModel):
    """One file on disk, referenced in place and never copied (ADR-0001)."""

    model_config = ConfigDict(frozen=True)

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


class Split(BaseModel):
    """A named, seeded partition of a dataset's samples. Immutable once created."""

    model_config = ConfigDict(frozen=True)

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

    model_config = ConfigDict(frozen=True)

    split_id: int
    sample_id: int
    subset: Subset
