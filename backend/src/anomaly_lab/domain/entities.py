"""Domain entities (ADR-0005).

Only the entities M1 actually returns are modelled here. The rest of schema v1 exists in
`001_initial.sql`, which is the authoritative description of the data model (ADR-0004);
models are added as the milestones that use them arrive.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobKind(StrEnum):
    IMPORT = "import"
    TRAIN = "train"
    INFER = "infer"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None

    @field_validator("params", mode="before")
    @classmethod
    def _decode_params(cls, value: object) -> object:
        """`job.params` is stored as a JSON string; callers work with a dict."""
        if isinstance(value, str):
            return json.loads(value)
        return value
