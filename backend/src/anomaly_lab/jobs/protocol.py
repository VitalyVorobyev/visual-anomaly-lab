"""The worker-to-parent event protocol (ADR-0009).

One JSON object per line on stdout, flushed. Line-delimited JSON is used because it is
trivial to produce, parsed incrementally without a framing layer, and readable in a log
file months later.

Two properties matter more than the schema itself:

  * **stdout is a structured channel.** Anything a worker wants a human to read goes to
    stderr. `emit` is the only thing that should ever write to stdout in a worker.
  * **the parser tolerates garbage.** ADR-0009 accepts that a third-party library may
    print to stdout at any moment. A line that is not a valid event is log material, not
    a crash, so `parse_line` returns `None` rather than raising.
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class ProgressEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ev: Literal["progress"] = "progress"
    fraction: float = Field(ge=0.0, le=1.0)
    message: str | None = None


class LogEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ev: Literal["log"] = "log"
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str


class MetricEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ev: Literal["metric"] = "metric"
    name: str
    value: float
    step: int | None = None


class DoneEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ev: Literal["done"] = "done"
    result: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    # `type` on the wire, `error_type` in Python. The wire name is the one ADR-0009
    # documents; shadowing a builtin in code to match it would be the wrong trade.
    # Split validation/serialization aliases rather than `alias=`, so that constructing
    # the event in Python stays `error_type=...` for both the type checker and the reader.
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    ev: Literal["error"] = "error"
    error_type: str = Field(validation_alias="type", serialization_alias="type")
    message: str
    traceback: str | None = None


JobEvent = Annotated[
    ProgressEvent | LogEvent | MetricEvent | DoneEvent | ErrorEvent,
    Field(discriminator="ev"),
]

_EVENT_ADAPTER: TypeAdapter[JobEvent] = TypeAdapter(JobEvent)


def encode(event: JobEvent) -> str:
    """Serialize one event to its wire form, without a trailing newline."""
    return event.model_dump_json(by_alias=True)


def emit(event: JobEvent) -> None:
    """Write one event to stdout, flushed.

    Flushing every line is what makes progress live rather than arriving in 8 KB bursts
    when the pipe buffer happens to fill.
    """
    sys.stdout.write(encode(event) + "\n")
    sys.stdout.flush()


def parse_line(line: str) -> JobEvent | None:
    """Parse one line, or return `None` if it is not an event.

    Non-JSON output, JSON that is not an object, and objects with an unknown or missing
    `ev` all return `None`. Callers tee such lines to the log and carry on.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return _EVENT_ADAPTER.validate_python(payload)
    except ValidationError:
        return None
