"""What a job handler is handed, and what it is allowed to do.

A handler receives a `JobContext` and returns a JSON-serializable result. It reports
progress through the context rather than printing, because stdout is the event channel
(ADR-0009), and it checks `should_cancel()` at natural boundaries so that a cancel
request is honoured before the parent has to escalate to a signal.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.protocol import LogEvent, MetricEvent, ProgressEvent, emit


class JobCancelledError(Exception):
    """Raised by `raise_if_cancelled` so a handler can unwind through its own cleanup."""


@dataclass
class JobContext:
    """The handler's whole view of the world."""

    job_id: int
    kind: JobKind
    params: Mapping[str, Any]
    settings: Settings
    cancel_flag: threading.Event = field(default_factory=threading.Event)

    def progress(self, fraction: float, message: str | None = None) -> None:
        emit(ProgressEvent(fraction=min(max(fraction, 0.0), 1.0), message=message))

    def log(self, message: str, level: str = "info") -> None:
        emit(LogEvent(level=_level(level), message=message))

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        emit(MetricEvent(name=name, value=value, step=step))

    def should_cancel(self) -> bool:
        return self.cancel_flag.is_set()

    def raise_if_cancelled(self) -> None:
        if self.should_cancel():
            raise JobCancelledError


def _level(level: str) -> Any:
    """Narrow a caller-supplied level onto the four the protocol allows."""
    return level if level in {"debug", "info", "warning", "error"} else "info"


JobHandler = Callable[[JobContext], dict[str, Any]]
