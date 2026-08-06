"""The kind → handler registry.

An explicit table of lazy loaders rather than import-time decorators, for two reasons:
reading this file tells you every kind of work the application can start, and the worker
imports a handler's dependencies only when it is about to run one — an import job should
not pay for torch, and a train job should not pay for Pillow.
"""

from __future__ import annotations

from collections.abc import Callable

from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobHandler


class UnknownJobKindError(Exception):
    """No handler is registered for this kind — a bug, not a user error."""


def _import_scan() -> JobHandler:
    from anomaly_lab.datasets.scan import run_scan_job

    return run_scan_job


def _verify() -> JobHandler:
    from anomaly_lab.datasets.verify import run_verify_job

    return run_verify_job


def _prewarm() -> JobHandler:
    from anomaly_lab.media.prewarm import run_prewarm_job

    return run_prewarm_job


# `train` and `infer` arrive with the model plugins in M3. The queue, protocol,
# cancellation and event fan-out are already kind-agnostic, so each will need nothing
# here but another entry.
LOADERS: dict[JobKind, Callable[[], JobHandler]] = {
    JobKind.IMPORT: _import_scan,
    JobKind.VERIFY: _verify,
    JobKind.PREWARM: _prewarm,
}


def get_handler(kind: JobKind) -> JobHandler:
    loader = LOADERS.get(kind)
    if loader is None:
        msg = f"no handler is registered for job kind {kind.value!r}"
        raise UnknownJobKindError(msg)
    return loader()


def supported_kinds() -> frozenset[JobKind]:
    return frozenset(LOADERS)
