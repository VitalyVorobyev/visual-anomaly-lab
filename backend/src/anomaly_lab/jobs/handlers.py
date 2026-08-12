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


def _reference_import() -> JobHandler:
    from anomaly_lab.datasets.reference_packs import run_reference_pack_job

    return run_reference_pack_job


def _verify() -> JobHandler:
    from anomaly_lab.datasets.verify import run_verify_job

    return run_verify_job


def _prewarm() -> JobHandler:
    from anomaly_lab.media.prewarm import run_prewarm_job

    return run_prewarm_job


def _train() -> JobHandler:
    from anomaly_lab.experiments.train import run_train_job

    return run_train_job


def _infer() -> JobHandler:
    from anomaly_lab.experiments.infer import run_infer_job

    return run_infer_job


def _distill() -> JobHandler:
    from anomaly_lab.models.distill import run_distill_job

    return run_distill_job


# `train` and `infer` arrived in M3 and cost exactly what was predicted: one entry each
# and one handler function each. The queue, protocol, cancellation and event fan-out
# needed no change to carry a method that trains a neural network for minutes on a GPU,
# which is the claim ADR-0009 made and this is the test of it.
#
# `distill` in M6 cost the same one entry, plus a migration for the `kind` CHECK — which
# is the one place the queue's kind-agnosticism stops at the database. It belongs to no
# experiment, exactly as `import` does, and produces an asset rather than a result row.
LOADERS: dict[JobKind, Callable[[], JobHandler]] = {
    JobKind.IMPORT: _import_scan,
    JobKind.REFERENCE_IMPORT: _reference_import,
    JobKind.VERIFY: _verify,
    JobKind.PREWARM: _prewarm,
    JobKind.TRAIN: _train,
    JobKind.INFER: _infer,
    JobKind.DISTILL: _distill,
}


def get_handler(kind: JobKind) -> JobHandler:
    loader = LOADERS.get(kind)
    if loader is None:
        msg = f"no handler is registered for job kind {kind.value!r}"
        raise UnknownJobKindError(msg)
    return loader()


def supported_kinds() -> frozenset[JobKind]:
    return frozenset(LOADERS)
