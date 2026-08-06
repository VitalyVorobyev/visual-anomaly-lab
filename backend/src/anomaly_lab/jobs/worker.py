"""Job worker entrypoint: `python -m anomaly_lab.jobs.worker <job_id>`.

One process per job (ADR-0009). The process is spawned, never forked, so a crash, an OOM
kill or a wedged native call takes down the worker and nothing else, and every byte of
memory is returned when it exits.

Two rules this module enforces on behalf of every handler:

  * **stdout carries events and nothing else.** Python logging is pointed at stderr here,
    once, so a handler that uses `logging` cannot corrupt the stream.
  * **SIGTERM is a request, not a kill.** It sets the cancel flag; a cooperative handler
    unwinds and exits. The parent escalates to SIGKILL only if that does not happen.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import traceback
from types import FrameType

from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.jobs.handlers import get_handler
from anomaly_lab.jobs.protocol import DoneEvent, ErrorEvent, LogEvent, emit

EXIT_OK = 0
EXIT_FAILED = 1
# 128 + SIGTERM, the shell convention. The parent decides the job's final status from its
# own cancellation record, so this is diagnostic rather than load-bearing.
EXIT_CANCELLED = 143


def _configure_logging() -> None:
    """Send every log record to stderr, so stdout stays a structured channel."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _install_cancel_handler(flag: threading.Event) -> None:
    def _handle(_signum: int, _frame: FrameType | None) -> None:
        flag.set()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


def run_job(job_id: int, settings: Settings) -> int:
    with connection(settings.db_path) as conn:
        job = jobs_repo.get_job(conn, job_id)

    if job is None:
        emit(ErrorEvent(error_type="LookupError", message=f"job {job_id} does not exist"))
        return EXIT_FAILED

    cancel_flag = threading.Event()
    _install_cancel_handler(cancel_flag)

    context = JobContext(
        job_id=job.id,
        kind=job.kind,
        params=job.params,
        settings=settings,
        cancel_flag=cancel_flag,
    )

    try:
        handler = get_handler(job.kind)
        result = handler(context)
    except JobCancelledError:
        emit(LogEvent(level="warning", message="cancelled; stopping at the next safe point"))
        return EXIT_CANCELLED
    # Broad on purpose: a worker's contract is to report what went wrong as an `error`
    # event, never to die with a traceback the parent has to reconstruct from stderr.
    except Exception as exc:
        emit(
            ErrorEvent(
                error_type=type(exc).__name__,
                message=str(exc) or repr(exc),
                traceback=traceback.format_exc(),
            )
        )
        return EXIT_FAILED

    emit(DoneEvent(result=result))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -m anomaly_lab.jobs.worker <job_id>\n")
        return EXIT_FAILED

    try:
        job_id = int(args[0])
    except ValueError:
        sys.stderr.write(f"not a job id: {args[0]!r}\n")
        return EXIT_FAILED

    _configure_logging()
    return run_job(job_id, get_settings())


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
