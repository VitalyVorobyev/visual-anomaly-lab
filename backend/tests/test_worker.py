"""The worker process's own behaviour.

`test_job_queue.py` and `test_import.py` cover the parent's half of cancellation — the
signal reaching the process group and the terminal state being recorded. This file covers
the worker's half deterministically, without racing a real scan: the signal handler is
installed and exercised in-process, and `run_job` is called directly with handlers that
succeed, fail, and unwind.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from typing import Any

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs import handlers, worker
from anomaly_lab.jobs.context import JobContext


@pytest.fixture
def job_id(settings: Settings) -> int:
    apply_migrations(settings.db_path)
    with connection(settings.db_path) as conn:
        return jobs_repo.create_job(conn, kind=JobKind.IMPORT, params={"n": 1}).id


def _events(captured: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    """Every JSON line the worker wrote to stdout."""
    out = captured.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def test_sigterm_sets_the_cancel_flag_rather_than_killing_the_worker() -> None:
    """SIGTERM is a request, not a kill. The parent escalates only if it is ignored.

    Sending the signal to this process is safe precisely because the handler under test
    replaces the default disposition first — if it did not, pytest would die here.
    """
    flag = threading.Event()
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        worker.install_cancel_handler(flag)
        os.kill(os.getpid(), signal.SIGTERM)
        # Delivery is asynchronous; the handler runs between bytecodes.
        assert flag.wait(timeout=2.0) is True
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def test_a_handler_that_unwinds_reports_cancelled_not_failed(
    settings: Settings,
    job_id: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def cancelling(context: JobContext) -> dict[str, Any]:
        context.progress(0.5, "halfway")
        context.cancel_flag.set()
        context.raise_if_cancelled()
        raise AssertionError("unreachable")

    monkeypatch.setitem(handlers.LOADERS, JobKind.IMPORT, lambda: cancelling)

    exit_code = worker.run_job(job_id, settings)
    events = _events(capsys)

    assert exit_code == worker.EXIT_CANCELLED
    assert events[0]["ev"] == "progress"
    # No `done` and no `error`: a cancelled job produced neither a result nor a fault.
    assert [event["ev"] for event in events] == ["progress", "log"]
    assert "stopping at the next safe point" in events[-1]["message"]


def test_a_successful_handler_emits_its_result(
    settings: Settings,
    job_id: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setitem(
        handlers.LOADERS, JobKind.IMPORT, lambda: lambda context: {"seen": context.params["n"]}
    )

    exit_code = worker.run_job(job_id, settings)
    events = _events(capsys)

    assert exit_code == worker.EXIT_OK
    assert events[-1] == {"ev": "done", "result": {"seen": 1}}


def test_a_raising_handler_becomes_an_error_event_with_its_traceback(
    settings: Settings,
    job_id: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A worker reports what went wrong; it never dies with a bare traceback."""

    def explode(_context: JobContext) -> dict[str, Any]:
        msg = "the disk is on fire"
        raise RuntimeError(msg)

    monkeypatch.setitem(handlers.LOADERS, JobKind.IMPORT, lambda: explode)

    exit_code = worker.run_job(job_id, settings)
    error = _events(capsys)[-1]

    assert exit_code == worker.EXIT_FAILED
    assert error["ev"] == "error"
    assert error["type"] == "RuntimeError"
    assert error["message"] == "the disk is on fire"
    assert "explode" in error["traceback"]


def test_a_job_that_does_not_exist_is_reported_not_crashed(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    apply_migrations(settings.db_path)

    exit_code = worker.run_job(9999, settings)
    error = _events(capsys)[-1]

    assert exit_code == worker.EXIT_FAILED
    assert error["type"] == "LookupError"


def test_the_entrypoint_rejects_bad_arguments() -> None:
    assert worker.main([]) == worker.EXIT_FAILED
    assert worker.main(["not-a-number"]) == worker.EXIT_FAILED
    assert worker.main(["1", "2"]) == worker.EXIT_FAILED
