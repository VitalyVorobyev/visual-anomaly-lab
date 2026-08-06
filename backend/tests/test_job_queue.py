"""The job queue, its fan-out, and the worker subprocess.

The end-to-end case here deliberately runs a job whose kind has no handler. That exercises
the whole pipeline — spawn, stdout parsing, error propagation, terminal state — without
needing a handler to exist yet. The success, progress and cancellation paths are covered
where real handlers land: the import scan and the thumbnail pre-warm.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import Job, JobKind, JobStatus
from anomaly_lab.jobs.protocol import LogEvent, ProgressEvent
from anomaly_lab.jobs.queue import SUBSCRIBER_BUFFER, JobQueue

TERMINAL_WAIT_SECONDS = 30.0


def _queue(client: TestClient) -> JobQueue:
    """The running app's queue. `TestClient.app` is typed as a bare ASGI callable."""
    app = cast(FastAPI, client.app)
    queue: JobQueue = app.state.job_queue
    return queue


def _await_terminal(client: TestClient, job_id: int) -> dict[str, Any]:
    """Poll the snapshot endpoint until the job stops moving."""
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    msg = f"job {job_id} never reached a terminal state"
    raise AssertionError(msg)


def test_enqueue_creates_a_visible_queued_job(settings: Settings) -> None:
    """A job is inspectable and cancellable before it starts (ADR-0009)."""
    apply_migrations(settings.db_path)
    job = JobQueue(settings).enqueue(kind=JobKind.PREWARM, params={"dataset_id": 3})

    assert job.status is JobStatus.QUEUED
    assert job.params == {"dataset_id": 3}
    assert job.result == {}
    assert job.started_at is None


def test_the_queue_is_fifo(settings: Settings) -> None:
    apply_migrations(settings.db_path)
    queue = JobQueue(settings)
    first = queue.enqueue(kind=JobKind.IMPORT)
    queue.enqueue(kind=JobKind.IMPORT)

    with connection(settings.db_path) as conn:
        assert jobs_repo.next_queued_job(conn) == first


def test_cancelling_a_queued_job_never_starts_it(settings: Settings) -> None:
    apply_migrations(settings.db_path)
    queue = JobQueue(settings)
    job = queue.enqueue(kind=JobKind.IMPORT)

    assert queue.request_cancel(job.id) is True

    with connection(settings.db_path) as conn:
        cancelled = jobs_repo.get_job(conn, job.id)
    assert cancelled is not None
    assert cancelled.status is JobStatus.CANCELLED
    assert cancelled.started_at is None


def test_cancelling_a_finished_job_reports_rather_than_pretends(settings: Settings) -> None:
    apply_migrations(settings.db_path)
    with connection(settings.db_path) as conn:
        job = jobs_repo.create_job(conn, kind=JobKind.IMPORT)
        jobs_repo.finish_job(conn, job.id, status=JobStatus.SUCCEEDED)

    assert JobQueue(settings).request_cancel(job.id) is False


def test_log_paths_separate_experiment_bound_jobs_from_the_rest(settings: Settings) -> None:
    """§6: import, verify and prewarm have no experiment directory to log into."""
    queue = JobQueue(settings)
    free = Job(id=7, kind=JobKind.IMPORT, status=JobStatus.QUEUED)
    bound = Job(id=8, kind=JobKind.TRAIN, status=JobStatus.QUEUED, experiment_id=12)

    assert queue.log_path_for(free) == settings.data_dir / "jobs" / "logs" / "7.log"
    assert (
        queue.log_path_for(bound) == settings.data_dir / "artifacts" / "exp-12" / "logs" / "8.log"
    )


def test_subscribers_receive_events_and_a_closing_frame(settings: Settings) -> None:
    async def scenario() -> list[str]:
        queue = JobQueue(settings)
        subscription = queue.subscribe(42)

        await queue.publish(42, ProgressEvent(fraction=0.5, message="halfway"))
        await queue.publish_end(42, JobStatus.SUCCEEDED)

        return [subscription.get_nowait() for _ in range(subscription.qsize())]

    frames = [json.loads(frame) for frame in asyncio.run(scenario())]

    assert frames[0]["ev"] == "progress"
    assert frames[0]["fraction"] == 0.5
    # Without this frame a client cannot tell "finished" from "socket dropped".
    assert frames[-1] == {"ev": "end", "job_id": 42, "status": "succeeded"}


def test_a_subscriber_that_stops_reading_is_dropped_not_waited_for(settings: Settings) -> None:
    """A stalled UI must never be able to block the worker's output."""

    async def scenario() -> int:
        queue = JobQueue(settings)
        subscription = queue.subscribe(1)
        for index in range(SUBSCRIBER_BUFFER + 25):
            await queue.publish(1, LogEvent(message=f"line {index}"))
        return subscription.qsize()

    assert asyncio.run(scenario()) == SUBSCRIBER_BUFFER


def test_publishing_to_a_job_nobody_watches_is_harmless(settings: Settings) -> None:
    async def scenario() -> None:
        await JobQueue(settings).publish(99, LogEvent(message="into the void"))

    asyncio.run(scenario())


def test_a_job_with_no_handler_fails_with_its_reason_recorded(client: TestClient) -> None:
    """End-to-end: spawn, event stream, parse, persist, terminal state.

    `train` has no handler until M3, which makes it the honest way to prove the pipeline
    reports a worker's failure rather than hanging or silently claiming success.
    """
    job = _queue(client).enqueue(kind=JobKind.TRAIN, params={"experiment_id": 1})

    finished = _await_terminal(client, job.id)
    log_tail: list[str] = finished["log_tail"]

    assert finished["status"] == "failed"
    assert "no handler is registered" in finished["error"]
    assert finished["finished_at"] is not None
    # The raw stream is on disk whether or not anyone was watching (ADR-0009).
    assert any("UnknownJobKindError" in line for line in log_tail)


def test_a_running_job_left_by_a_previous_process_is_reconciled(settings: Settings) -> None:
    """The owner of a `running` row at startup is provably gone (ADR-0009)."""
    apply_migrations(settings.db_path)
    with connection(settings.db_path) as conn:
        job = jobs_repo.create_job(conn, kind=JobKind.IMPORT)
        jobs_repo.mark_running(conn, job.id, log_path=str(settings.jobs_log_dir / "x.log"))

    with TestClient(create_app(settings)) as client:
        payload = client.get(f"/api/jobs/{job.id}").json()

    assert payload["status"] == "failed"
    assert "restarted" in payload["error"]


def test_the_job_list_narrows_by_kind_and_status(client: TestClient) -> None:
    queue = _queue(client)
    queue.enqueue(kind=JobKind.PREWARM)
    queue.enqueue(kind=JobKind.VERIFY)

    prewarm = client.get("/api/jobs", params={"kind": "prewarm"}).json()

    assert [job["kind"] for job in prewarm] == ["prewarm"]


def test_an_unknown_job_is_a_404_not_an_empty_snapshot(client: TestClient) -> None:
    assert client.get("/api/jobs/9999").status_code == 404
    assert client.post("/api/jobs/9999/cancel").status_code == 404


def test_cancelling_an_already_finished_job_is_a_conflict(client: TestClient) -> None:
    job = _queue(client).enqueue(kind=JobKind.TRAIN)
    _await_terminal(client, job.id)

    response = client.post(f"/api/jobs/{job.id}/cancel")

    assert response.status_code == 409
    assert "already finished" in response.json()["detail"]
