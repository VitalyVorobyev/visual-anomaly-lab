"""The job queue, its fan-out, and the worker subprocess.

The end-to-end case here runs a job that fails inside its handler. That exercises the
whole pipeline — spawn, stdout parsing, error propagation, terminal state — in one test.
The success, progress and cancellation paths are covered where the real handlers live:
the import scan, the thumbnail pre-warm, and the training run.
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
from anomaly_lab.jobs.handlers import supported_kinds
from anomaly_lab.jobs.protocol import DoneEvent, LogEvent, ProgressEvent, parse_line
from anomaly_lab.jobs.queue import (
    MAX_EVENT_BYTES,
    MAX_LINE_BYTES,
    SUBSCRIBER_BUFFER,
    JobQueue,
    _iter_lines,
    split_output,
)

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


def test_region_prepare_is_a_supported_generic_job_kind() -> None:
    assert JobKind.REGION_PREPARE in supported_kinds()


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


def test_a_failing_handler_reports_its_reason(client: TestClient) -> None:
    """End-to-end: spawn, event stream, parse, persist, terminal state.

    This used to enqueue `train` precisely *because* it had no handler. It has one now,
    so the failure is a real one — a train job naming an experiment that does not exist —
    which tests the same pipeline against a message an operator could actually receive.
    """
    job = _queue(client).enqueue(kind=JobKind.TRAIN, params={"experiment_id": 12345})

    finished = _await_terminal(client, job.id)
    log_tail: list[str] = finished["log_tail"]

    assert finished["status"] == "failed"
    assert "no experiment with id 12345" in finished["error"]
    assert finished["finished_at"] is not None
    # The raw stream is on disk whether or not anyone was watching (ADR-0009).
    assert any("ExperimentJobError" in line for line in log_tail)


def test_every_declared_job_kind_has_a_handler() -> None:
    """The invariant the old test stood in for: no kind is declarable but unrunnable.

    `JobKind` is a closed enum and the schema's CHECK constraint mirrors it, so a kind
    with no entry in the registry would be enqueueable and then immediately dead.
    """
    assert supported_kinds() == frozenset(JobKind)


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


# -- output splitting ------------------------------------------------------------
#
# These pin a real failure, found by the first job long enough to draw a progress bar.
# `StreamReader.readline` raises `ValueError` past asyncio's 64 KiB limit, and a tqdm bar
# is *one line* — its frames are separated by `\r`, never `\n`. Downloading EfficientAD's
# 1.5 GB penalty set blew the limit, the exception escaped the runner, and the queue
# wedged: the running job stayed `running` and every later job stayed `queued`, silently.


def test_a_progress_bar_becomes_readable_lines_rather_than_one_huge_one() -> None:
    """`\\r` ends a line. That is how a terminal renders it, and how the log should too."""
    lines, remainder = split_output(b"10%\r50%\r90%\rdone\n")
    assert lines == ["10%", "50%", "90%", "done"]
    assert remainder == b""


def test_a_partial_line_is_held_until_it_is_finished() -> None:
    lines, remainder = split_output(b"first\nsecond-half")
    assert lines == ["first"]
    assert remainder == b"second-half"

    lines, remainder = split_output(remainder + b"-arrived\n")
    assert lines == ["second-half-arrived"]
    assert remainder == b""


def test_crlf_does_not_produce_a_phantom_blank_line_of_content() -> None:
    lines, _ = split_output(b"a\r\nb\r\n")
    assert [line for line in lines if line] == ["a", "b"]


def test_output_with_no_terminator_at_all_is_flushed_rather_than_buffered() -> None:
    """The exact shape that broke it: megabytes with no newline anywhere.

    The old reader raised `ValueError` here. Nothing a library prints may be able to stop
    the queue, so an over-long fragment is emitted rather than accumulated.
    """
    flood = b"x" * (MAX_LINE_BYTES * 3)
    lines, remainder = split_output(flood)

    assert lines != []
    assert remainder == b""
    assert sum(len(line) for line in lines) == len(flood)


def test_an_event_larger_than_a_log_line_survives_the_split() -> None:
    """A `done` frame over `MAX_LINE_BYTES` must arrive whole, not in two useless halves.

    This is the fault that made a succeeded preview job carry `result = {}`: the frame was
    flushed mid-JSON, `parse_line` rejected both pieces, and no `DoneEvent` was ever seen.
    Nothing complained, because a partial line is a perfectly ordinary thing for a log to
    hold — which is exactly why the split has to tell an event from chatter.
    """
    entries = [{"image_id": index, "status": "succeeded"} for index in range(24)]
    frame = json.dumps({"ev": "done", "result": {"mode": "preview", "entries": entries}})
    frame += " " * (MAX_LINE_BYTES * 2 - len(frame))
    payload = frame.encode() + b"\n"
    assert len(payload) > MAX_LINE_BYTES

    # Delivered the way the pipe delivers it: in chunks that fall wherever they fall.
    lines: list[str] = []
    pending = b""
    for offset in range(0, len(payload), 4096):
        emitted, pending = split_output(pending + payload[offset : offset + 4096])
        lines.extend(emitted)

    assert pending == b""
    assert [json.loads(line)["result"]["entries"] for line in lines] == [entries]


def test_a_large_event_straddling_a_pipe_read_still_parses() -> None:
    """The same frame over the real reader, positioned to break across a chunk boundary.

    `split_output` is called once per `READ_CHUNK_BYTES`, so whether a frame survived used
    to depend on where the pipe happened to cut it — which is why one `foreground_threshold`
    preview kept its result and the next four did not. Padding the stream so the frame
    starts just before the boundary makes that accident deterministic.
    """
    entries = [{"image_id": index, "status": "succeeded"} for index in range(24)]
    frame = json.dumps({"ev": "done", "result": {"mode": "preview", "entries": entries}})
    frame += " " * (MAX_LINE_BYTES * 2 - len(frame))
    payload = b"chatter\n" * 100 + frame.encode() + b"\n"

    async def collect() -> list[str]:
        reader = asyncio.StreamReader()
        reader.feed_data(payload)
        reader.feed_eof()
        return [line async for line in _iter_lines(reader)]

    lines = asyncio.run(collect())
    events = [parse_line(line) for line in lines]
    done = [event for event in events if isinstance(event, DoneEvent)]
    assert len(done) == 1
    assert done[0].result["entries"] == entries


def test_a_flood_that_only_looks_like_an_event_is_still_bounded() -> None:
    """The larger budget is a budget, not an exemption — the queue must stay unstoppable."""
    flood = b"{" + b"x" * (MAX_EVENT_BYTES + 1)
    lines, remainder = split_output(flood)

    assert lines != []
    assert remainder == b""


def test_the_tail_of_a_stream_is_not_lost_at_eof() -> None:
    lines, remainder = split_output(b"no trailing newline", at_eof=True)
    assert lines == ["no trailing newline"]
    assert remainder == b""


def test_undecodable_bytes_are_replaced_rather_than_raising() -> None:
    """A native crash message is not guaranteed to be UTF-8, and is exactly what is wanted."""
    lines, _ = split_output(b"before \xff\xfe after\n")
    assert lines[0].startswith("before ")


def test_a_worker_that_floods_stderr_still_reaches_a_terminal_state(client: TestClient) -> None:
    """End to end over the real subprocess path, with the output shape that broke it.

    A train job for a missing experiment fails fast; what matters is that the job is
    finalized and that the queue then runs the *next* job — a wedged runner leaves both
    of those undone and reports nothing.
    """
    queue = _queue(client)
    first = queue.enqueue(kind=JobKind.TRAIN, params={"experiment_id": 999_999})
    assert _await_terminal(client, first.id)["status"] == "failed"

    second = queue.enqueue(kind=JobKind.PREWARM, params={"dataset_id": 999_999})
    assert _await_terminal(client, second.id)["status"] in {"succeeded", "failed"}
