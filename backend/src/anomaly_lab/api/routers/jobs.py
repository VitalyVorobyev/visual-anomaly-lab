"""Job status, cancellation, and the live event stream.

The client rule is **snapshot then subscribe** (§6): `GET /api/jobs/{id}` returns current
status, progress and a log tail, and only then is the WebSocket opened. First load,
reconnection after a dropped socket, and a UI opened halfway through a running job are
therefore all the same code path, with no replay buffer in the server and no missed-event
reconciliation in the client.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import Job, JobKind, JobStatus
from anomaly_lab.jobs.queue import END_EVENT, JobQueue
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/jobs", tags=["jobs"])
ws_router = APIRouter(tags=["ws"])

# Enough to fill a console on first paint without shipping a whole training log.
LOG_TAIL_LINES = 200


class JobSummary(BaseModel):
    """A job as the list view sees it."""

    model_config = API_MODEL_CONFIG

    id: int
    kind: JobKind
    status: JobStatus
    progress: float
    message: str | None = None
    experiment_id: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class JobDetail(JobSummary):
    """A job plus everything needed to render its screen before the socket opens."""

    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    log_tail: list[str] = Field(default_factory=list)


def summary_of(job: Job) -> JobSummary:
    return JobSummary(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        message=job.message,
        experiment_id=job.experiment_id,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
    )


def _log_tail(log_path: str | None, limit: int = LOG_TAIL_LINES) -> list[str]:
    """The last few lines of a job's log, or nothing if it has not written one yet.

    Read whole rather than seeked: a job log is kilobytes, and a partial read of the
    final line would show the operator a truncated event.
    """
    if not log_path:
        return []
    path = Path(log_path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle.readlines()[-limit:]]


def _load_job(request: Request, job_id: int) -> Job:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        job = jobs_repo.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"no job with id {job_id}")
    return job


@router.get("", summary="Recent jobs, newest first")
def list_jobs(
    request: Request,
    kind: JobKind | None = None,
    status: JobStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[JobSummary]:
    """List jobs, optionally narrowed to one kind or one status."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        found = jobs_repo.list_jobs(conn, kind=kind, status=status, limit=limit)
    return [summary_of(job) for job in found]


@router.get("/{job_id}", summary="Current state of one job, with a log tail")
def get_job(request: Request, job_id: int) -> JobDetail:
    """Snapshot a job. Take this before subscribing to its event stream."""
    job = _load_job(request, job_id)
    return JobDetail(
        **summary_of(job).model_dump(),
        params=dict(job.params),
        result=dict(job.result),
        log_tail=_log_tail(job.log_path),
    )


@router.post("/{job_id}/cancel", summary="Cancel a queued or running job")
def cancel_job(request: Request, job_id: int) -> JobSummary:
    """Request cancellation.

    A queued job is cancelled outright. A running one is sent SIGTERM, and killed if it
    has not stopped within the grace period — cancellation that does not actually stop
    the work would be worse than none (ADR-0009).
    """
    job = _load_job(request, job_id)
    if job.status.is_terminal:
        raise HTTPException(
            status_code=409,
            detail=f"job {job_id} already finished as {job.status.value}",
        )

    queue: JobQueue = request.app.state.job_queue
    queue.request_cancel(job_id)
    return summary_of(_load_job(request, job_id))


@ws_router.websocket("/ws/jobs/{job_id}")
async def stream_job(websocket: WebSocket, job_id: int) -> None:
    """Live events for one job, in the worker's own envelope (ADR-0009).

    The stream carries `progress`, `log`, `metric`, `done` and `error` frames verbatim,
    followed by one parent-generated `end` frame naming the terminal status — a client
    otherwise has no way to distinguish "finished" from "socket dropped".
    """
    queue: JobQueue = websocket.app.state.job_queue
    await websocket.accept()
    subscription = queue.subscribe(job_id)
    try:
        while True:
            payload = await subscription.get()
            await websocket.send_text(payload)
            if _is_end_frame(payload):
                return
    except WebSocketDisconnect:
        return
    finally:
        queue.unsubscribe(job_id, subscription)


def _is_end_frame(payload: str) -> bool:
    with contextlib.suppress(json.JSONDecodeError):
        decoded = json.loads(payload)
        return isinstance(decoded, dict) and decoded.get("ev") == END_EVENT
    return False
