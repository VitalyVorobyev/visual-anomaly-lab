"""The single-slot FIFO job queue (ADR-0009).

One machine, one user, one MPS device: concurrent jobs would contend for memory and make
timing measurements meaningless, so exactly one worker runs at a time and the rest wait
in `job` rows that are visible and cancellable before they start.

The `job` table is the queue. There is no in-memory pending list to drift away from it —
the runner asks the database for the next queued row every time — so the only state held
in this process is the child that is running right now.

The runner lives on the API process's event loop rather than in a thread, which keeps
child-process I/O and WebSocket fan-out in the same concurrency model. The two database
calls per event are pushed to a threadpool so a slow disk cannot stall the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import Job, JobKind, JobStatus
from anomaly_lab.jobs.protocol import (
    DoneEvent,
    ErrorEvent,
    JobEvent,
    LogEvent,
    ProgressEvent,
    encode,
    parse_line,
)

# How long a cancelled worker is given to unwind before the group is killed outright.
CANCEL_GRACE_SECONDS = 5.0

# The runner blocks on this between wake-ups. `enqueue` signals it, so the timeout is
# only a backstop against a missed signal, not the normal path to picking up work.
IDLE_POLL_SECONDS = 5.0

# Progress events can arrive per image. The row is refreshed at most this often so that
# a REST poll is accurate without turning a scan into thousands of writes; terminal
# states are always written immediately.
PROGRESS_PERSIST_INTERVAL_SECONDS = 0.25

# A subscriber that has stopped reading must not be able to stall the worker's output.
SUBSCRIBER_BUFFER = 256

# The parent's own closing event. Not part of the worker protocol — no worker emits it —
# but a WebSocket client needs to know the stream ended and how.
END_EVENT = "end"


class JobQueue:
    """Owns the worker child process and the fan-out of its events."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._wake = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[int, set[asyncio.Queue[str]]] = {}
        self._current_id: int | None = None
        self._current_process: asyncio.subprocess.Process | None = None
        self._cancelled: set[int] = set()
        self._escalations: set[asyncio.Task[None]] = set()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._settings.jobs_log_dir.mkdir(parents=True, exist_ok=True)
        self._runner = asyncio.create_task(self._run(), name="job-queue")

    async def stop(self) -> None:
        """Stop accepting work and take the running child down with us.

        The sidecar is itself a child of the desktop shell, so leaving a worker behind
        here would produce exactly the orphan the shell's teardown exists to prevent.
        """
        if self._runner is not None:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None

        process = self._current_process
        if process is not None and process.returncode is None:
            self._signal_group(process, signal.SIGTERM)
            with contextlib.suppress(TimeoutError, ProcessLookupError):
                await asyncio.wait_for(process.wait(), CANCEL_GRACE_SECONDS)
            if process.returncode is None:
                self._signal_group(process, signal.SIGKILL)

    # -- public API --------------------------------------------------------------

    def enqueue(
        self,
        *,
        kind: JobKind,
        params: Mapping[str, Any] | None = None,
        experiment_id: int | None = None,
    ) -> Job:
        """Create a queued job row and nudge the runner.

        Synchronous, because it is called from synchronous route handlers running in
        FastAPI's threadpool. The nudge crosses back to the loop thread explicitly.
        """
        with connection(self._settings.db_path) as conn:
            job = jobs_repo.create_job(conn, kind=kind, params=params, experiment_id=experiment_id)
        self._notify()
        return job

    def request_cancel(self, job_id: int) -> bool:
        """Cancel a queued or running job. Returns False if it already finished.

        A queued job is cancelled by a row update. A running one is signalled on the
        event loop, since that is where its process handle lives.
        """
        with connection(self._settings.db_path) as conn:
            job = jobs_repo.get_job(conn, job_id)
            if job is None or job.status.is_terminal:
                return False
            if jobs_repo.cancel_queued_job(conn, job_id):
                return True

        self._cancelled.add(job_id)
        loop = self._loop
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self._terminate(job_id), loop)
        return True

    def subscribe(self, job_id: int) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=SUBSCRIBER_BUFFER)
        self._subscribers.setdefault(job_id, set()).add(queue)
        return queue

    def unsubscribe(self, job_id: int, queue: asyncio.Queue[str]) -> None:
        listeners = self._subscribers.get(job_id)
        if listeners is None:
            return
        listeners.discard(queue)
        if not listeners:
            del self._subscribers[job_id]

    def log_path_for(self, job: Job) -> Path:
        """Where a job's raw output is teed (§6).

        An experiment-bound job logs beside its artifacts; import, verify and prewarm
        have no experiment and log under `data/jobs/logs/`.
        """
        if job.experiment_id is not None:
            return self._settings.experiment_dir(job.experiment_id) / "logs" / f"{job.id}.log"
        return self._settings.jobs_log_dir / f"{job.id}.log"

    # -- runner ------------------------------------------------------------------

    def _notify(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._wake.set)

    async def _run(self) -> None:
        while True:
            job = await asyncio.to_thread(self._next_queued)
            if job is None:
                await self._idle()
                continue
            await self._execute(job)

    async def _idle(self) -> None:
        self._wake.clear()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._wake.wait(), IDLE_POLL_SECONDS)

    def _next_queued(self) -> Job | None:
        with connection(self._settings.db_path) as conn:
            return jobs_repo.next_queued_job(conn)

    async def _execute(self, job: Job) -> None:
        # Cancelled between being picked up and being started: never spawn at all,
        # rather than starting work only to report it as cancelled afterwards.
        if job.id in self._cancelled:
            await asyncio.to_thread(self._finish, job, 0, _JobRunState())
            return

        log_path = self.log_path_for(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        state = _JobRunState()
        try:
            process = await self._spawn(job)
        except OSError as exc:
            state.error = f"the worker could not be started: {exc}"
            await asyncio.to_thread(self._finish, job, 1, state)
            return

        self._current_id = job.id
        self._current_process = process
        # Marked running only once a process exists. Doing it before the spawn would make
        # `running` mean "about to start", and a cancel arriving in that window would find
        # nothing to signal, report the job cancelled, and leave the worker running.
        await asyncio.to_thread(self._mark_running, job.id, str(log_path))
        if job.id in self._cancelled:
            await self._terminate(job.id)

        try:
            stdout, stderr = process.stdout, process.stderr
            if stdout is None or stderr is None:  # pragma: no cover - PIPE was requested
                msg = "the worker was started without pipes"
                raise RuntimeError(msg)

            with log_path.open("a", encoding="utf-8") as log:
                await asyncio.gather(
                    self._drain(job.id, stdout, log, state, is_stderr=False),
                    self._drain(job.id, stderr, log, state, is_stderr=True),
                )
            returncode = await process.wait()
        finally:
            self._current_id = None
            self._current_process = None
            _release_transport(process)

        await asyncio.to_thread(self._finish, job, returncode, state)

    async def _spawn(self, job: Job) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        # The worker resolves its own settings from the environment, so it must be
        # pointed at the same data directory rather than inheriting a default.
        env["ANOMALY_LAB_DATA_DIR"] = str(self._settings.data_dir)
        # Unbuffered, so a flushed event line reaches us immediately.
        env["PYTHONUNBUFFERED"] = "1"

        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "anomaly_lab.jobs.worker",
            str(job.id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            # Its own process group, so cancellation reaches anything the worker itself
            # spawned rather than just the interpreter.
            start_new_session=True,
        )

    async def _drain(
        self,
        job_id: int,
        stream: asyncio.StreamReader,
        log: TextIO,
        state: _JobRunState,
        *,
        is_stderr: bool,
    ) -> None:
        """Read one pipe to EOF, teeing every byte and interpreting what it can.

        Both pipes must be drained for the life of the process: a child whose pipe fills
        up blocks on write, and a blocked worker looks exactly like a hung one.
        """
        while True:
            raw = await stream.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            log.write(line + "\n")
            log.flush()

            event = None if is_stderr else parse_line(line)
            if event is None:
                # Library chatter, a stray print, a native crash message. ADR-0009 names
                # this as the accepted cost of using stdout, so it is log material rather
                # than a parser error.
                if line:
                    await self.publish(job_id, LogEvent(level="info", message=line))
                continue

            await self._observe(job_id, event, state)
            await self.publish(job_id, event)

    async def _observe(self, job_id: int, event: JobEvent, state: _JobRunState) -> None:
        if isinstance(event, ProgressEvent):
            now = time.monotonic()
            state.progress = event.fraction
            state.message = event.message or state.message
            if now - state.last_persisted >= PROGRESS_PERSIST_INTERVAL_SECONDS:
                state.last_persisted = now
                await asyncio.to_thread(
                    self._persist_progress, job_id, event.fraction, event.message
                )
        elif isinstance(event, DoneEvent):
            state.result = event.result
        elif isinstance(event, ErrorEvent):
            state.error = f"{event.error_type}: {event.message}"

    async def publish(self, job_id: int, event: JobEvent) -> None:
        """Fan one event out to every live subscriber of this job.

        The outbound half of the queue's contract with `/ws/jobs/{id}`: subscribers see
        the worker's own frames, unchanged, in the order they were produced.
        """
        listeners = self._subscribers.get(job_id)
        if not listeners:
            return
        payload = encode(event)
        for queue in list(listeners):
            # A client that has stopped reading loses events rather than stalling the
            # worker. It can recover the truth with a REST snapshot (§6).
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    async def publish_end(self, job_id: int, status: JobStatus) -> None:
        """Close the stream with the terminal status.

        Generated here, not by the worker: a client otherwise cannot distinguish a job
        that finished from a socket that dropped.
        """
        listeners = self._subscribers.get(job_id)
        if not listeners:
            return
        payload = f'{{"ev":"{END_EVENT}","job_id":{job_id},"status":"{status.value}"}}'
        for queue in list(listeners):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    # -- database side effects (run in a threadpool) -----------------------------

    def _mark_running(self, job_id: int, log_path: str) -> None:
        with connection(self._settings.db_path) as conn:
            jobs_repo.mark_running(conn, job_id, log_path=log_path)

    def _persist_progress(self, job_id: int, fraction: float, message: str | None) -> None:
        with connection(self._settings.db_path) as conn:
            jobs_repo.update_progress(conn, job_id, fraction=fraction, message=message)

    def _finish(self, job: Job, returncode: int, state: _JobRunState) -> None:
        cancelled = job.id in self._cancelled
        self._cancelled.discard(job.id)

        if cancelled:
            status = JobStatus.CANCELLED
            error = None
        elif returncode == 0 and state.error is None:
            status = JobStatus.SUCCEEDED
            error = None
        else:
            status = JobStatus.FAILED
            error = state.error or f"the worker exited with status {returncode}"

        with connection(self._settings.db_path) as conn:
            jobs_repo.finish_job(
                conn,
                job.id,
                status=status,
                result=state.result,
                error=error,
                message=state.message,
            )

        loop = self._loop
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self.publish_end(job.id, status), loop)

    # -- cancellation ------------------------------------------------------------

    async def _terminate(self, job_id: int) -> None:
        """Ask the worker to stop, and arrange for it to be made to.

        Returns as soon as the signal is sent; the grace period is timed separately so
        that this never blocks the loop that is draining the worker's output.
        """
        process = self._current_process
        if process is None or self._current_id != job_id or process.returncode is not None:
            return

        self._signal_group(process, signal.SIGTERM)
        escalation = asyncio.create_task(self._escalate(job_id, process))
        # Held so the task cannot be garbage collected mid-flight.
        self._escalations.add(escalation)
        escalation.add_done_callback(self._escalations.discard)

    async def _escalate(self, job_id: int, process: asyncio.subprocess.Process) -> None:
        await asyncio.sleep(CANCEL_GRACE_SECONDS)

        still_running = (
            self._current_process is process
            and self._current_id == job_id
            and process.returncode is None
        )
        if still_running:
            # Cooperative cancellation was offered and declined. A cancel button that
            # does not stop the work is worse than no cancel button (ADR-0009).
            self._signal_group(process, signal.SIGKILL)

    @staticmethod
    def _signal_group(process: asyncio.subprocess.Process, sig: int) -> None:
        """Signal the worker's whole process group, not just the interpreter."""
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(process.pid), sig)


def _release_transport(process: asyncio.subprocess.Process) -> None:
    """Close the exited child's transport while its loop is still running.

    `asyncio` has no public API for this, and its finalizer runs whenever the garbage
    collector gets around to it — which, for a long-lived server, can be after the loop
    has closed, at which point the finalizer raises `Event loop is closed` from a
    destructor where nothing can handle it. Closing here is deterministic instead.
    """
    transport = getattr(process, "_transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):
            transport.close()


class _JobRunState:
    """What the parent learned from the event stream while a job ran."""

    def __init__(self) -> None:
        self.progress: float = 0.0
        self.message: str | None = None
        self.result: dict[str, Any] = {}
        self.error: str | None = None
        self.last_persisted: float = 0.0
