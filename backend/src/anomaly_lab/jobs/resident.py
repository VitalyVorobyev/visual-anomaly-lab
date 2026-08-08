"""The resident inference worker (ADR-0026).

ADR-0009 gives every unit of work its own process and runs them one at a time. That is
right for training and scoring, and wrong for "show me what the branches did on this
image": the model load dominates, so a job per request would pay eleven seconds of startup
to do a hundred milliseconds of work, and would queue behind whatever is training.

So one process is kept alive per experiment, and requests are written to its stdin. It is
the only long-lived compute state in this application, and everything below exists to keep
that from being a problem.

**One lock, not a check.** `diagnose` and `evict` take the same `asyncio.Lock`, and the
job queue awaits `evict` before it spawns a worker. A resident and a job worker therefore
cannot coexist — not because anything checks, but because starting a job has to wait for
the lock an in-flight request holds. The failure this prevents is the expensive one: MPS
out of memory during training because a resident still held the device. The cost is
honest: a train job can be delayed by up to one request, bounded by `REQUEST_TIMEOUT`.

**Keyed by `(experiment_id, generation)`.** The generation is a fingerprint of the model
directory. A retrain changes it, the manager sees a different key, and the resident is
replaced — so serving from stale weights is impossible by construction rather than by an
eviction hook firing in time.

**A crash is normal.** There is no supervision and no restart policy: the process is
killed for any protocol deviation, and the next request spawns another.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.jobs.protocol import (
    REQUEST_ID,
    DoneEvent,
    ErrorEvent,
    LogEvent,
    parse_line,
)
from anomaly_lab.jobs.queue import CANCEL_GRACE_SECONDS, release_transport

logger = logging.getLogger(__name__)

# How long one request may take before the resident is killed rather than waited on. The
# first request of a run loads a checkpoint, which is seconds; this is the ceiling that
# keeps a wedged native call from holding the job queue behind it indefinitely.
REQUEST_TIMEOUT_SECONDS = 120.0

# A resident with nothing to do is a process holding an accelerator. Evicted after this.
IDLE_TIMEOUT_SECONDS = 600.0

# Kept for the 503 body. A failure to load a model is usually explained by a library on
# stderr, and the alternative is an error that says only "the resident stopped".
STDERR_TAIL_LINES = 20


class ResidentError(Exception):
    """The request could not be served by a resident worker."""


@dataclass(frozen=True)
class ResidentSnapshot:
    """What `/api/health` reports. A plain read, so it never waits on the lock."""

    experiment_id: int
    generation: str
    evicted_in_seconds: float
    requests_served: int


def generation_of(model_dir: Path) -> str:
    """A fingerprint of a checkpoint directory: what makes staleness impossible.

    Sizes and modification times rather than content: a checkpoint is tens of megabytes
    and this is computed on the request path. It cannot miss a retrain — `save` rewrites
    every file — and the consequence of a false *positive* is one respawn, which is
    cheap. Content hashing would trade seconds per request for that.
    """
    if not model_dir.is_dir():
        return "none"
    digest = hashlib.sha256()
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}|".encode())
    return digest.hexdigest()[:16]


class ResidentWorker:
    """Owns at most one live inspector process, and serialises everything about it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._experiment_id: int | None = None
        self._generation: str | None = None
        self._stderr: list[str] = []
        self._stderr_task: asyncio.Task[None] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._requests = 0
        self._next_rid = 0
        self._idle_deadline = 0.0

    # -- lifecycle ---------------------------------------------------------------

    async def stop(self) -> None:
        """Take the resident down with us, as the queue does with its worker."""
        async with self._lock:
            await self._kill()

    def snapshot(self) -> ResidentSnapshot | None:
        """Whether a resident is live, and for what. Lock-free by design.

        `/api/health` is the only place the one invisible process in this system becomes
        visible, and a health endpoint that can block behind a model load is not one.
        """
        process = self._process
        if process is None or process.returncode is not None:
            return None
        if self._experiment_id is None or self._generation is None:
            return None
        return ResidentSnapshot(
            experiment_id=self._experiment_id,
            generation=self._generation,
            evicted_in_seconds=max(self._idle_deadline - time.monotonic(), 0.0),
            requests_served=self._requests,
        )

    # -- the two operations that matter ------------------------------------------

    async def evict(self) -> None:
        """Ensure no resident is running, waiting for any in-flight request to finish.

        This is what the job queue awaits before it spawns (`before_spawn`). It is the
        whole coexistence guarantee, so it must not be made non-blocking to "avoid
        delaying a job" — the delay is the guarantee.
        """
        async with self._lock:
            await self._kill()

    async def request(self, experiment_id: int, image_id: int) -> tuple[list[str], bool]:
        """Diagnose one image; return its keys and whether the resident was already warm.

        Every failure below kills the process. That is not pessimism: the states this can
        fail in — a stream whose framing is in doubt, a child that stopped reading, a
        request that never came back — are all states in which the next answer could
        belong to a different question. Respawning costs one model load.
        """
        async with self._lock:
            generation = await asyncio.to_thread(
                generation_of, self._settings.experiment_dir(experiment_id) / MODEL_SUBDIR
            )
            warm = self._matches(experiment_id, generation)

            try:
                if not warm:
                    await self._kill()
                    await self._spawn(experiment_id, generation)
                keys = await self._exchange(image_id)
            except ResidentError:
                await self._kill()
                raise

            self._requests += 1
            self._arm_idle_timer()
            return keys, warm

    # -- process management ------------------------------------------------------

    def _matches(self, experiment_id: int, generation: str) -> bool:
        process = self._process
        return (
            process is not None
            and process.returncode is None
            and self._experiment_id == experiment_id
            and self._generation == generation
        )

    async def _spawn(self, experiment_id: int, generation: str) -> None:
        env = dict(os.environ)
        env["ANOMALY_LAB_DATA_DIR"] = str(self._settings.data_dir)
        env["PYTHONUNBUFFERED"] = "1"

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "anomaly_lab.jobs.inspector",
                str(experiment_id),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Its own group, so a kill reaches whatever a library spawned beneath it —
                # the same reason the job queue does this.
                start_new_session=True,
            )
        except OSError as exc:
            msg = f"the resident worker could not be started: {exc}"
            raise ResidentError(msg) from exc

        self._process = process
        self._experiment_id = experiment_id
        self._generation = generation
        self._stderr = []
        self._loop = asyncio.get_running_loop()
        if process.stderr is not None:
            self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))

        # The ready line is the model load. Waiting for it here rather than inside the
        # first request means a checkpoint that cannot be loaded fails as a failure to
        # start, with the library's own reason, rather than as a mysterious timeout.
        await self._await_ready()

    async def _await_ready(self) -> None:
        event = await self._read_until_terminal(expect_ready=True)
        if isinstance(event, ErrorEvent):
            msg = f"the resident worker could not load experiment: {event.message}"
            raise ResidentError(msg)

    async def _kill(self) -> None:
        process, self._process = self._process, None
        self._experiment_id = None
        self._generation = None
        self._cancel_idle_timer()

        task, self._stderr_task = self._stderr_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if process is None:
            return
        if process.returncode is None:
            _signal_group(process, signal.SIGTERM)
            with contextlib.suppress(TimeoutError, ProcessLookupError):
                await asyncio.wait_for(process.wait(), CANCEL_GRACE_SECONDS)
            if process.returncode is None:
                _signal_group(process, signal.SIGKILL)
                with contextlib.suppress(TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(process.wait(), CANCEL_GRACE_SECONDS)
        release_transport(process)

    # -- the request exchange ----------------------------------------------------

    async def _exchange(self, image_id: int) -> list[str]:
        process = self._process
        if process is None or process.stdin is None:
            msg = "the resident worker is not running"
            raise ResidentError(msg)

        self._next_rid += 1
        rid = self._next_rid
        frame = f'{{"{REQUEST_ID}": {rid}, "image_id": {image_id}}}\n'
        try:
            process.stdin.write(frame.encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            msg = f"the resident worker stopped reading requests: {exc}"
            raise ResidentError(msg) from exc

        event = await self._read_until_terminal()
        if isinstance(event, ErrorEvent):
            raise ResidentError(event.message)
        if not isinstance(event, DoneEvent):
            msg = f"the resident worker sent an unexpected frame: {event}"
            raise ResidentError(msg)

        if event.result.get(REQUEST_ID) != rid:
            # Impossible while one request is in flight, which the lock guarantees — so if
            # it happens the stream is not what this code thinks it is, and the only safe
            # answer is to stop trusting the process.
            msg = "the resident worker answered a question that was not asked"
            raise ResidentError(msg)
        keys = event.result.get("keys")
        return [str(key) for key in keys] if isinstance(keys, list) else []

    async def _read_until_terminal(
        self, *, expect_ready: bool = False
    ) -> DoneEvent | ErrorEvent | LogEvent:
        """Read stdout until a `done` or `error` frame, or until the process dies.

        `expect_ready` treats the inspector's ready `log` line as terminal: it is the
        acknowledgement that the checkpoint loaded, and there is no `done` to follow it.
        """
        process = self._process
        if process is None or process.stdout is None:  # pragma: no cover - PIPE requested
            msg = "the resident worker was started without pipes"
            raise ResidentError(msg)
        stdout = process.stdout

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                while True:
                    raw = await stdout.readline()
                    if not raw:
                        detail = " ".join(self._stderr[-3:]) or "no output"
                        msg = f"the resident worker exited: {detail}"
                        raise ResidentError(msg)
                    event = parse_line(raw.decode("utf-8", errors="replace"))
                    if event is None:
                        continue
                    if isinstance(event, ErrorEvent | DoneEvent):
                        return event
                    if expect_ready and isinstance(event, LogEvent):
                        return event
                    logger.debug("resident: %s", event)
        except TimeoutError as exc:
            msg = f"the resident worker did not answer within {REQUEST_TIMEOUT_SECONDS:.0f}s"
            raise ResidentError(msg) from exc

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        """Keep the pipe empty and the last lines to hand.

        Both halves matter. A child whose stderr fills blocks on write and looks hung, and
        the reason a model failed to load is nearly always on this stream.
        """
        while True:
            raw = await stream.readline()
            if not raw:
                return
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr.append(line)
                del self._stderr[:-STDERR_TAIL_LINES]

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr[-5:])

    # -- idle eviction -----------------------------------------------------------

    def _arm_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
        self._idle_task = asyncio.create_task(self._evict_when_idle())

    def _cancel_idle_timer(self) -> None:
        task, self._idle_task = self._idle_task, None
        self._idle_deadline = 0.0
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _evict_when_idle(self) -> None:
        """Give the accelerator back after a while of nobody asking anything.

        The handle is dropped **before** the eviction rather than after. `evict` calls
        `_kill`, which cancels the armed timer — and this *is* the armed timer, so
        cancelling it here would deliver a `CancelledError` into the middle of the process
        teardown and leave a half-killed child. Clearing the handle first, and never
        cancelling the running task, are two guards against the same mistake.
        """
        try:
            await asyncio.sleep(IDLE_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            return
        self._idle_task = None
        logger.info("evicting the idle resident worker")
        await self.evict()


def _signal_group(process: asyncio.subprocess.Process, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(process.pid), sig)
