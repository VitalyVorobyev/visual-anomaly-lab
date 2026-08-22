# ADR-0009: Job execution — subprocess per job, single FIFO queue

**Status:** Accepted (2026-08-06)

## Context

Training, inference, and import are long-running. The brief requires them to be asynchronous from
the UI's perspective and to expose progress, logs, completion, and failure states. In practice the
user also needs to cancel a run that is clearly going nowhere.

Running this work in the FastAPI process (ADR-0003) is tempting and wrong. PyTorch with MPS is not
fork-safe and interacts badly with an event loop; a training run that exhausts unified memory would
take the API server down with it; and Python does not reliably return large allocations to the OS,
so a long-lived server would accumulate memory across runs. Nor can a thread be cancelled: once
inside a C extension, cooperative cancellation is the only option, and a wedged thread stays wedged
for the life of the process.

## Decision

**Every job runs in its own spawned subprocess, drawn from a single FIFO queue, communicating by
JSON-lines events on stdout, fanned out to the UI over WebSocket.**

- **Process per job.** Each train / infer / import job is a freshly spawned worker process (spawn,
  not fork). Crashes, segfaults, and OOM kills are contained: the API server survives and marks the
  job failed. All memory is reclaimed at exit, so run *N+1* starts as clean as run 1.
- **Honest cancellation.** Cancel sends **SIGTERM**, letting the worker flush partial results and
  exit gracefully; after a grace period it sends **SIGKILL**. Because the process boundary is real,
  cancellation cannot be ignored — unlike the cooperative check in `TrainContext` (ADR-0007), which
  handles the graceful path.
- **Single-job FIFO queue.** One machine, one GPU: concurrent jobs would contend for MPS and unified
  memory and finish later than if serialized. Queue state is mirrored into the `Job` table so it
  survives inspection and restart. On backend startup, any job still marked *running* is
  reconciled to *failed* — its process died with the previous server.
- **JSON-lines events.** The worker writes one JSON object per line to stdout:
  `{"ev": "progress"|"log"|"metric"|"done"|"error", ...}`. The parent parses each line, persists what
  belongs in the database (ADR-0004), **tees the raw stream to a log file** under the experiment's
  artifact directory, and fans events out to subscribers of `/ws/jobs/{id}`.
- **Reconnect semantics.** The frontend fetches a REST snapshot of job state first, then subscribes
  to the WebSocket. A dropped connection, a reloaded window, or a UI opened mid-run all converge to
  the same view without a replay buffer in the server.

## Consequences

The API server stays responsive and alive no matter what a model does. Job logs are complete on disk
whether or not anyone was watching, which is what makes a failed run diagnosable after the fact.
Progress reporting needs no shared memory or IPC library — one text stream, trivially inspectable by
running the worker by hand. Serialization means resource behaviour is predictable and benchmark
timings are not polluted by contention.

Negative consequences, accepted honestly:

- **Process startup is expensive.** Every job pays interpreter startup plus multi-second PyTorch and
  anomalib imports before doing any work. For a short inference on a handful of images, that
  overhead dominates the actual computation.
- **Serialization is a real limit.** A queued job waits for a long training run to finish even when
  it is a CPU-only classical baseline (handbook methods.md) that would not have contended at all. There is no
  priority lane and no way to jump the queue.
- **stdout is a fragile channel.** Any library that prints to stdout — a progress bar, a warning,
  a `print` left in a model — corrupts the event stream. Workers must redirect stray output to
  stderr, and the parser must tolerate non-JSON lines rather than crash.
- **State can be lost at the boundary.** A SIGKILLed worker leaves whatever it had not yet emitted:
  partially written `.npy` files, result rows for some images and not others.
- **Two sources of truth for job state.** The in-memory queue and the `Job` table can diverge under
  crash or race; the startup reconciliation covers the common case but not all of them.
- **Debugging is harder.** A breakpoint in a model does not stop in the server; interactive
  debugging requires running the worker standalone.
