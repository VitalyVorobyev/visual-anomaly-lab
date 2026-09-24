# ADR-0009: Job execution — subprocess per job, single FIFO queue

**Status:** Accepted (2026-08-06)

## Context

Training, inference and import are long-running. They must be asynchronous from the UI's point of
view, expose progress, logs, completion and failure, and be cancellable when a run is clearly going
nowhere.

Running this work inside the API process (see ADR-0003) — on a thread or a task — is tempting and
wrong. PyTorch with MPS is not fork-safe and interacts badly with an event loop; a run that
exhausts unified memory takes the server down with it; Python does not reliably return large
allocations to the OS, so a long-lived server accumulates memory across runs; and a thread inside a
C extension cannot be cancelled. A worker pool with concurrent jobs was the other alternative.

## Decision

**Every job runs in its own spawned subprocess, drawn from a single FIFO queue, reporting by
JSON-lines events on stdout that are fanned out to the UI over WebSocket.**

- **Process per job**, spawned rather than forked. Crashes, segfaults and OOM kills are contained:
  the server survives and marks the job failed, and all memory is reclaimed at exit.
- **Honest cancellation.** Cancel sends SIGTERM to the worker's process group so it can exit
  gracefully, then SIGKILL after a grace period. The cooperative check in the plugin contexts (see
  ADR-0007) is the polite path; the process boundary is the guarantee.
- **One job at a time.** One machine, one accelerator: concurrent jobs contend for unified memory
  and finish later than if serialized. Queue state is mirrored into the `Job` table, and at startup
  any job still marked running is reconciled to failed.
- **JSON-lines events** — progress, log, metric, done, error — one object per line. The parent
  persists what belongs in the database, tees the raw stream to a log file, and fans events out to
  subscribers. A new job kind is one handler; the queue and the protocol are kind-agnostic.
- **Snapshot, then subscribe.** The frontend reads job state over REST and then subscribes, so a
  dropped connection, a reload or a window opened mid-run converge on the same view without a
  replay buffer.

The one sanctioned exception is the resident inference worker, which serves interactive requests
and yields to the queue (see ADR-0026).

## Consequences

The server stays alive whatever a model does. Logs are complete on disk whether or not anyone was
watching, which is what makes a failed run diagnosable. Progress needs no IPC library, and a worker
can be run by hand and read. Serialized jobs keep timings free of contention.

- **Start-up is expensive.** Every job pays interpreter start and multi-second PyTorch imports; for a
  short inference that overhead dominates.
- **No priority lane.** A CPU-only job waits behind a long training run it would not have contended
  with.
- **stdout is a fragile channel.** Any library that prints corrupts the stream, so workers redirect
  stray output and the parser tolerates non-JSON lines. A result that grows with the dataset is a
  finding about the result, because one line is bounded.
- **State can be lost at the boundary.** A killed worker leaves whatever it had not emitted:
  partial files, result rows for some images and not others.
- **Two sources of job state.** The in-memory queue and the table can diverge under a crash or a
  race; startup reconciliation covers the common case, not all of them.
- **Debugging is harder.** A breakpoint in a model does not stop in the server; interactive
  debugging means running the worker standalone.
