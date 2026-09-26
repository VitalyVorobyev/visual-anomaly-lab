# ADR-0026: A resident inference worker beside the job queue

**Status:** Accepted (2026-08-08)

## Context

Some interactive questions are cheap to answer and expensive to set up: "show me what the branches
did on this image" for a scored run, or a promptable segmentation model assisting a contour. The
model load dominates — seconds against a fraction of one. ADR-0009 gives every unit of work its own
process and runs them one at a time through a single FIFO queue.

## Decision

**Exactly one resident worker exists, keyed by `(kind, target key, artifact generation)`, and it is
kept off the device by a lock rather than by a check.**

- **One resident, whatever the target.** It holds either an experiment inspector or a promptable
  segmentation asset; switching targets replaces the process. There is never a second device owner.
- **One lock.** A request and `ResidentWorker.evict` take the same `asyncio.Lock`, and `JobQueue`
  awaits an injected `before_spawn` hook before it starts any worker. A resident and a job worker
  therefore cannot coexist: starting a job waits for an in-flight request. A hook that fails fails
  the job. The wiring lives in `api/app.py`; the queue never imports the resident.
- **A request arriving while a job runs is refused**, naming the job, rather than queued behind it.
- **Requests are not jobs.** No `job` row, no log, no `JobKind`. They travel as JSON lines on the
  worker's stdin and answers reuse the job protocol's envelope and parser.
- **The key includes a generation fingerprint** of the model directory or asset, so a retrain
  replaces the resident and stale weights cannot be served.
- **A request changes no score, map or metric**; those come from jobs. Only on-demand diagnostics
  persist.
- **Any protocol deviation kills the process**; there is no supervision, and the next request
  spawns another. It is evicted when idle and torn down with the application.

## Alternatives considered

- **A job per request.** Reuses the queue, its protocol, cancellation and log tee, and adds nothing
  architecturally; but every click pays a model load, and a request made while a model trains waits
  behind the training.
- **A resident per experiment.** N checkpoints and N claims on the device.
- **A thread inside the API process.** The process boundary is what makes a wedged native call
  survivable.
- **Guarding coexistence with a status check.** A check against state mutated on the event loop is
  a race, and losing it is an out-of-memory failure in an unrelated training job.

## Consequences

- Any image in a scored run can be asked about, and after the first question the answer is
  immediate.
- **The API process owns long-lived compute state** outside both the queue and the `job` table.
  The lock makes coexistence structurally impossible, but it is not the process boundary jobs have.
- **A job can be delayed by one in-flight request**, bounded by the request timeout. That delay is
  the guarantee, so the hook must never be made non-blocking.
- **The resident holds whatever its checkpoint costs until it is evicted**, and nothing warns about
  memory pressure from it.
- **Its failures are asynchronous to the user's click**; a stderr tail in the error is the best
  explanation available.
- **The protocol has an inbound half only one worker speaks**, and eviction correctness rests on
  tests rather than types.
