# ADR-0026: A resident inference worker beside the job queue

**Status:** Accepted (2026-08-08)

Extends **ADR-0009** (subprocess per job, single FIFO queue). It reverses nothing: every job still
gets its own process and they still run one at a time. It adds one process that is not a job, and
most of this record is about keeping that from being a problem.

## Context

An inference run records per-image diagnostics for a bounded sample of what it scored — 64 images by
default, spread evenly across the run. For every other image, the answer to "show me what the
branches did here" is "that image was not one of the sixty-four", which makes the decomposition that
makes a two-branch method legible reachable only by luck.

Serving that on demand is a hundred milliseconds of work behind eleven seconds of setup. EfficientAD's
checkpoint is 75 MB and building the module touches the pretrained teacher; `pixel_reference` is
cheaper but still pays a process start and an import of numpy and Pillow. Two designs were on the
table:

**A job per request.** It reuses everything — the queue, the JSON-lines protocol, cancellation, the
log tee, the WebSocket fan-out — and costs nothing new architecturally. It also means every browse
click spawns a process, loads a checkpoint, answers, and throws the checkpoint away; and because the
queue is a single FIFO by design, a request made while a model is training waits for the training to
finish. Clicking through six images would cost six model loads.

**A resident worker.** The checkpoint is loaded once and stays. Requests are milliseconds after the
first. It also makes the API process the parent of a long-lived compute process that is in neither
the queue nor the `job` table — and on this machine, one that holds the MPS device.

The user chose the resident, knowing the cost. This record is about paying it honestly.

## Decision

**One resident inference worker, keyed by `(experiment_id, checkpoint generation)`, kept off the
device by a lock rather than by a check.**

- **It mirrors the queue's layering exactly**, so there is one shape to learn rather than two:
  `jobs/resident.py` is the manager in the API process, `jobs/inspector.py` is the entrypoint
  (`python -m … <experiment_id>`, as `worker.py` takes a job id), and `experiments/diagnose.py` is the
  work (as `experiments/infer.py` is). The process-teardown workaround `release_transport` is shared
  rather than copied.
- **Requests are not jobs.** No `job` row, no log file, no `JobKind` — and therefore **no migration**,
  because the CHECK constraint on `job.kind` is never touched. A browse request is not a unit of work
  anyone needs to cancel, resume, or read the history of; giving it a row would put a hundred
  meaningless entries in the run list that the Jobs & files tab exists to make readable.
- **Requests travel on stdin, one JSON line, `{"rid": n, "image_id": i}`.** This is the genuine
  extension to ADR-0009, whose protocol is one-way. **Responses keep the existing envelope**, parsed
  with the same `parse_line` — which already tolerates the arbitrary chatter torch and anomalib write
  to stdout, and that tolerance is worth more here than a smaller protocol would be.
- **One lock, not a check.** `request` and `evict` take the same `asyncio.Lock`, and `JobQueue` awaits
  an injected `before_spawn` hook immediately before it spawns a worker. A resident and a job worker
  therefore **cannot coexist** — not because anything tests for it, but because starting a job has to
  wait for the lock an in-flight request holds. The dependency is injected from `api/app.py` into
  both, never queue→resident — the injection discipline of handbook frontend.md applied to a
  second thing.
- **A hook that fails fails the job.** "Freeing the device did not work, so we started training on
  top of whatever was holding it" is precisely the state the hook exists to prevent, so the job is
  finished as failed rather than spawned anyway.
- **A request arriving while a job runs is refused with 409, naming the job.** Queuing it behind a
  two-hour train would make a button that sometimes takes two hours. The refusal reads the running
  `job` row *and* the queue's claim on the job it is executing: the row is the durable answer, and the
  claim covers the window between the pre-spawn hook finishing and the row being written — a window
  in which a request would otherwise see an idle queue and start a resident against a worker that was
  already launching.
- **Keyed by a generation fingerprint** over the model directory's names, sizes and modification
  times, computed in the API process and compared on every request. A retrain changes it and the
  resident is replaced, so **serving from stale weights is impossible by construction** rather than by
  an eviction hook firing in time. Sizes and mtimes rather than content: this is on the request path,
  and the cost of a false positive is one respawn.
- **`InferContext.maps_subdir`**, defaulting to `maps` and set to `scratch-maps` here, then removed.
  `predict` writes an anomaly map unconditionally; without this, a browse request would overwrite
  `maps/{image_id}.npy` under a `range.json` fitted by a different run, leaving that image's map a
  generation ahead of its own score row with nothing saying so.
- **A request changes no score, no map and no metric.** Those come from a job and stay the run's
  (ADR-0011). What persists is the diagnostics, marked `on_demand` in the index (handbook diagnostics.md).
- **Any deviation kills the process.** A timeout, a broken pipe, a mismatched `rid`, an unexpected
  frame: every one of these is a state in which the next answer might belong to a different question,
  and respawning costs one model load. There is no restart policy and no supervision — a crash is
  normal, and the next request spawns another.
- **Idle eviction after ten minutes**, and teardown with the application's lifespan, before the
  queue's: the sidecar is itself a child of the desktop shell, so an orphan here is an orphan there.
- **`GET /api/health` reports it** — which experiment, which generation, how long until eviction, how
  many requests served. This is the only place the one invisible process in the system becomes
  visible, and it is a lock-free field read, because a health check that can block behind a model load
  is not a health check.
- **A request that cannot succeed fails as a request**: 422 when the method records no diagnostics or
  nothing is trained, 404 when the image is not in this experiment's split. All of these are cheap and
  need no torch, which is the point — the process that would otherwise discover them costs a model
  load to start. 503 is reserved for the resident actually failing, and carries its stderr tail,
  because the reason is nearly always a library's.

**Ruled out:** a job per request (a model load per click, and queued behind training); a resident per
experiment (N loaded checkpoints, N times the device); a thread inside the API process (the process
boundary is what makes a wedged native call survivable, and it is the thing ADR-0009 bought); and
guarding coexistence with a status check rather than a lock (a check taken in FastAPI's threadpool
against state mutated on the event loop is a race, and the race it loses is an out-of-memory error in
an unrelated training job).

## Consequences

Browsing a scored set, any image can be asked about, and after the first it is immediate.

Negative consequences, accepted honestly:

- **The API process now owns long-lived compute state.** ADR-0009 already lists "two sources of truth
  for job state" as a cost; this is a third thing, in neither the queue nor the `job` table. The lock
  makes coexistence structurally impossible, which is stronger than a check, but it is still not the
  process boundary the queue has.
- **A job can be delayed by an in-flight request** — up to the 120-second request ceiling, and in
  practice the seconds a model load takes. That delay *is* the guarantee, so it must not be optimised
  away by making the hook non-blocking.
- **The resident is unbounded in memory in one direction**: it holds whatever the method's checkpoint
  costs, for ten minutes after the last question. On a machine also running a browser and a Vite dev
  server that is real memory, and nothing warns.
- **Its failures are asynchronous to the user's click.** A 503 with a stderr tail is the best this can
  do; the underlying cause is in a process the reader never asked to start.
- **The protocol now has an inbound half that only one worker speaks.** `jobs/worker.py` ignores
  stdin, and nothing enforces that the two stay in step beyond both importing `REQUEST_ID` from
  `jobs/protocol.py`.
- **Eviction correctness rests on tests, not on types.** `tests/test_resident.py` covers the three
  cases that matter — a job start evicts, a hook that cannot evict fails the job, nothing outlives the
  application — and they are the only thing standing between this design and a mysterious OOM.

## Changelog

### 2026-08-12 — Generalised target, unchanged single-resident decision

MobileSAM contour assistance introduced a second interactive workload with the same expensive-load,
short-request shape. The resident is now keyed by `(kind, target key, artifact generation)` and may
hold either an experiment inspector or the promptable segmentation model. Switching targets replaces
the child process. There is still exactly one resident, the same lock still excludes every queued
job, requests still persist no job rows, and the asset catalogue — not the request — chooses the
checkpoint. This amends the target-specific wording; it does not change the decision or add another
device owner.
