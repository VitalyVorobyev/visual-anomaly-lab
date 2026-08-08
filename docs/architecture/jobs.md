# Async job system

Training and inference are long-running and must be asynchronous from the UI's perspective, with progress,
logs, completion and failure states. The execution model is **one subprocess per job** (ADR-0009).

## Why a subprocess

- **PyTorch + MPS are fork-unsafe.** Running training in a thread inside the API process, or forking it, risks
  deadlocks and driver-state corruption. A clean `spawn`ed process avoids the whole class of problem.
- **Crash and OOM isolation.** A segfault or an out-of-memory kill in a training run takes down the worker,
  not the API — the UI stays responsive and reports a failed job with its log.
- **Memory is reclaimed on exit.** Model weights, memory banks and MPS allocations disappear when the process
  ends; a long-lived server would otherwise accumulate them across experiments.
- **Cancellation is honest.** Cooperative cancellation is tried first via `should_cancel()`; if the worker does
  not stop, the parent escalates **`SIGTERM` → grace period → `SIGKILL`**. A thread offers no equivalent
  guarantee, and "cancel" that does not actually stop the work is worse than no cancel button.

## Queue

A **single-job FIFO queue** lives in the API process, mirrored into the `Job` table. One machine, one user, one
MPS device — concurrency would only cause contention and confusing timing measurements. Queued jobs are
visible and cancellable before they start. The queue is intentionally in-process: no Celery, no Redis, no
broker (ADR-0009).

## Worker → parent event protocol

The worker communicates with its parent over **JSON-lines on stdout** — one JSON object per line, flushed:

```json
{"ev":"progress","fraction":0.42,"message":"epoch 8/20"}
{"ev":"log","level":"info","message":"device=mps batch=8"}
{"ev":"metric","name":"train_loss","step":800,"value":0.0137}
{"ev":"done","result":{"images":189,"artifact_dir":"data/artifacts/exp-12"}}
{"ev":"error","type":"RuntimeError","message":"...","traceback":"..."}
```

Line-delimited JSON is chosen because it is trivial to produce, trivially parsed incrementally, human-readable
in a log file, and needs no shared-memory or socket setup between the two processes.

The parent does three things with each line:

1. **persists** `progress` / `message` / terminal state to the `Job` row (so a REST poll is always accurate);
2. **tees the full stream** to the job's `log_path` — including any non-JSON output such as third-party
   library chatter or a native crash message, which is exactly what is needed for post-mortem. A job bound to
   an experiment logs to `data/artifacts/exp-<id>/logs/<job>.log`; one that is not — an import job — logs to
   `data/jobs/logs/<job_id>.log`;
3. **fans out** to subscribers of `WS /ws/jobs/{id}`.

## Frontend reconnection

WebSockets drop — on sleep, on network stack changes, on reload. The client rule is **snapshot then
subscribe**: `GET /api/jobs/{id}` for current status, progress and recent log tail, *then* open the WebSocket
for the live stream. This makes reconnection, first load and late-joining a running job all the same code
path, with no missed-event reconciliation logic.

## Scalar series, after the fact

`metric` events are streamed and tee'd, and stored in no column. That is enough for a live chart and not
enough for one that survives a reload: `log_tail` is 200 raw lines of a stream that also carries progress
frames and library chatter, which for a 20 000-step training run is its final seconds.

`GET /api/jobs/{id}/metrics` therefore **parses the job's own log file** and returns the named series it
finds, downsampled to a drawable number of points with the drop reported. No table, no migration, and no
second event channel — the log is already the durable copy of the stream. The client takes this snapshot
when it opens the socket and appends live frames to it, under the same freeze rule the console follows: the
snapshot cannot be re-read live, because every event invalidates it and it comes back already containing the
points the socket just delivered (**ADR-0020**).

## Import jobs

Import scanning uses the **same machinery** (`kind = "import"`). Hashing several hundred ~4 MB BMPs is slow
enough to need progress reporting, and reusing the job system means no second progress mechanism exists.

## Continuing a run

Pressing **Train** on a trained experiment used to throw the model away and start over, which made the
workbench a thing you take readings with rather than one you tune in. Resume is a **declared
capability** — `Capabilities.supports_resume` plus a `runtime_checkable` `SupportsResume` protocol
carrying `completed_steps()` and `fit_more(train, ctx, *, additional_steps)`. A protocol rather than
two more abstract methods, so `pixel_reference` — which has no notion of a step — grows no stub. The
train handler checks the flag *and* the protocol against each other and names a disagreement as a
plugin bug (**ADR-0025**).

- **`TrainParams.additional_steps`, so the experiment's config is untouched.** How long to continue
  for is a property of *this run*; the frozen record that makes an experiment reproducible stays
  frozen.
- **The checkpoint carries optimizer moments, LR-scheduler state, the absolute step counter and both
  RNG streams.** A weights-only warm start restarts Adam's moments — most of what a long run has
  learned about its own gradients — and produces a visible loss spike; it is honest but it is
  measurably not a continuation. There is **no option to skip** the extra state, because an option
  would make "can I continue this run?" depend on a flag chosen before anyone knew the answer. On the
  reference configuration this takes a checkpoint from 32 MB to 75 MB.
- **Steps reported to `ctx.metric` are absolute across an experiment's training**, so a continued
  run's chart is a continuation of the first with no stitching.
- **`model/train_state.json`** carries `completed_steps`, `runs` and `last_run_steps`, written by the
  *handler*. The API process has no torch by design and cannot open a `.pt`; without the sidecar the
  configuration panel would show `max_steps: 4000` beside an 8000-step model.
- **A continuation that cannot succeed is refused as a form, with 422** — the method cannot resume,
  nothing is trained, or the checkpoint predates the format.

**What "exact" means, precisely, because the loose version is false.** Continuing through a save and a
load is bit-identical to continuing without ever leaving the process, pinned at `rtol=0, atol=0`.
It is **not** true that 10 + 10 equals 20: `max_steps` is a per-run budget, so a run of 10 sizes its
own `StepLR` for 10 and completes its tenfold decay inside those steps, and the continuation then
resizes the schedule to the new total. Both facts are deliberate and both are printed before the run
starts.

That schedule arithmetic is the one place this has already been wrong. `StepLR.get_lr` is
*multiplicative on the param group's current rate*, and `Adam.load_state_dict` restores the rate the
previous leg ended on — always the decayed one, since every leg anneals over its own last 5%. Left
alone, each continuation started a tenth low and dropped again: 1e-5 instead of 1e-4 on the first
resume, 1e-9 by the fifth. `_build_scheduler` now **computes** the rate from the schedule's closed
form at the resume point rather than inheriting it, in both EfficientAD plugins, pinned by two tests.

Outside the checkpoint, and said on screen rather than only here: **the ImageNette penalty-set
iterator restarts** in the anomalib wrapper. `efficientad_custom` resumes it.

## The one process that is not a job

Serving a per-image diagnostic on demand is a hundred milliseconds of work behind eleven seconds of
setup, so a job per request would mean a model load per click — and, because the queue is a single
FIFO by design, a request made during training would wait for the training. There is instead **one
resident inference worker**, keyed by `(experiment_id, checkpoint generation)` (**ADR-0026**).

It mirrors the queue's layering exactly, so there is one shape to learn rather than two:
`jobs/resident.py` is the manager, `jobs/inspector.py` is the entrypoint, `experiments/diagnose.py`
is the work — as `jobs/queue.py`, `jobs/worker.py` and `experiments/infer.py` are.

- **Requests are not jobs.** No `job` row, no log file, no `JobKind`, and therefore no migration. A
  browse click is not a unit of work anyone needs to cancel or resume.
- **Requests travel on stdin**, one JSON line, `{"rid": n, "image_id": i}` — the genuine extension to
  a protocol that is otherwise one-way. Responses keep the existing envelope and the same
  `parse_line`, whose tolerance for library chatter is worth more here than a tighter protocol.
- **One lock, not a check.** `request` and `evict` take the same `asyncio.Lock`, and the queue awaits
  an injected `before_spawn` hook immediately before spawning a worker. A resident and a job worker
  **cannot coexist** — not because anything tests for it, but because starting a job has to wait for
  the lock an in-flight request holds. **A job may therefore be delayed by one in-flight request, and
  that delay *is* the guarantee**: the hook must not be made non-blocking. The dependency is injected
  from `api/app.py` into both; the queue never imports the resident.
- **Keyed by a generation fingerprint** over the model directory's names, sizes and mtimes, compared
  on every request, so serving from stale weights is impossible by construction rather than by an
  eviction hook firing in time.
- **A request arriving while a job runs is refused with 409, naming the job.** Queuing it behind a
  two-hour train would make a button that sometimes takes two hours.
- **A request changes no score, no map and no metric.** `InferContext.maps_subdir` points its
  unconditional map write at `scratch-maps`, which is then removed; without it a browse request would
  overwrite an image's map under a range fitted by a different run.
- **Any deviation kills the process** — a timeout, a broken pipe, a mismatched `rid`. Each is a state
  in which the next answer might belong to a different question, and respawning costs one model load.
- **Idle eviction after ten minutes**, torn down with the application's lifespan before the queue's.
  `GET /api/health` reports which experiment, which generation, time to eviction and requests served
  — the only place the one invisible process in the system becomes visible, and a lock-free field
  read, because a health check that can block behind a model load is not a health check.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
