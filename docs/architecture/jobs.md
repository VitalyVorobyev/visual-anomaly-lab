# Async job system

Training, inference, import and every other long operation run as asynchronous jobs with progress, logs,
completion and failure states. The execution model is **one subprocess per job** (ADR-0009).

## Subprocess rationale

- **PyTorch + MPS are fork-unsafe.** A thread in the API process, or a fork, risks deadlocks and driver-state
  corruption; a clean `spawn`ed process avoids the class of problem.
- **Crash and OOM isolation.** A segfault or out-of-memory kill takes down the worker, not the API; the UI
  reports a failed job with its log.
- **Memory is reclaimed on exit.** Weights, memory banks and MPS allocations disappear with the process.
- **Cancellation is honest.** Cooperative cancellation via `should_cancel()` first; if the worker does not
  stop, **`SIGTERM` → grace period → `SIGKILL`**.

## Queue

A **single-job FIFO queue** lives in the API process, mirrored into the `Job` table — one machine, one user,
one MPS device, and concurrency would only distort timing. Queued jobs are visible and cancellable before
they start. No Celery, no Redis, no broker (ADR-0009).

**A new kind costs one `JobKind` member, one entry in `jobs/handlers.py` and its handler**, and nothing in
the schema ([domain model](domain-model.md)); the queue, protocol, cancellation, log tee and WebSocket
fan-out are kind-agnostic.

**Deletion takes the queue's lifecycle guard.** `enqueue` takes the same cross-thread lock, so a delete
cannot observe idle state and race a new job into existence. Queued or running work blocks deletion and is
named in the preview. Dataset lookup covers both experiment-bound jobs and dataset ids in kind-agnostic job
parameters.

## Job kinds

- **`import`, `verify`, `prewarm`** — scanning and hashing is slow enough to need progress, so import reuses
  this machinery rather than growing a second progress mechanism ([import](import.md), [media](media.md)).
- **`reference_import`** — one worker scans every missing dataset in the selected public packs and
  atomically commits the manifests. The worker receives the app-data and reference-data directories
  explicitly, so discovery in the API and execution agree in a packaged build.
- **`model_asset_download`** — the handler knows only an asset key; the fixed catalogue supplies the URL,
  byte count and SHA-256, so the sidecar cannot become an arbitrary downloader. The licence is accepted
  before enqueue. Bytes go to a job-specific partial file and reach the managed path only after size and
  digest verification; cancellation or failure removes the partial.
- **`region_prepare`** — two modes. Preview selects at most 24 evenly spaced images and returns transforms
  without writing pixels. Build visits the whole dataset, checks cancellation between images, writes into
  managed staging and publishes atomically once its manifest and summary are complete. Per-image failures
  are recorded and reduce coverage; they never fall back to identity. A malformed job, missing asset or
  extractor construction failure fails the job before processing.
- **`distill`** — produces a teacher asset, not an experiment, so its `experiment_id` is null; it is started
  from the command line ([methods](methods.md)).
- **`train`, `infer`, `export`** — experiment-bound. `export` writes an ONNX bundle into staging,
  runs graph parity on CPU, hashes the payloads and publishes atomically ([deployment](deployment.md)).

**A job may name its successor.** A `done` result carrying `follow_up: {"kind", "params"}`
(`jobs/protocol.FOLLOW_UP_KEY`) is queued by `JobQueue._finish` only when the job `succeeded`, bound to the
same experiment. The handler decides whether; the queue decides when, without knowing either kind. `train`
with `then_score` uses it to queue an `infer` of the default subsets ("Train & score"). A malformed
follow-up, or one whose experiment was deleted meanwhile, is logged and dropped.

## Worker → parent event protocol

The worker talks to its parent in **JSON lines on stdout**, one flushed object per line:

```json
{"ev":"progress","fraction":0.42,"message":"epoch 8/20"}
{"ev":"log","level":"info","message":"device=mps batch=8"}
{"ev":"metric","name":"train_loss","step":800,"value":0.0137}
{"ev":"done","result":{"images":189,"artifact_dir":"data/artifacts/exp-12"}}
{"ev":"error","type":"RuntimeError","message":"...","traceback":"..."}
```

The parent, for each line:

1. **persists** `progress` / `message` / terminal state to the `Job` row, so a REST poll is accurate;
2. **tees the full stream**, non-JSON output included, to the job's `log_path` —
   `data/artifacts/exp-<id>/logs/<job>.log` for an experiment-bound job, `data/jobs/logs/<job>.log`
   otherwise;
3. **fans out** to subscribers of `WS /ws/jobs/{id}`.

### Framing

`readline` is not used: it raises past asyncio's 64 KiB limit, and a tqdm progress bar is one line (frames
separated by `\r`). Output is read in chunks and split by `split_output`, which flushes an unterminated
fragment rather than buffering without bound. The budget depends on what the fragment could be: chatter
can be cut anywhere (`MAX_LINE_BYTES`, 16 KiB), while a fragment beginning with `{` may be a protocol event,
and half an event is no event (`MAX_EVENT_BYTES`, 8 MiB). Both are finite, so no output can stop the queue.

**A result travels as one line.** A handler tested in-process bypasses the pipe, so it proves nothing about
whether its result reaches the parent; a result whose size grows with the dataset is a finding about the
result.

## Frontend reconnection

The client rule is **snapshot then subscribe**: `GET /api/jobs/{id}` for status, progress and recent log
tail, *then* the WebSocket for the live stream. First load, late-joining and reconnection are one code path
with no missed-event reconciliation. A dropped socket reopens on that path after a short delay, unless the
stream ended with an `end` frame.

**Progress belongs to the job row, and only there.** A `progress` frame invalidates `["jobs", id]`
*exactly* — not the experiment (four frames a second would be a poll), and not the key's children, since
`["jobs", id, "metrics"]` is built by parsing the whole log file. Anything drawing live progress reads the
job row; the copy inside the experiment detail payload refreshes only on window focus and terminal frames.

## Scalar series

`metric` events are streamed and tee'd but stored in no column. `GET /api/jobs/{id}/metrics` **parses the
job's own log file** and returns the named series, downsampled to a drawable number of points with the drop
reported — the log is already the durable copy of the stream. The client takes this snapshot when it opens
the socket and appends live frames to it; the snapshot is not re-read live.

## Resume

Continuing training is a **declared capability** — `Capabilities.supports_resume` plus a
`runtime_checkable` `SupportsResume` protocol with `completed_steps()` and
`fit_more(train, ctx, *, additional_steps)` — so a method with no notion of a step grows no stub. The train
handler checks the flag against the protocol and names a disagreement as a plugin bug.
`efficientad_custom`, `dinomaly_custom` and `glass_anomalib` support it.

- **`TrainParams.additional_steps`** carries the continuation length, so the experiment's frozen config is
  untouched.
- **The checkpoint carries optimizer moments, LR-scheduler state, the absolute step counter and its RNG
  streams**, with no option to skip them: a weights-only warm start restarts Adam's moments and is
  measurably not a continuation.
- **Steps reported to `ctx.metric` are absolute** across the experiment, so a continued run's chart needs no
  stitching.
- **`model/train_state.json`** (`completed_steps`, `runs`, `last_run_steps`) is written by the handler,
  because the torch-free API process cannot open a `.pt`.
- **A continuation that cannot succeed is refused with 422** — the method cannot resume, nothing is trained,
  or the checkpoint predates the format.

**Exactness.** Continuing through a save and a load is bit-identical to continuing in-process, pinned at
`rtol=0, atol=0`. It is **not** true that 10 + 10 steps equals 20: `max_steps` is a per-run budget, each leg
sizes its own schedule, and the continuation resizes it to the new total — both printed before the run.
`StepLR` is multiplicative on the group's current rate and `Adam.load_state_dict` restores the decayed one,
so `_build_scheduler` **computes** the resume rate from the schedule's closed form rather than inheriting
it.

## The resident worker

An interactive request is a hundred milliseconds of work behind seconds of setup, and the queue is a single
FIFO, so a job per click would mean a model load per click and a wait behind training. There is instead
**one resident compute worker**, keyed by `(kind, target key, artifact generation)` (ADR-0026), holding
one of three things at a time: an experiment inspector, MobileSAM, or a few-shot preview.

It mirrors the queue's layering: `jobs/resident.py` is the manager; `jobs/inspector.py`,
`jobs/segmenter.py` and `jobs/previewer.py` are thin entrypoints; `experiments/diagnose.py`,
`model_assets/mobile_sam.py` and `experiments/preview.py` do the work — as `jobs/queue.py`,
`jobs/worker.py` and a job handler do.

- **A few-shot preview** (`few_shot_preview`, ADR-0040) is one method at its defaults, fitted on the
  reference studio's current references through the same `PreparedClassTargets` a run uses. Its spec
  (dataset, class, method, profile, references) is the resident's command line, never a request field. Its
  generation fingerprints the spec, the pinned region build and every reference image's pinned truth, so
  new references are a new resident. A request segments one image into `previews/<generation>/maps/`,
  which `GET /api/studio/previews/{generation}/{image}.png` renders on the fixed range [0, 1]. A preview is
  stored for nobody and evaluated by nothing.

- **Requests are not jobs.** No `job` row, no log file, no `JobKind`, and therefore no migration. A
  browse click is not a unit of work anyone needs to cancel or resume.
- **Requests travel on stdin**, one JSON line, `{"rid": n, "image_id": i}` — the genuine extension to
  a protocol that is otherwise one-way. Responses keep the existing envelope and the same
  `parse_line`, whose tolerance for library chatter is worth more here than a tighter protocol.
- **One lock, not a check.** `request` and `evict` take the same `asyncio.Lock`, and the queue awaits
  an injected `before_spawn` hook immediately before spawning a worker, so a resident and a job worker
  **cannot coexist**: starting a job waits for the lock an in-flight request holds. **A job may
  therefore be delayed by one in-flight request, and that delay *is* the guarantee**: the hook must not
  be made non-blocking. The dependency is injected from `api/app.py` into both; the queue never imports
  the resident. **A `before_spawn` hook that fails fails the job**, which is not started.
- **Keyed by a generation fingerprint** over the experiment model directory or verified asset file,
  compared on every request, so serving from stale weights is impossible by construction. The
  fingerprint is over names, sizes and mtimes, not content: it is on the request path, and a false
  positive costs one respawn. Changing target kind also replaces the process.
- **A request arriving while a job runs is refused with 409, naming the job.** Queuing it behind a
  two-hour train would make a button that sometimes takes two hours. The check reads the running `job`
  row *and* the queue's claim, which covers the window between the pre-spawn hook and the row being
  written.
- **A request that cannot succeed fails as a request**, checked torch-free in the API process: 422 when
  the method records no diagnostics or is untrained, 404 when the image is outside the experiment's split
  or channel selection. 503 is reserved for the resident itself failing, and carries its stderr tail
  (`STDERR_TAIL_LINES`). A request times out after `REQUEST_TIMEOUT_SECONDS`.
- **`jobs/worker.py` ignores stdin.** The two sides agree only by both importing `REQUEST_ID` from
  `jobs/protocol.py`. `tests/test_resident.py` pins that a job start evicts, that a hook which cannot
  evict fails the job, and that nothing outlives the application.
- **A request changes no score, annotation, map or metric.** A MobileSAM answer is an ephemeral set of
  ranked cropped bitmap candidates; only explicit acceptance puts one into the annotation draft.
  `InferContext.maps_subdir` points diagnostic inference's unconditional map write at `scratch-maps`,
  which is then removed, so a browse request never overwrites a stored map.
- **Any deviation kills the process** — a timeout, a broken pipe, a mismatched `rid`. Each is a state
  in which the next answer might belong to a different question, and respawning costs one model load.
- **Idle eviction after ten minutes**, torn down with the application's lifespan before the queue's.
- **Destructive artifact work holds an eviction guard**, not merely a one-shot `evict`: the resident is
  killed and its lock remains held until the database row and app-owned artifact directory are gone. A
  diagnostic request therefore cannot respawn into the interval between those two operations.
- `GET /api/health` reports the resident kind, target key, generation, time to eviction and requests served
  as a lock-free field read, because a health check that can block behind a model load is not one.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
