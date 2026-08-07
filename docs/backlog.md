# Backlog

Working task list for **visual-anomaly-lab**, organised as epics keyed to the milestones in [roadmap.md](roadmap.md). The roadmap says *what* each milestone must achieve and how it is judged done; this file says *what to actually do next*.

**Sizes.** `S` ≈ half a day or less. `M` ≈ one focused day. `L` = multi-day, and should be split into subtasks before it is started rather than after. Sizes assume a solo developer working with Claude Code, and include the time to review generated code properly — a task is not done until its output has been read.

**Detail falls off with distance.** Tasks for M0–M3 are concrete and directly actionable. Tasks for M4–M7 are deliberately coarse: their real shape depends on what M2 and M3 teach us about the data, the job machinery, and how anomalib behaves on this machine. Estimating them precisely now would be false precision.

**Re-triage at the end of every milestone**: close what shipped, delete what stopped mattering, split any `L` that is next up, and promote anything the milestone revealed from *Later / ideas* into a real epic. Items are added to the bottom of the relevant epic, not slipped into the middle of one in progress.

---

## E1 — Repo & docs (M0)

- [x] Write hardened `.gitignore`: `privatedata/`, `*.bmp` / `*.BMP`, `data/`, `results/`, SQLite files, model artifacts, Python/Node/Rust/macOS noise (S) — **ADR-0001**
- [x] Write `scripts/check-repo-safety.sh`: fail on tracked or staged private paths and image extensions; usable as a pre-push check (S)
- [x] Write `README.md` stub: one-liner, private-data warning, links into `docs/` (S)
- [x] Write `docs/system-design.md`: architecture, process boundaries, domain model, API surface, directory layout, artifact conventions (M)
- [x] Write `docs/roadmap.md` and `docs/backlog.md` (S)
- [x] Write ADRs 0001–0011 (M)
- [x] Verify safety end-to-end, then first commit + push: `git check-ignore -v` on a deep real `privatedata/` BMP path, `git ls-files` free of images, `check-repo-safety.sh` green, docs cross-consistent (S)
- [x] `CLAUDE.md` + full user-facing `README.md`; genericize all committed docs (showcase dataset, no product identity); untrack the original brief (S)

## E2 — Walking skeleton (M1)

- [x] Backend scaffold: `uv` project on Python 3.12, package layout, FastAPI app, `GET /api/health` (version, schema version, DB path), settings module, ruff + mypy + pytest wired (S) — **ADR-0003**
- [x] SQLite migration runner (forward-only numbered SQL files, `PRAGMA user_version`, applied at startup) + schema v1 covering **all** ADR-0005 entities: Dataset, Channel, Sample, Image, Mask, Split, SplitAssignment, Experiment, Job, ImageResult, SampleResult, MetricSet — with indices and foreign keys (M) — **ADR-0004**, **ADR-0005**
- [x] Frontend scaffold: Vite + React + TypeScript, router, app shell/layout, API client generated from the OpenAPI schema with a single configurable base URL (S) — **ADR-0012**
- [x] Tauri shell: spawn the Python sidecar, read the port it bound back from it, hand it to the frontend, terminate the child on window close *and* on crash; verify no orphan process survives (M) — **ADR-0003**
- [x] WebSocket echo endpoint + frontend hook, proven end-to-end in both the desktop app and a plain browser (S)
- [x] Dev scripts (backend only, frontend only, full desktop) + a README dev section documenting the browser-against-`uv run` workflow (S)
- [x] Guardrails: versioned `.githooks/pre-commit` running the private-data guard (closing ADR-0001's "advisory unless invoked" gap), and CI running the guard, ruff, mypy, pytest, tsc, vitest, the frontend build, `cargo fmt`/`clippy`, and an API-contract check that regenerates the TypeScript client and fails on a diff (S)

## E3 — Dataset & import (M2)

- [x] Manifest schema (JSON): datasets, proposed samples, images, per-image `sha256`, detected channels, warnings — versioned, reviewable before commit (M) — **ADR-0006**
- [x] `channel_folders` adapter: classifies path *components* against label and channel vocabularies in any nesting order, with token-level channel matching so a channel fused into a group folder's name still resolves; every vocabulary is an option carried in the manifest (M) — **ADR-0006**, **ADR-0013**
- [x] Scan + commit endpoints: `scan` runs as a Job and produces a manifest without writing to the DB; `commit` is one synchronous idempotent transaction, because the walk and the hash are the expensive part and the inserts take milliseconds (M) — **ADR-0009**, **ADR-0013**
- [x] Import review UI: dataset summary, editable channel mapping, per-group label correction, and a **warnings panel** — variable channel count, unassigned channels, duplicate hashes, unreadable and empty files — with commit blocked until warnings are acknowledged (M)
- [x] Dataset browser: virtualized thumbnail grid, filter by label / channel / split membership, sample count header (M)
- [x] Grouped sample viewer: channel tabs (count driven by the data), synchronized zoom/pan across channels, side-by-side comparison, keyboard labelling, metadata panel (M)
- [x] Split creation: seeded RNG, **sample-level** assignment, normal-only train strategy, stratified by capture group, val/test containing normals and defects, persisted with its seed *and its fractions* so it is reproducible (M) — **ADR-0011**
- [x] Dataset verify operation: re-check every recorded path exists and its `sha256` matches; report drift and never repair it (S) — **ADR-0013**

## E4 — Media (M2)

- [x] Thumbnail + preview cache under `data/`, keyed by image id because imported files are immutable, with an `ETag` derived from `sha256` + tier and immutable cache headers (M)
- [x] Post-import pre-warm job that generates thumbnails for a whole dataset with progress (S)
- [x] Full-resolution lossless PNG endpoint for the sample viewer, rendered on demand and deliberately not cached — a cached full tier costs most of a gigabyte per dataset (S)

## E5 — Model interface & jobs (M3) — done

*The job machinery landed in M2, which needed it for the import scan and the thumbnail
pre-warm. The model side landed in M3, and cost exactly what ADR-0009 predicted: one
registry entry and one handler function per new job kind, with no change inside the
queue, the protocol, cancellation or the fan-out.*

- [x] `models/base.py`: `AnomalyModel` interface (`fit`, `predict`, `save`, `load`, config JSON Schema, capability flags) + a lazy-loading registry (S) — **ADR-0007**
- [x] Subprocess worker + JSON-lines event protocol on stdout (`progress`, `log`, `metric`, `done`, `error`), with a parent-side parser that tolerates interleaved library output (L) — **ADR-0009** *(M2)*
- [x] FIFO job queue + Job table persistence + crash recovery on startup (any Job left `running` → `failed`, log preserved) (M) — **ADR-0009** *(M2)*
- [x] WebSocket fan-out of job events to the UI + per-job log files (M) *(M2)*
- [x] Cancellation: SIGTERM → grace period → SIGKILL, with the Job ending in `cancelled` and no orphan children (S) *(M2)*
- [x] `train` and `infer` job handlers (S) — **ADR-0009**
- [x] Preprocessing bridge: one config on the experiment, one loader every method uses, so a comparison is not partly measuring a resize (M)
- [x] Diagnostics contract: `produces_diagnostics` + `emit_diagnostic`, float32 `.npy` behind a self-describing index, scalar series reusing the existing `metric` event (M) — **ADR-0018**
- [x] Device resolution that never raises and always records the reason it chose what it chose (S)

## E6 — Classical baseline (M8, optional)

*Moved off the critical path by **ADR-0015**: making the showcase-specific method the
first one would have proved the architecture works for exactly one dataset. `pixel_reference`
took its place in the slice and is the geometry-free core of the same algorithm, so what
remains here is a circle-fit front end onto a component that already exists and is tested.*

- [ ] Circle detection: Hough seed → radial-ray subpixel edge sampling → robust circle fit (RANSAC + Taubin), with a per-dataset median-prior fallback when the fit is unreliable (M) — **ADR-0010**
- [ ] Polar transform + orientation estimation by FFT angular correlation, with reference bootstrap (first pass builds the reference from mutually aligned training samples) (M) — **ADR-0010**
- [ ] Per-channel robust reference build: median + MAD over aligned polar training images (S)
- [ ] Predict path: z-map (deviation / MAD) → Gaussian smoothing → inverse-polar warp back to image space → high-percentile image score (M)
- [ ] Parameter defaults sweep on `set1` (ray count, smoothing sigma, score percentile, polar resolution); record chosen defaults and the numbers that justified them (M)
- [ ] Unit tests on synthetic discs: known centre/radius/rotation recovered within tolerance; injected blob raises the score (S)

## E7 — Evaluation (M3) — done

- [x] Channel → sample score aggregation (`max` and `mean`, configurable per experiment) (S) — **ADR-0011**
- [x] Sample-level and image-level ROC-AUC and average precision, tie-aware, returning `None` rather than a fabricated number when a subset has one class (S)
- [x] **Pixel-level ROC-AUC and AU-PRO**, streamed through fixed-bin histograms so memory is constant in the number of test images (M) — **ADR-0017**
- [x] On-demand threshold endpoint returning the confusion matrix *and* the classified rows, so the threshold rule lives in one language (M)
- [x] Ranked lists: most-anomalous and most-normal samples with scores (S)
- [x] Timing statistics: per-sample inference time, mean/median/p95, recorded in the MetricSet (S)
- [ ] ROC and PR *curve* endpoints for the M4 benchmark charts — the arrays exist, nothing serves them yet (S)

## E8 — Experiment UI (M3) — done

- [x] Experiment create screen: dataset + split pickers, method picker, config form generated from the plugin's JSON Schema, preprocessing and evaluation sections (M) — **ADR-0007**
- [x] Progress + logs screen: live WebSocket progress bar, streaming log view, cancel button, terminal states — reusing `JobProgress` and `useJob` unchanged (M)
- [x] Results screen: threshold slider driving the confusion matrix, TP/FP/TN/FN filtering, ranked lists (L)
- [x] Sample result viewer: anomaly-map overlay with an opacity slider, ground-truth outline where a mask exists (M)
- [x] Experiment list with status, headline metric, and reopen-from-persistence (S)

## E9 — Anomalib integration (M3 done; PatchCore at M7)

- [x] Add anomalib as an optional dependency group + standalone MPS smoke test script — **run before writing wrapper code** (S) — **ADR-0008**. It earned itself immediately: the penalty batch turned out to be mandatory despite defaulting to `None`.
- [x] `efficientad_anomalib`: `fit` / `predict` / `save` / `load`, step progress and per-branch losses onto the job event protocol, forward-hook diagnostics, explicit downloads (L)
- [x] Preprocessing config bridge — see E5 (M)
- [ ] `patchcore_anomalib` wrapper + memory sizing (coreset ratio, memory-bank footprint, documented limits) (M) — **M7**
- [ ] Revisit the training loop against anomalib's Lightning path when their datamodule stops reaching into `trainer.datamodule`; ours exists only because that coupling would cost the preprocessing bridge (S)

## E10 — Comparison UI (M5)

- [ ] Multi-experiment comparison: metric table (ROC-AUC, confusion at a chosen threshold, timing) + overlay A/B view of two methods' anomaly maps on the same sample (M)

## E11 — Custom EfficientAD (M6)

- [ ] Reimplement EfficientAD from arXiv:2303.14535 as `efficientad_custom` behind the unchanged plugin interface: PDN student/teacher distillation, autoencoder branch, quantile-based score normalization, MPS training loop (L)
  - split when scheduled: PDN backbone → teacher pretraining/distillation → student loss → autoencoder branch → quantile normalization → training loop + progress events → comparison run against `efficientad_anomalib`

## E12 — README & polish (M9)

- [ ] Full README: setup from a fresh machine, dataset conventions, architecture overview, troubleshooting (M)
- [ ] "How to add a new anomaly-detection method" guide, written against the real interface with a worked example (S) — **ADR-0007**
- [ ] Docs refresh (system design + ADR amendments where implementation diverged) and backlog re-triage (S)

---

## Later / ideas

Not scheduled. Revisit at each milestone re-triage; promote into an epic when there is a concrete reason to.

- Per-channel score quantile normalization before aggregation, so one illumination channel cannot dominate the sample score by scale alone.
- Explicit detector for the part's asymmetric surface features as an orientation fallback when FFT angular correlation is ambiguous (near-rotationally-symmetric parts).
- Per-set classical references (separate `set1` / `set2` references) if lighting or fixturing drift between sets turns out to matter.
- Uninformed Students (Student-Teacher Anomaly Detection, [papers.md](papers.md) #4) as a fourth method.
- Export / report generation: experiment results and comparison tables to PDF or HTML for sharing without the app.
- **`mask.sha256`, as a numbered migration.** `verify` can check that a mask file is still there and not that it is still the same file, so a mask re-exported in place silently changes a pixel metric. Worth doing before pixel numbers are relied on (ADR-0016, ADR-0017).
- **Warn when an imported split's train subset contains defects.** The `imported` strategy trusts the source completely, which is the point, but a bad benchmark file currently produces a bad experiment quietly. The training handler already excludes them and logs it; the split screen says nothing.
- Per-image inference batching for the deep methods — currently one image per forward pass, which is simple and leaves throughput on the table.
- Show the pixel-metric protocol on the results screen. "Normal images count, with an empty mask" moves the number substantially and is documented nowhere the reader will look.
