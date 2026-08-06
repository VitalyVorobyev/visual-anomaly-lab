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

## E5 — Model interface & jobs (M3)

*The job machinery landed in M2, which needed it for the import scan and the thumbnail
pre-warm. What remains here is the model side.*

- [ ] `models/base.py`: `ModelPlugin` interface (`fit`, `predict`, `save`, `load`, config JSON Schema, capability flags) + registry keyed by `classical_circular`, `efficientad_anomalib`, `patchcore_anomalib`, `efficientad_custom` (S) — **ADR-0007**
- [x] Subprocess worker + JSON-lines event protocol on stdout (`progress`, `log`, `metric`, `done`, `error`), with a parent-side parser that tolerates interleaved library output (L) — **ADR-0009** *(M2)*
- [x] FIFO job queue + Job table persistence + crash recovery on startup (any Job left `running` → `failed`, log preserved) (M) — **ADR-0009** *(M2)*
- [x] WebSocket fan-out of job events to the UI + per-job log files (M) *(M2)*
- [x] Cancellation: SIGTERM → grace period → SIGKILL, with the Job ending in `cancelled` and no orphan children (S) *(M2)*
- [ ] `train` and `infer` job handlers, once there is a model to run (S) — **ADR-0009**

## E6 — Classical baseline (M3)

- [ ] Circle detection: Hough seed → radial-ray subpixel edge sampling → robust circle fit (RANSAC + Taubin), with a per-dataset median-prior fallback when the fit is unreliable (M) — **ADR-0010**
- [ ] Polar transform + orientation estimation by FFT angular correlation, with reference bootstrap (first pass builds the reference from mutually aligned training samples) (M) — **ADR-0010**
- [ ] Per-channel robust reference build: median + MAD over aligned polar training images (S)
- [ ] Predict path: z-map (deviation / MAD) → Gaussian smoothing → inverse-polar warp back to image space → high-percentile image score (M)
- [ ] Parameter defaults sweep on `set1` (ray count, smoothing sigma, score percentile, polar resolution); record chosen defaults and the numbers that justified them (M)
- [ ] Unit tests on synthetic discs: known centre/radius/rotation recovered within tolerance; injected blob raises the score (S)

## E7 — Evaluation (M3)

- [ ] Channel → sample score aggregation (`max` and `mean`, configurable per experiment) (S) — **ADR-0011**
- [ ] Sample-level and image-level ROC-AUC (image-level metrics only — no pixel masks in this dataset) (S)
- [ ] On-demand threshold endpoint: given a threshold, return the confusion matrix plus FP and FN sample lists without recomputing inference (M)
- [ ] Ranked lists: most-anomalous and most-normal samples with scores (S)
- [ ] Timing statistics: per-sample inference time, mean/median/p95, recorded in the MetricSet (S)

## E8 — Experiment UI (M3)

- [ ] Experiment create screen: dataset + split pickers, method picker, config form generated from the plugin's JSON Schema, preprocessing section (M) — **ADR-0007**
- [ ] Progress + logs screen: live WebSocket progress bar, streaming log view, cancel button, terminal states (M)
- [ ] Results screen: anomaly-map overlay with opacity slider, ranked lists, threshold slider driving the confusion matrix, TP/FP/TN/FN filtering of the sample list (L)
  - split when scheduled: overlay viewer → threshold + confusion panel → filtered sample list + ranked lists
- [ ] Experiment list with status, metrics summary, and reopen-from-persistence (S)

## E9 — Anomalib integration (M4–M5)

- [ ] Add anomalib dependency + standalone MPS smoke test script (train a few steps on a handful of images, report device and any unsupported ops) — **do this before writing wrapper code** (S) — **ADR-0008**
- [ ] `efficientad_anomalib` wrapper: `fit` / `predict` / `save` / `load`, epoch and step progress mapped onto the job event protocol, checkpoints into the experiment artifact directory (L)
  - split when scheduled: dataloader bridge from our Split → training loop + progress mapping → predict + map extraction → save/load round-trip
- [ ] `patchcore_anomalib` wrapper + memory sizing for 1280×1024 inputs (coreset ratio, memory-bank footprint, documented limits) (M)
- [ ] Preprocessing config bridge: our resize/normalization config drives anomalib rather than its defaults, so all methods see identical inputs (M)

## E10 — Comparison UI (M5)

- [ ] Multi-experiment comparison: metric table (ROC-AUC, confusion at a chosen threshold, timing) + overlay A/B view of two methods' anomaly maps on the same sample (M)

## E11 — Custom EfficientAD (M6)

- [ ] Reimplement EfficientAD from arXiv:2303.14535 as `efficientad_custom` behind the unchanged plugin interface: PDN student/teacher distillation, autoencoder branch, quantile-based score normalization, MPS training loop (L)
  - split when scheduled: PDN backbone → teacher pretraining/distillation → student loss → autoencoder branch → quantile normalization → training loop + progress events → comparison run against `efficientad_anomalib`

## E12 — README & polish (M7)

- [ ] Full README: setup from a fresh machine, dataset conventions, architecture overview, troubleshooting (M)
- [ ] "How to add a new anomaly-detection method" guide, written against the real interface with a worked example (S) — **ADR-0007**
- [ ] Docs refresh (system design + ADR amendments where implementation diverged) and backlog re-triage (S)

---

## Later / ideas

Not scheduled. Revisit at each milestone re-triage; promote into an epic when there is a concrete reason to.

- Per-channel score quantile normalization before aggregation, so one illumination channel cannot dominate the sample score by scale alone.
- Explicit detector for the part's asymmetric surface features as an orientation fallback when FFT angular correlation is ambiguous (near-rotationally-symmetric parts).
- Per-set classical references (separate `set1` / `set2` references) if lighting or fixturing drift between sets turns out to matter.
- `flat_folder` import adapter for single-image-per-sample datasets with no channel structure.
- Pixel-mask support (mask storage, pixel-level ROC-AUC / PRO) if masks are ever produced for this dataset.
- Uninformed Students (Student-Teacher Anomaly Detection, [papers.md](papers.md) #4) as a fourth method.
- Export / report generation: experiment results and comparison tables to PDF or HTML for sharing without the app.
