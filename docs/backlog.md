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

- [ ] Backend scaffold: `uv` project, package layout, FastAPI app, `GET /health` (version, schema version, DB path), settings module, ruff + pytest wired (S) — **ADR-0003**
- [ ] SQLite migration runner (forward-only numbered SQL files, `schema_version` table, applied at startup) + schema v1 covering **all** ADR-0005 entities: Dataset, Channel, Sample, Image, Split, SplitAssignment, Experiment, Job, ImageResult, SampleResult, MetricSet — with indices and foreign keys (M) — **ADR-0004**, **ADR-0005**
- [ ] Frontend scaffold: Vite + React + TypeScript, router, app shell/layout, typed API client with a single configurable base URL (S)
- [ ] Tauri shell: spawn the Python sidecar, pick a free port, hand it to the frontend, terminate the child on window close *and* on crash; verify no orphan process survives (M) — **ADR-0003**
- [ ] WebSocket echo endpoint + frontend hook, proven end-to-end in both the desktop app and a plain browser (S)
- [ ] Dev scripts (backend only, frontend only, full desktop) + a README dev section documenting the browser-against-`uv run` workflow (S)

## E3 — Dataset & import (M2)

- [ ] Manifest schema (JSON): datasets, proposed samples, images, per-image `sha256`, detected channels, warnings — versioned, reviewable before commit (M) — **ADR-0006**
- [ ] `channel_folders` adapter: walk the tree, canonicalize channel names with fuzzy matching (case, separators, near-miss folder spellings) onto a per-dataset channel set, group by sample key, flag ungroupable files (M) — **ADR-0006**
- [ ] Scan + commit endpoints: `scan` produces a manifest without writing to the DB, `commit` runs as a Job with progress and is idempotent on re-import (M) — **ADR-0009**
- [ ] Import review UI: dataset summary, detected channels, sample preview, and a **warnings panel** — 2-channel group, orphan images, duplicate hashes, unreadable files — with commit blocked until warnings are acknowledged (M)
- [ ] Dataset browser: virtualized thumbnail grid, filter by label / channel / split membership, sample count header (M)
- [ ] Grouped sample viewer: channel tabs (count driven by the data), synchronized zoom/pan across channels, metadata panel (M)
- [ ] Split creation: seeded RNG, **sample-level** assignment, normal-only train strategy, val/test containing normals and defects, persisted with its seed and strategy so it is reproducible (M) — **ADR-0011**
- [ ] Dataset verify operation: re-check every recorded path exists and its `sha256` matches; report drift (S)

## E4 — Media (M2)

- [ ] Thumbnail + preview cache under `data/` (content-addressed by image `sha256` + size) and serving endpoints with `ETag` and immutable cache headers (M)
- [ ] Post-import pre-warm job that generates thumbnails for a whole dataset with progress (S)
- [ ] Full-resolution PNG endpoint for the sample viewer (BMP decoded on demand, never sent raw) (S)

## E5 — Model interface & jobs (M3)

- [ ] `models/base.py`: `ModelPlugin` interface (`fit`, `predict`, `save`, `load`, config JSON Schema, capability flags) + registry keyed by `classical_circular`, `efficientad_anomalib`, `patchcore_anomalib`, `efficientad_custom` (S) — **ADR-0007**
- [ ] Subprocess worker + JSON-lines event protocol on stdout (`progress`, `log`, `metric`, `artifact`, `done`, `error`), with a parent-side parser resilient to partial lines and interleaved library output (L) — **ADR-0009**
  - split when scheduled: protocol + parser → worker entrypoint + config handoff → parent supervisor + status transitions
- [ ] FIFO job queue + Job table persistence + crash recovery on startup (any Job left `running` with a dead PID → `failed`, log preserved) (M) — **ADR-0009**
- [ ] WebSocket fan-out of job events to the UI + per-job log files written under the experiment's artifact directory (M)
- [ ] Cancellation: SIGTERM → grace period → SIGKILL, with the Job ending in `cancelled` and no orphan children (S)

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
