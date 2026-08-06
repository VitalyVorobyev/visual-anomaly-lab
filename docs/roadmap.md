# Roadmap

**visual-anomaly-lab** is a local desktop workbench for training, evaluating and comparing visual anomaly-detection methods. The long-term goal is a *universal* anomaly-detection explorer for arbitrary image datasets; the showcase dataset, images of a manufactured circular part, is the first reference dataset — the initial focus, not the final scope — and only the classical baseline is showcase-dataset-specific (circular parts) (**ADR-0010**). The loop: import a dataset, browse grouped multi-view samples, train and compare methods under one evaluation protocol, and inspect anomaly-map overlays. This roadmap follows the delivery order the brief asks for — *plan first* (M0: system design, ADRs, repo safety), *then one working vertical slice end-to-end with a single method* (M1 walking skeleton → M2 import and browse → M3 the six vertical-slice capabilities running on `classical_circular`), *then the remaining methods behind the same interface* (M4 EfficientAD, M5 PatchCore, M6 custom EfficientAD), *then the README* (M7). Nothing in M4–M6 is allowed to change the application outside the model plugin boundary (**ADR-0007**); if it does, the slice was not actually vertical.

**Sizing is honest, not aspirational.** This is one developer plus Claude Code on an Apple Silicon Mac, working in evenings and weekends. M0 and M1 are day-scale. **M2 and M3 are the two big ones — budget 1–2 weeks each**; they carry the whole product surface (import, browse, jobs, evaluation, three UI screens) and everything after them is comparatively cheap because it reuses that machinery. M4 is medium with real schedule risk: anomalib on MPS is the one place where an upstream incompatibility could cost days. M5 is medium. M6 is large but isolated — it is a from-scratch paper reimplementation that touches nothing but one plugin. M7 is medium. Milestones are strictly sequential; the backlog ([backlog.md](backlog.md)) is re-triaged at the end of each one.

---

## M0 — Repo safety + foundation docs

**Status: complete (2026-08-06).**

**Goal.** Make it impossible to leak the private dataset by accident, and write down the architecture before writing any application code, so that the vertical slice is implemented against a decided design rather than an improvised one.

**Scope**

- Hardened `.gitignore`: `privatedata/`, `*.bmp` / `*.BMP` as defense in depth, `data/`, model artifact extensions, SQLite files (**ADR-0001**).
- `scripts/check-repo-safety.sh` — pre-push verification that nothing private is staged or tracked.
- `README.md` stub with the private-data warning and pointers into `docs/`.
- `docs/system-design.md` — architecture, domain model, API surface, directory layout.
- `docs/roadmap.md` and `docs/backlog.md` (this document and its companion).
- ADRs 0001–0011: private data isolation, monorepo layout, FastAPI sidecar, SQLite + filesystem artifacts, domain model, import adapters + manifest, model plugin interface, hybrid DL strategy on MPS, subprocess jobs, classical baseline algorithm, evaluation protocol.
- First commit and push to GitHub, verified clean.

**Exit criteria**

- [x] `git check-ignore -v` returns a match for real paths under `privatedata/` (not just the directory name — an actual `.bmp` file path several levels deep).
- [x] `git ls-files` contains no image files of any extension; `scripts/check-repo-safety.sh` exits 0.
- [x] `docs/system-design.md`, `docs/roadmap.md`, `docs/backlog.md` and ADRs 0001–0011 are mutually consistent: same terminology (Dataset, Channel, Sample, Image, Split, SplitAssignment, Experiment, Job, ImageResult, SampleResult, MetricSet), same registry keys (`classical_circular`, `efficientad_anomalib`, `patchcore_anomalib`, `efficientad_custom`), no contradictory claims about storage or process boundaries.
- [x] Repository pushed to GitHub; the pushed tree contains documentation and scripts only.

**Size:** days. Mostly writing, and the writing is the point — every hour here removes a day of rework in M2/M3.

---

## M1 — Walking skeleton

**Goal.** Get one thin thread of every technology in the stack running at once: Tauri window → React app → HTTP + WebSocket → FastAPI sidecar → SQLite. No features, just the wiring, so that later milestones never have to debug infrastructure and a feature at the same time.

**Scope**

- `uv`-managed Python backend package on Python 3.12; FastAPI app with `GET /api/health` returning version and database status (**ADR-0003**).
- SQLite migration runner (forward-only, versioned SQL files, `PRAGMA user_version`) and schema v1 covering every entity in the domain model (**ADR-0004**, **ADR-0005**).
- Vite + React + TypeScript frontend scaffold: routing, application shell, and an API client generated from the backend's OpenAPI schema (**ADR-0012**).
- Tauri desktop shell that spawns the sidecar as a child process, reads the chosen port back from it, and tears the process down on window close — including the crash and force-quit paths.
- WebSocket echo endpoint proven end-to-end from the React app.
- Dev scripts: run backend alone, run frontend alone against it, run the full desktop app.
- Guardrails: `ruff`, `mypy --strict`, `pytest`, `vitest`, a versioned pre-commit hook running the private-data guard, and CI.

**Exit criteria**

- [x] The desktop app opens and displays live backend health (version, schema version, DB path) fetched from the spawned sidecar.
- [x] The *same* React UI, served by `vite dev` in an ordinary browser, works against a manually started `uv run` backend — the browser path stays first-class for the whole project, because it is how debugging will actually happen.
- [x] Closing the app leaves no orphaned Python process (verified with `ps`).
- [ ] A fresh clone reaches a running app using only the documented dev scripts.

**Size:** days. Mostly scaffolding; the only genuinely fiddly part is sidecar lifecycle and port handoff.

**What the fiddly part actually turned out to be.** Three things, all worth knowing before M7 touches packaging:
`uv run` sits between the shell and the interpreter, so the sidecar's orphan watchdog must probe the *recorded*
parent pid rather than compare `os.getppid()`; macOS keeps an application alive after its last window closes,
which would strand a sidecar; and a path-based router renders a blank page when the WebView loads
`…/index.html`, which is why routing is fragment-based (**ADR-0012**).

---

## M2 — Import + browse

**Goal.** Turn 3.2 GB of BMPs on disk into a queryable dataset of grouped samples, and make browsing them fast enough to be pleasant. This is the first milestone that touches the real data, so it is also where the data's irregularities have to be handled rather than assumed away.

**Scope**

- Manifest schema and the `channel_folders` import adapter: scan a directory tree, canonicalize channel names, group images into samples, compute `sha256`, and emit a reviewable manifest before anything is written to the database (**ADR-0006**).
- Import review UI: proposed samples, detected channels, and **explicit warnings** for irregular groups — the 2-channel group in `unsorted/`, ungroupable orphans, duplicate hashes, unreadable files. Channel count is data, never a constant (**ADR-0005**).
- Import commit runs as a Job with progress (**ADR-0009**).
- Dataset browser: virtualized thumbnail grid with label and channel filters.
- Grouped sample viewer: one sample, channel tabs, zoom/pan, metadata panel.
- Split creation: seeded, **sample-level** (never image-level — no channel of a sample may straddle the split boundary), normal-only training set, labeled normals and defects in val/test (**ADR-0011**).
- Thumbnail and preview cache under `data/`, with a post-import pre-warm job.
- Dataset verify operation: re-check that every recorded path exists and its `sha256` still matches.

**Exit criteria**

- [ ] `set1` + `set2` import as **189 samples** (98 normal + 91 defect), each with its illumination channels correctly grouped; the count is asserted, not eyeballed.
- [ ] `unsorted/` imports as unlabeled samples with the 2-channel group surfaced as a warning rather than silently dropped or padded.
- [ ] Scrolling the browser over the full dataset is fluid (thumbnails served from cache, no full-resolution BMP decode on the grid path).
- [ ] A created split persists across an application relaunch and reports its exact composition (samples per subset, normal/defect counts).
- [ ] Re-running import on an already-imported directory is idempotent — no duplicate samples.

**Size:** **1–2 weeks — one of the two big milestones.** The adapter is fiddly (real filenames are messier than expected), the review UI is real UI work, and the media cache has to be built properly or every later milestone feels slow.

---

## M3 — Vertical slice complete (classical baseline)

**Goal.** Deliver all six capabilities the brief requires from the vertical slice — import, display, create experiment, train/run a method, show scores and maps, persist and reopen results — with `classical_circular` as the one method. When this milestone closes, the application is *complete in shape*; everything afterwards is a new plugin or a new view.

**Scope**

- Model plugin interface (`fit` / `predict` / `save` / `load` / config JSON Schema) and a registry keyed by method name (**ADR-0007**).
- Job machinery: subprocess workers, JSON-lines event protocol on stdout, FIFO queue, Job table, log files under artifacts, WebSocket fan-out to the UI, cancellation (SIGTERM → grace → SIGKILL), and crash recovery on startup (**ADR-0009**).
- `classical_circular` implementation (**ADR-0010**): circle detection (Hough seed → radial-ray subpixel edges → robust circle fit, with a median-prior fallback), polar transform, FFT angular-correlation orientation alignment against a bootstrapped reference, per-channel median/MAD reference build, and a predict path producing a z-map → smoothing → inverse-polar warp → percentile-based image score.
- Experiment screens: create (config form driven by the plugin's JSON Schema), progress + live logs, results.
- Anomaly-map overlay with an opacity slider, over the selected channel.
- Evaluation layer, independent of any model (**ADR-0011**): channel→sample score aggregation, sample-level and image-level ROC-AUC, an interactive threshold slider driving a confusion matrix, FP/FN lists, ranked most-normal / most-anomalous lists, per-sample inference timing. **Image-level metrics only** — this dataset has no pixel masks.
- Experiments persist and reopen with their full config, metrics, results, and logs.

**Exit criteria**

- [ ] All six vertical-slice bullets from the brief work end-to-end with `classical_circular`, driven entirely from the UI, on the real dataset.
- [ ] Sample-level ROC-AUC is reported on the test split, and the threshold slider recomputes the confusion matrix and FP/FN lists without retraining.
- [ ] Anomaly maps display as overlays at adjustable opacity and are spatially aligned with the source image (the inverse-polar warp is verified, not assumed).
- [ ] Force-quitting the app mid-training leaves an orphan-free system, and the interrupted Job is marked `failed` with its log preserved on the next startup — not left `running` forever.
- [ ] Cancelling a running job from the UI actually stops the worker within the grace period.
- [ ] Closing and reopening the app restores the experiment list; any past experiment reopens with identical numbers.

**Size:** **1–2 weeks — the other big milestone**, and the highest-value one. The job machinery and the results screen are each multi-day; the classical algorithm needs a parameter sweep on `set1` before its numbers mean anything.

---

## M4 — EfficientAD via anomalib

**Goal.** Prove the plugin interface is real by adding a deep method that shares none of the classical method's implementation — and confirm anomalib actually trains on this Mac's MPS backend.

**Scope**

- Add the anomalib dependency and run an MPS smoke test **first**, before writing wrapper code (**ADR-0008**).
- `efficientad_anomalib` wrapper: `fit` / `predict` / `save` / `load`, epoch and step progress mapped onto the existing Job event protocol, checkpoints written into the experiment's artifact directory.
- Preprocessing config bridge: resize/normalization decided by our config, not hidden inside anomalib defaults, so methods are compared on the same inputs.
- Method selection in the existing experiment-create screen — with **no changes** to the jobs, evaluation, or results layers.

**Exit criteria**

- [ ] EfficientAD trains to completion on this Mac (MPS, with a documented CPU fallback and its runtime), on the same split used in M3.
- [ ] It produces anomaly maps and image scores through the identical `ModelPlugin` interface, rendered by the same results screen.
- [ ] Classical vs EfficientAD numbers are visible side by side (even if only as two experiment result pages — the comparison view lands in M5).
- [ ] Any MPS incompatibility that had to be worked around is recorded in an ADR amendment, not just in a commit message.

**Size:** medium — but this is the milestone most likely to slip. If an anomalib op has no MPS kernel, the fallback path (per-op CPU fallback, pinned versions, or CPU-only training with documented runtime) costs days rather than hours. The smoke test exists to find that out on day one instead of day four.

---

## M5 — PatchCore + comparison UI

**Goal.** Add the third method and build the view that makes the whole workbench worth having: several methods, one split, one evaluation protocol, compared directly.

**Scope**

- `patchcore_anomalib` wrapper, with explicit attention to memory: 1280×1024 inputs and a coreset memory bank need a sized, documented configuration rather than library defaults.
- Experiment comparison view: multi-experiment metric table (ROC-AUC, confusion at a chosen threshold, timing) and an overlay A/B view showing two methods' anomaly maps on the same sample.
- TP / FP / TN / FN filtering polish across the results and comparison screens.

**Exit criteria**

- [ ] `classical_circular`, `efficientad_anomalib` and `patchcore_anomalib` are comparable side by side on the same split under the same evaluation protocol, with identical preprocessing.
- [ ] PatchCore trains and infers without exhausting memory on the full training set at native resolution, and its memory-bank configuration is documented.
- [ ] Any sample can be opened in A/B overlay to see where two methods disagree.
- [ ] Every requirement in the brief's UI list is now implemented.

**Size:** medium. The wrapper reuses M4's integration path; the comparison view is a focused piece of frontend work.

---

## M6 — Custom EfficientAD

**Goal.** Reimplement EfficientAD from the paper (arXiv:2303.14535) in PyTorch, behind the same interface, so the anomalib version and the from-scratch version can be compared directly — the research payoff of having built the workbench.

**Scope**

- PDN student/teacher architecture and distillation loss; the autoencoder branch; quantile-based score normalization; an MPS training loop.
- Registered as `efficientad_custom` behind the unchanged plugin interface (**ADR-0007**).
- Direct comparison against `efficientad_anomalib` on the same split in the M5 comparison view.

**Exit criteria**

- [ ] `efficientad_custom` trains and infers on MPS and produces maps and scores through the standard interface.
- [ ] The comparison view shows both implementations side by side, and the gap between them is measured and explained (a gap is an acceptable outcome; an unexplained gap is not).
- [ ] No application code outside the plugin needed modification to add it — if it did, that is a finding about the interface and gets an ADR.

**Size:** large, but **isolated and low-risk to the rest of the system**. It is a self-contained research task that can be paused and resumed without blocking anything. Split it into subtasks (backbone → distillation → autoencoder branch → normalization → training loop) when it is actually scheduled.

---

## M7 — Polish + README

**Goal.** Make the project reproducible by someone who is not the author (including the author six months later), and bring the documentation back in line with what was actually built.

**Scope**

- Full README: setup from a fresh machine, dataset conventions and directory layout, architecture overview, and a **"how to add a new anomaly-detection method"** guide walking through the plugin interface with a working example.
- UX polish pass: loading and empty states, error surfaces, keyboard navigation, consistent layout across screens — it should read as an engineering tool, not a debug panel.
- Documentation refresh: `system-design.md` and the ADRs updated where implementation diverged from the decision (with amendments, not silent edits).
- Backlog re-triage: drop what no longer matters, promote what the work revealed.

**Exit criteria**

- [ ] Following the README alone on a fresh machine (plus a private dataset) reaches full M3 functionality — import, browse, train `classical_circular`, view results — with no undocumented steps.
- [ ] The "add a new method" guide has been followed end-to-end at least once (M6 counts if it was written first and used as the recipe).
- [ ] Documentation matches reality: no ADR describes a decision that was silently reversed.
- [ ] Backlog re-triaged and the "Later / ideas" list refreshed.

**Size:** medium. README and docs work expands to fill the time available; timebox it.
