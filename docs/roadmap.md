# Roadmap

**visual-anomaly-lab** is a local desktop workbench for training, evaluating and comparing visual anomaly-detection methods. The goal is a *universal* anomaly-detection explorer for arbitrary image datasets, validated against public benchmarks whose published numbers can be checked against our own (**ADR-0015**). The private showcase dataset, images of a manufactured circular part, is one reference dataset among them, and only the classical baseline is showcase-dataset-specific (**ADR-0010**). The loop: import a dataset, browse grouped multi-view samples, train and compare methods under one evaluation protocol, and inspect anomaly-map overlays.

This roadmap follows the delivery order the brief asks for — *plan first* (M0: system design, ADRs, repo safety), *then one working vertical slice end-to-end with a single method* (M1 walking skeleton → M2 import and browse → M3 the six vertical-slice capabilities), *then the depth and the remaining methods behind the same interface* (M4 the workbench UI, M5 comparison, M6 custom EfficientAD, M7 PatchCore, M8 the classical baseline), *then the README* (M9). Nothing in M5–M8 is allowed to change the application outside the model plugin boundary (**ADR-0007**); if it does, the slice was not actually vertical.

**The order changed after M2.** M3 originally ran the vertical slice on `classical_circular`, which would have made the first end-to-end proof of the architecture a proof that it works for one dataset. The slice now runs on a dataset-agnostic method against public benchmarks, and the classical baseline moved to an optional M8. **ADR-0015** records why and what it costs.

**Sizing is honest, not aspirational.** This is one developer plus Claude Code on an Apple Silicon Mac, working in evenings and weekends. M0 and M1 are day-scale. **M2 and M3 are the two big ones**; they carry the whole product surface (import, browse, jobs, evaluation, three UI screens) and everything after them is comparatively cheap because it reuses that machinery. M3 carries real schedule risk: anomalib on MPS is the one place where an upstream incompatibility could cost days, which is why the smoke test comes before the wrapper. M4 is large but low-risk frontend work over a contract that already exists. M6 is large but isolated — a from-scratch paper reimplementation that touches nothing but one plugin. Milestones are strictly sequential; the backlog ([backlog.md](backlog.md)) is re-triaged at the end of each one.

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

**Status: complete (2026-08-06).**

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
- [x] A fresh clone reaches a running app using only the documented dev scripts.

**Size:** days. Mostly scaffolding; the only genuinely fiddly part is sidecar lifecycle and port handoff.

**What the fiddly part actually turned out to be.** Three things, all worth knowing before M7 touches packaging:
`uv run` sits between the shell and the interpreter, so the sidecar's orphan watchdog must probe the *recorded*
parent pid rather than compare `os.getppid()`; macOS keeps an application alive after its last window closes,
which would strand a sidecar; and a path-based router renders a blank page when the WebView loads
`…/index.html`, which is why routing is fragment-based (**ADR-0012**).

---

## M2 — Import + browse

**Status: complete (2026-08-06).**

**Goal.** Turn 3.2 GB of BMPs on disk into a queryable dataset of grouped samples, and make browsing them fast enough to be pleasant. This is the first milestone that touches the real data, so it is also where the data's irregularities have to be handled rather than assumed away.

**Scope**

- Manifest schema and the `channel_folders` import adapter: scan a directory tree, canonicalize channel names, group images into samples, compute `sha256`, and emit a reviewable manifest before anything is written to the database (**ADR-0006**).
- Import review UI: proposed samples, detected channels, and **explicit warnings** for irregular groups — the 2-channel group in `unsorted/`, ungroupable orphans, duplicate hashes, unreadable files. Channel count is data, never a constant (**ADR-0005**).
- Import **scan** runs as a Job with progress; commit is one synchronous transaction, because
  the expensive work is the walk and the hash, not the few hundred inserts (**ADR-0009**, **ADR-0013**).
- Dataset browser: virtualized thumbnail grid with label and channel filters.
- Grouped sample viewer: one sample, channel tabs, zoom/pan, metadata panel.
- Split creation: seeded, **sample-level** (never image-level — no channel of a sample may straddle the split boundary), normal-only training set, labeled normals and defects in val/test (**ADR-0011**).
- Thumbnail and preview cache under `data/`, with a post-import pre-warm job.
- Dataset verify operation: re-check that every recorded path exists and its `sha256` still matches.

**Exit criteria**

- [x] `set1` + `set2` import as **189 samples** (98 normal + 91 defect), each with its illumination channels correctly grouped; the count is asserted, not eyeballed — see `backend/tests/test_showcase_import.py`, which runs against the private tree on demand and is skipped everywhere else.
- [x] The irregular tree imports as **113 unlabeled samples** with its 2-channel group surfaced as a warning rather than silently dropped or padded, and the commit is blocked until that warning is acknowledged. *Rewritten from the original wording: the labelled corpus is `set1` + `set2` only, because the third tree has no clean defect / no-defect split. It is imported as a separate dataset during verification, which is what keeps the variable-channel-count path exercised against real data rather than only against fixtures.*
- [x] Scrolling the browser over the full dataset is fluid — the grid requests the `thumb` tier and nothing else, verified in the network panel.
- [x] A created split persists across an application relaunch and reports its exact composition (samples per subset, normal/defect counts).
- [x] Re-running import on an already-imported directory is idempotent — no duplicate samples.

**Size:** **1–2 weeks — one of the two big milestones.** The adapter is fiddly (real filenames are messier than expected), the review UI is real UI work, and the media cache has to be built properly or every later milestone feels slow.

**What measuring the data changed.** Three things, recorded in **ADR-0013**. The files that
were assumed not to group by stem do group, perfectly. Hashing the whole corpus takes about
two seconds rather than the minutes assumed, so the scan is cheap and *thumbnail rendering*
is the slow operation that actually justifies the job system. And one capture group has its
channel name fused into its own directory name, which a component-matching adapter reads as
twice as many single-image samples with no error anywhere — the reason channel matching is
token-level. **The ADR-0009 job machinery was built here rather than in M3**, since M2 needs
it for the scan and the pre-warm; M3 adds train and infer by writing one handler each.

---

## M3 — Universal vertical slice on EfficientAD

**Goal.** Deliver all six capabilities the brief requires from the vertical slice — import, display, create experiment, train/run a method, show scores and maps, persist and reopen results — on a **dataset-agnostic** method, measured against public benchmarks whose published numbers we can check ourselves against (**ADR-0015**). When this milestone closes, the application is *complete in shape*; everything afterwards is a new plugin or a new view.

The milestone was re-aimed after M2. It previously put `classical_circular` first, which would have made the first end-to-end proof of the architecture a proof that it works for one dataset. The infrastructure in scope was always method-agnostic; only the first method changed.

**Scope**

- **Import layer for public datasets (done).** `folder_classes` — point at the directories holding defect-free and defective images; one image becomes one sample with no channel, which is also the first thing to exercise the single-view path (**ADR-0016**). `csv_table` — every column name configurable, so a benchmark's own split table is read rather than re-drawn. Masks enter the catalog and `verify` walks them. `SplitStrategy.IMPORTED` materializes a published partition. The adapter-options form is generated from each adapter's JSON Schema, as **ADR-0006** always specified and as nothing previously implemented.
- **Bulk labelling and sample paging (done).** Label a whole filtered set in one action; walk an open sample's neighbours with the arrow keys, auto-advancing after each label.
- Model plugin interface (`fit` / `predict` / `save` / `load` / config JSON Schema) and a registry keyed by method name (**ADR-0007**).
- `train` and `infer` job handlers — one registry entry and one function each. The queue, protocol, cancellation, log tee-ing and WebSocket fan-out were built in M2 and are kind-agnostic; if either needs a change inside them, that is a finding about the boundary.
- **The diagnostics contract.** `Capabilities.produces_diagnostics`, and `ctx.emit_diagnostic(key, title, kind, payload)` writing float32 `.npy` maps under `artifacts/exp-<id>/diagnostics/` behind a self-describing `diagnostics.json`. Scalar series reuse the **existing** `metric` job event, which is itself a test of **ADR-0009**. M4's visualization is built once against this and never branches on model name.
- MPS smoke test **first**, as a standalone script, before any wrapper code (**ADR-0008**). Preprocessing config bridge so every method sees identical inputs.
- `efficientad_anomalib`: `fit` / `predict` / `save` / `load`, a Lightning callback mapping epochs and per-branch losses onto the job protocol, and diagnostics from forward hooks on the teacher, student and autoencoder. The ImageNette penalty set is an explicit, visible, cancellable download step — not a hidden network call in a tool that claims to be local-only.
- `pixel_reference`: the dataset-agnostic floor baseline. numpy and Pillow only, trains in seconds, exercises the whole results path before torch is involved.
- Evaluation layer, independent of any model (**ADR-0011**): channel→sample aggregation, sample- and image-level ROC-AUC, average precision, ranked lists, timing — **plus pixel-level ROC-AUC and PRO** over the samples that have masks. Pixel metrics stream through a fixed-bin score histogram rather than accumulating maps, so memory stays constant in the number of test images.
- Experiment screens: create (config form from the plugin's JSON Schema), progress + live logs, results with a threshold slider, confusion matrix and TP/FP/TN/FN lists, and a sample viewer with an anomaly-map overlay and ground-truth mask contours.

**Exit criteria**

- [x] A whole filtered set of samples is labelled in **one action**, and an open sample pages to its neighbours with the arrow keys, auto-advancing after each label.
- [x] `git check-ignore` matches `datasets/…` and does **not** match the backend datasets package; `scripts/check-repo-safety.sh` exits 0 with a `datasets/` rule in place.
- [x] GKN imports through `folder_classes` as 203 normal + 197 defect single-image samples, with `Nick` / `Scratch` recorded as the defect type.
- [x] One VisA class imports through `csv_table` with its **official** train/test split (900 train normal, 100 test normal, 100 test anomaly) and its 100 ground-truth masks attached, and `verify` reports no drift on either.
- [x] The import screen's options form is generated from the adapter's schema, and a required option blocks the scan with the field named.
- [x] The **MPS smoke test runs first**, as a standalone script, before any wrapper code (**ADR-0008**). Measured on this Mac: MPS 123 ms/step against CPU 295 ms. It earned itself immediately — EfficientAD's penalty batch defaults to `None` in the signature and is dereferenced unconditionally in training, which the script found before a line of wrapper existed.
- [x] EfficientAD trains to completion on this Mac (MPS, with a documented CPU fallback and its runtime) on that split. Two runs at the default 256×256: **4 000 steps in 526 s** and **20 000 steps in 2 624 s**, both on MPS, inference 46.3 ms/image. The CPU fallback is selected automatically when MPS is unavailable and costs the ratio the smoke test measured — 295 against 123 ms/step, so roughly 2.4× — putting the 20 000-step run near 1 h 45 m on CPU. That ratio is measured per step; no full CPU run was made, and the number is presented as the extrapolation it is.
- [x] Image-level and **pixel-level** ROC-AUC are both reported, and the gap against the published VisA figure is accounted for rather than waved at. Best run — 20 000 steps on the official split — is **0.809 sample ROC-AUC, 0.810 image AP, 0.933 pixel ROC-AUC, 0.809 AU-PRO**, against a published EfficientAD figure of roughly 0.98 image AU-ROC (reported as a mean over the twelve VisA classes, not for candle alone). The gap is real and is broken down below rather than left as a number.

  *Measured, by controlled experiment:*

  | | sample ROC-AUC | pixel ROC-AUC | AU-PRO |
  | --- | --- | --- | --- |
  | 4 000 steps, official split | 0.744 | 0.853 | 0.571 |
  | 4 000 steps, quantiles on held-out normals | 0.769 | 0.873 | 0.595 |
  | 20 000 steps, official split | **0.809** | **0.933** | **0.809** |

  Step count is the larger effect (+0.065 image, +0.080 pixel, +0.238 AU-PRO from 4 000 to 20 000);
  score-normalization calibration is the smaller one (+0.025 image at fixed steps on a byte-identical
  test set), and is a lower bound because the holdout also costs 90 training images.

  *Ruled out by reading anomalib's source rather than by assumption:* input normalization
  (`imagenet_norm_batch` runs inside every branch's forward, so `[0, 1]` tensors are what it wants);
  preprocessing (anomalib's own EfficientAD pre-processor is a bare `Resize` and it *rejects* a
  `Normalize` in the transform, which is exactly what our bridge feeds); optimizer and schedule
  (Adam plus `StepLR(0.95 · max_steps, γ = 0.1)`, identical); batch size 1; and the penalty
  pipeline, since `prepare_imagenette_data` is anomalib's routine called directly.

  *Named and untested:* the paper trains **70 000 steps**, 3.5× what was run here, and the measured
  trend points that way — that run was started and then deliberately stopped, so it is deferred, not
  overlooked. Two smaller candidates remain open: `model_size` is `small` (EfficientAD-S) where
  headline tables often quote the -M variant, and the training loop samples uniformly *with
  replacement* where a shuffled `DataLoader` would not.
- [x] `pixel_reference` runs on the same split through the identical interface and the same results screen, giving the deep result a floor to beat: **0.814 sample ROC-AUC, 0.888 pixel ROC-AUC, 0.808 AU-PRO** on VisA candle, trained and scored in 6 seconds on CPU.
- [x] Adding both methods required **no** change to the queue, protocol, cancellation or fan-out — each cost one registry entry and one handler function. Two defects were *found* in the queue by the first job long enough to draw a progress bar, and are recorded below rather than papered over.
- [x] Anomaly maps overlay in spatial alignment with the source image, with ground-truth contours where masks exist.
- [x] Force-quitting the app mid-training leaves an orphan-free system, and the interrupted Job is marked `failed` with its log preserved on the next startup.
- [x] Closing and reopening the app restores the experiment list; any past experiment reopens with identical numbers — nothing is recomputed on read, so this is structural rather than lucky.

**Findings recorded during M3**

Two are about the job machinery M2 built, and were invisible until a job ran for more than a few
seconds while a library drew a progress bar:

- **`readline` cannot read a progress bar.** `asyncio.StreamReader.readline` raises `ValueError`
  past a 64 KiB line, and tqdm separates its frames with `\r`, never `\n` — so a progress bar is
  one line, and a long enough download makes it an over-long one. Worker output is now read in
  chunks and split on both terminators, which also makes the log tail render as a terminal would.
- **The runner had no guard.** That `ValueError` escaped `_execute` and killed the runner task.
  Because nothing awaited it the failure was silent: the running job stayed `running` for ever and
  every later job stayed `queued`, with no error anywhere. `_execute` now finalizes its own job on
  failure and kills the worker, and the runner loop survives anything `_execute` can raise.

Three more were found by running the thing and looking at it, rather than by a test:

- **A stored anomaly map with a stray channel axis** was accepted by `write_map` and failed
  minutes later inside the evaluation layer, with an error naming a dtype rather than the
  plugin. Maps are squeezed and checked where they are written.
- **The inference run's diagnostics index erased the training run's**, leaving the
  architecture graph on disk but unreferenced. The index is merged on `(key, image_id)`.
- **An experiment left mid-training by a crash stayed `training` for ever.** Jobs were
  reconciled at startup; experiments were not. On screen, `training` is indistinguishable
  from a run genuinely in progress.

And two are about how the results read rather than whether they are right:

- **The anomaly-map overlay tinted the whole photograph.** A colormap's low end is still a
  colour, so at any opacity and under any blend mode the regions where the model found
  nothing looked as processed as the region where it found something. Alpha now follows the
  score.
- **A model's configuration panel was an empty box.** The rule that folds away
  already-defaulted options is right for an adapter and leaves a model — whose
  hyperparameters all have defaults — showing nothing at all.

Two are about the method:

- **"ROC-AUC is threshold-free, so the score normalization cannot affect it" is wrong**, and it
  was written into the wrapper as a reassurance next to the fallback that triggers it. EfficientAD
  scores an image as `max_p [ w_st·map_st[p] + w_ae·map_stae[p] ]`, and the two weights come from
  *different* quantile pairs. A shared scale and an offset are monotone and genuinely cannot move a
  ranking — but the fit decides the **ratio** of the two weights, which is the relative influence of
  the student-teacher and autoencoder branches before the max. Change it and images reorder, which
  is exactly what ROC-AUC measures. Measured, not argued: at a fixed 4 000 steps on a byte-identical
  test set, fitting the quantiles on 90 genuinely held-out normals instead of on the training
  normals moved sample ROC-AUC 0.744 → **0.769**, pixel ROC-AUC 0.853 → **0.873** and AU-PRO
  0.571 → **0.595** — and that understates it, since the holdout also costs 90 training images.
  `holdout_from_train` exists so this is testable without touching the published test set.
- **The EfficientAD wrapper does not use Lightning**, as the plan assumed it would.
  `EfficientAd.on_train_start` reads `self.trainer.datamodule`, so the Lightning path means adopting
  anomalib's datamodule and with it anomalib's preprocessing — which would break the property that
  makes any comparison meaningful. anomalib still supplies the architecture, the losses, the maps,
  the pretrained teacher and the statistics routines. Recorded in the module and in **ADR-0018**'s
  neighbourhood; the cost is that our loop can drift from theirs.

**Size:** **2–3 weeks — the big milestone.** The EfficientAD integration carries the schedule risk; the import layer and labelling workflow landed first and are usable on their own.

---

## M4 — The researcher's workbench UI

**Goal.** Make the method *legible*, not just runnable. Everything here is built on M3's diagnostics contract, so it renders whatever a model declares and never branches on model name.

**Scope**

- **Model architecture view** — interactive PDN teacher / student / autoencoder diagram with real tensor shapes and parameter counts, generated from a dry forward pass rather than hand-drawn, so it cannot go stale.
- **Teacher inspector** — pick a sample and see the input, the teacher's feature maps as both a PCA-to-RGB composite and a per-channel small-multiples grid, and a feature-magnitude heatmap.
- **Training charts** — per-branch loss curves (`loss_st` / `loss_ae` / `loss_stae`), learning rate, quantile-normalization parameters, and val AUROC per epoch where a val subset exists. Fed by the `metric` events already streaming over the WebSocket.
- **Benchmark charts** — score histograms by class with the threshold drawn on them, ROC and PR curves, confusion matrix, per-defect-type breakdown (from the `notes` an adapter recorded), timing summary.
- **Diagnostic overlays** — student–teacher error and autoencoder–student error side by side against the combined map and the ground-truth mask.

**Charting: hand-rolled SVG primitives** in `frontend/src/components/charts/`, not a charting library. The chart types needed are few and simple, the frontend is deliberately on current React / TypeScript / Tailwind, and adding a library that lags those versions would mean either downgrading the project or isolating a laggard — a cost that outweighs five straightforward chart components.

**Exit criteria**

- [ ] A researcher can see what the teacher produces on a chosen sample, without reading any code.
- [ ] Training is watchable live: per-branch losses update as the run progresses, and the charts survive a page reload mid-run.
- [ ] The architecture view's shapes and parameter counts are read from the model, and a change to the model's configuration is visible in the diagram without any edit here.
- [ ] Every visualization is driven by the diagnostics contract, so `efficientad_custom` gets all of them for free in M6.

**Size:** large, but low-risk and pausable — it is frontend work against a contract that already exists.

---

## M5 — Comparison UI

**Goal.** Several methods, one split, one evaluation protocol, compared directly. This is what makes the workbench worth having.

**Scope**

- Multi-experiment metric table: sample / image / pixel ROC-AUC, PRO, confusion at a chosen threshold, timing.
- A/B overlay showing two methods' anomaly maps on the same sample, to see where they disagree.
- TP / FP / TN / FN filtering polish across the results and comparison screens.

**Exit criteria**

- [ ] `pixel_reference` and `efficientad_anomalib` are comparable side by side on the same split under the same evaluation protocol, with identical preprocessing.
- [ ] Any sample can be opened in A/B overlay.
- [ ] Every requirement in the brief's UI list is implemented.

**Size:** medium — a focused piece of frontend work over an evaluation layer that already produces the numbers.

---

## M6 — Custom EfficientAD

**Goal.** Reimplement EfficientAD from the paper (arXiv:2303.14535) in PyTorch, behind the same interface, with the anomalib version's number as its yardstick — the research payoff of having built the workbench.

**Scope**

- PDN student/teacher architecture and distillation loss; the autoencoder branch; quantile-based score normalization; an MPS training loop.
- Registered as `efficientad_custom` behind the unchanged plugin interface (**ADR-0007**), emitting the same diagnostics as the wrapper so M4's views work on it unchanged.

**Exit criteria**

- [ ] `efficientad_custom` trains and infers on MPS and produces maps and scores through the standard interface.
- [ ] The comparison view shows both implementations side by side, and the gap between them is measured and explained (a gap is an acceptable outcome; an unexplained gap is not).
- [ ] Every M4 visualization works on it with no new code — if any needed a special case, that is a finding about the diagnostics contract and gets an ADR.

**Size:** large, but **isolated and low-risk to the rest of the system**. Split it into subtasks (backbone → distillation → autoencoder branch → normalization → training loop) when it is actually scheduled.

---

## M7 — PatchCore

**Goal.** A third method, and the first one whose resource profile is genuinely awkward.

**Scope**

- `patchcore_anomalib` wrapper, with explicit attention to memory: a coreset memory bank over a full training set at native resolution needs a sized, documented configuration rather than library defaults.

**Exit criteria**

- [ ] PatchCore trains and infers without exhausting memory, and its memory-bank configuration is documented.
- [ ] It appears in the comparison view alongside the others with no changes outside the plugin.

**Size:** small-to-medium — the wrapper reuses M3's integration path.

---

## M8 — `classical_circular` (optional)

**Goal.** The showcase-specific baseline, if it is still wanted once the universal tool exists.

**Scope**

- Circle detection (Hough seed → radial-ray subpixel edges → robust circle fit, with a median-prior fallback), polar transform, FFT angular-correlation orientation alignment against a bootstrapped reference, per-channel median/MAD reference build, and a predict path producing a z-map → smoothing → inverse-polar warp → percentile score (**ADR-0010**).
- Built as a circle-fit front-end onto `pixel_reference`, which is already the geometry-free core of the same algorithm.

**Exit criteria**

- [ ] It runs on the showcase dataset through the identical interface, and its inverse-polar warp is verified rather than assumed.
- [ ] It is the *only* component that assumes anything about the dataset's geometry.

**Size:** medium. Deferring an optional milestone in a spare-time project is close to cancelling it, and **ADR-0015** says so outright.

---

## M9 — Polish + README

**Goal.** Make the project reproducible by someone who is not the author (including the author six months later), and bring the documentation back in line with what was actually built.

**Scope**

- Full README: setup from a fresh machine, how to obtain and import each reference dataset, architecture overview, and a **"how to add a new anomaly-detection method"** guide walking through the plugin interface with a working example.
- UX polish pass: loading and empty states, error surfaces, keyboard navigation, consistent layout across screens — it should read as an engineering tool, not a debug panel.
- Documentation refresh: `system-design.md` and the ADRs updated where implementation diverged from the decision (with amendments, not silent edits).
- Backlog re-triage: drop what no longer matters, promote what the work revealed.

**Exit criteria**

- [ ] Following the README alone on a fresh machine reaches full M3 functionality on a **public** dataset — download, import, train, view results — with no undocumented steps and no private data required.
- [ ] The "add a new method" guide has been followed end-to-end at least once (M6 counts if it was written first and used as the recipe).
- [ ] Documentation matches reality: no ADR describes a decision that was silently reversed.
- [ ] Backlog re-triaged and the "Later / ideas" list refreshed.

**Size:** medium. README and docs work expands to fill the time available; timebox it.
