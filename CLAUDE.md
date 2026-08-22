# CLAUDE.md

Guidance for Claude Code working in this repository.

`visual-anomaly-lab` is a local desktop workbench (React + TypeScript + Tauri UI, Python FastAPI sidecar,
SQLite + filesystem artifacts) for importing image datasets, training anomaly-detection methods, and
comparing them under one evaluation protocol.

## Steering: the goal is universal

- The target is a **universal anomaly-detection explorer for arbitrary image datasets**. The private
  showcase dataset is one reference dataset, not the scope.
- **Only** a `classical_circular` plugin — not built, and optional — may assume anything about the
  showcase dataset's geometry. Domain model, import layer, DL methods (`efficientad_anomalib`,
  `efficientad_custom`, `patchcore_anomalib`, `dinomaly_anomalib`, `glass_anomalib`), evaluation
  layer, and UI must stay dataset-agnostic.
- **Public reference datasets live under `/datasets/` and are never committed** — gitignored for size, not
  secrecy, and credited in the README (ADR-0015). VisA (with masks and official splits) and GKN are the
  current two. `check-repo-safety.sh` fails if anything under `datasets/` is staged. Note the leading
  slash: an unanchored pattern would also match `backend/src/anomaly_lab/datasets/`, the adapter package.
- **Channel count is data, never schema.** No constant, enum, column, or UI layout may encode how many
  acquisition channels a dataset has. A 2-channel sample must render and score with no special case.

## Private data — highest priority

- **The private showcase images live outside this repository entirely (ADR-0022).** Never read, open,
  copy, or move them; never create a `privatedata/` directory or a symlink to one, which would put
  them back inside the tree where `git add -A` can reach them. They are referenced in place by
  absolute path. Do not sample them "just to check the format".
- **Never `git add -A` or `git add .`** — stage explicit paths, always.
- **Run `scripts/check-repo-safety.sh` after staging and before any commit or push.** It must exit 0.
- Test fixtures are **small synthetic images (PNG)** generated in code or checked in at trivial size.
  Never a real dataset file, never a `.bmp`.
- **Do not name or describe the showcase dataset's product identity in committed files** (README, code,
  comments, tests, commit messages). Use "showcase dataset" / "circular part".

## Toolchain

- **Python: `uv` only.** `uv run …`, `uv add …`, `uv sync`. Never `pip`, `poetry`, `conda`, or a manually
  activated venv. Backend lives in `backend/`; `uv.lock` is committed.
- **Frontend: `bun` only.** `bun install`, `bun run …`, `bunx …`. Never `npm`, `npx`, `yarn`, or `pnpm`.
  Frontend lives in `frontend/`; `bun.lock` is committed.
- **Tauri shell: `cargo`**, under `frontend/src-tauri/`.
- Compute target is **Apple Silicon / MPS** — no CUDA assumptions.
- All documentation, code comments, and identifiers in **English**.

## Where truth lives

- `docs/architecture/` — **the handbook: how the system works now.** One page per area — `README.md`
  (overview + components), `repository.md`, `domain-model.md`, `import.md`, `methods.md`,
  `annotations.md`, `diagnostics.md`, `jobs.md`, `evaluation.md`, `media.md`, `deployment.md`,
  `frontend.md`, `security.md`. **Read this
  first**; it replaced `system-design.md`, which no longer exists. Use its **canonical entity names
  exactly**: `Dataset`, `Channel`, `Sample`, `Image`, `Split`, `SplitAssignment`, `Experiment`, `Job`,
  `ImageResult`, `SampleResult`, `MetricSet`. Pages carry no status and are **edited freely** when the
  code changes — updating one is part of the change, not a follow-up.
- `docs/roadmap.md` — **what the workbench does today, and what is still open.** Status only; no
  milestone history. Read it before starting work.
- `docs/backlog.md` — the open task list, and nothing that has shipped.
- `docs/measurements.md` — **the numbers that still decide something**: each predeclared gate, its
  protocol and its verdict. Cite it rather than restating a figure.
- `docs/adr/` — **24 records, every one of them live** (`docs/adr/README.md` is the index). A record
  captures a choice **that had a live alternative**; the bar is *would a competent engineer plausibly
  have chosen otherwise, and would changing it now cost more than a refactor?* A contract detail, a
  helper, or a read path for something already decided is **handbook material, not a new ADR**.
  **Records are amended** with a dated `## Changelog` entry; only a **reversal** gets a new number
  that supersedes explicitly. A record whose truth has moved into the handbook is **removed**, with
  its citations repointed in the same change (ADR-0030). Numbers are never reused, so a surviving
  record keeps the number it has always had.
- When the handbook and a record disagree, the **handbook is right about what the code does** and the
  **record is right about why it was chosen**.
- **The design tokens live in `@vitavision/lab-ui` (ADR-0021)**, not in this repo: colour, type and
  radius are defined once for every lab app and `frontend/src/styles.css` only imports them (plus the
  one rule that is this app's own — `#root` is a mount point, not a design decision). Components name
  `surface`, `line`, `fg-muted`, `signal`, `normal`, `defect`, `warn` — **never a raw Tailwind ramp
  step** like `slate-500`. A raw colour will compile and look almost right, and quietly ignore the
  theme.

## Current status and working discipline

- **Everything below closes, and the one open piece of the loop is the visual pass** (see
  `docs/roadmap.md`). **N methods can be read against each other**: import a directory tree or a
  public benchmark, browse and label it, annotate it at pixel level, pin an invertible region profile, split it, train, score, read image- and pixel-level
  metrics, browse every scored sample and filter to the model's mistakes, ask the method about any
  image, continue training — then put N runs of one split side by side, find the samples they
  disagree on, open one of them with every method's map in its own pane, and export a fitted method
  as a verified ONNX bundle. Six methods ship: `pixel_reference` (numpy + Pillow, the floor),
  `efficientad_anomalib` and `efficientad_custom` (MPS), `patchcore_anomalib` (a coreset memory
  bank; nothing is trained), `dinomaly_anomalib` (transformer reconstruction) and `glass_anomalib`
  (learned anomaly synthesis). A grouped multi-view dataset is now *usable* and not merely
  representable: a run selects its channels by name, scores are normalized per channel before they are
  aggregated, one annotation covers every channel of a part, and the editor blends two channels to show
  the registration the scan measured.
- **Nothing is compared in score units (ADR-0028).** A score has no meaning outside its own run —
  `pixel_reference` operates around 14 and `efficientad_anomalib` around 0.065 on the same data.
  Threshold-independent metrics compare directly; anything threshold-dependent is resolved **per run
  by one shared rule** whose name and resolved value are printed on screen, and a cut carried between
  runs is a *fraction of each range*, never a value. A single slider over a comparison would be
  wrong in a way that looks exactly like being right.
- **A `dl`-gated test file must be named `test_dl_*.py`.** CI's `Backend (dl extra)` job globs exactly
  that, and a file outside the pattern is collected-and-skipped in the torch-free job and run in no
  job at all. A test that is merely *about* a deep method but needs no torch does not take the prefix.
- **Bound a memory-bank method before it runs, not after.** PatchCore's candidate pool is 5.66 GB on a
  full VisA class and its coreset selection is quadratic in that pool, so `plan_bank` resolves both
  caps and prints the footprint before the pass begins. Anything with that shape gets the same
  treatment.
- **A seed has to reach every random stream, and libraries hide some.** anomalib's coreset draws
  through scikit-learn's numpy RNG, which `torch.manual_seed` does not touch, so its bank is not
  reproducible; M6 found the same shape in torch's global stream for weight init. When adding a
  method, assert reproducibility in *both* directions — same seed identical, different seed different.
- **Controls come from `@vitavision/lab-ui`** — the shared design system for every lab app, not a
  helper extracted from this one; see its README for the consumer wiring. There is an `Input`,
  `NumberInput`, `Textarea`, `Select`, `SegmentedControl`, `Switch`, `Checkbox`, `Slider`, `Table`,
  `Dialog`, `ConfirmDialog`, `Tooltip`, `InfoHint`, `Disclosure`, `Field`, `Badge`, `CountRun`,
  `Empty`, `ErrorBox`, `Callout`, `Skeleton`, `ToggleChip`, `PageHeader`, `Panel`, `Section`,
  `ReadoutStrip`, `Tabs`, `SchemaForm`, the chart set and `ImageStage`. Reach for one before writing
  a bare `<select>`, `<input type="range">`, `<input type="checkbox">` or `<table>` — those are what
  the pass removed. A raw `<details>` in particular renders **with no caret**, because the base layer
  drops the UA marker; use `Disclosure`. **A primitive that needs improving is improved upstream in
  lab-ui**, never patched locally — a local copy is how the apps stop agreeing with each other.
- **One page-level scroller per screen, and the layout owns it.** Three route layouts —
  `ReadingLayout`, `DatasetLayout`, `CanvasLayout` — are marked with `data-layout`, `data-band` and
  `data-scroll` so the contract is assertable, and `frontend/src/routes/dataset/tabScroll.test.tsx`
  asserts it. A nested `max-h-* overflow-y-auto` inside a page that already scrolls is the bug this
  replaced. The one exception is a **peer column** — a rail beside the content, scrolling on its own.
- **A control never nests inside a link.** Card actions and grid selection boxes are
  absolutely-positioned siblings of their `<Link>`, not children. This is correctness, not styling: a
  control inside an anchor has to cancel the click to stop the navigation, and cancelling a
  checkbox's click makes the browser restore its previous state *after* React has written the new
  one, so the tick lands one render late.
- **A method or adapter option needs no frontend work, and that now holds for every pydantic shape.**
  `enum` is read before `type` and `$ref` is resolved through `$defs`, so `Literal` and `StrEnum`
  both become pickers and numeric bounds reach the control. If a new option still needs a change in
  `SchemaForm.tsx`, that is a finding about the schema-to-control mapping, not a place to
  special-case.
- **An empty control means unset, and that contract is load-bearing.** `toOptions` sends nothing for
  a field nobody touched, so a default is defined in Python alone. A segmented control highlights
  the *effective* value and stores `""` when that is the schema default; a select carries an
  explicit `Default · <value>` entry. Do not "fix" this by pre-filling.
- **Schema v1 is frozen.** It was amended in place through M2, as the rule below allowed; the first real
  import has now landed, so every further change is a new numbered migration (ADR-0004).
- **Regenerate `frontend/src/api/generated.ts`** with `scripts/gen-api-types.sh` after any API change; CI
  fails on a stale file.
- **Take work from `docs/backlog.md`, and finish what is open before starting what is new.** The
  roadmap says what stands today and what is still missing; the backlog says what to do about it.
- **A new job kind costs one entry** in `jobs/handlers.py` and one handler function. The queue, the
  JSON-lines protocol, cancellation, log tee-ing and WebSocket fan-out are kind-agnostic; if a new kind
  needs a change in any of them, that is a finding about the boundary. `train` and `infer` cost exactly
  that in M3.
- **A worker's result travels as one JSON line, and a handler tested in-process never proves it.**
  `split_output` flushes an unterminated fragment so no library can wedge the queue, which for years
  quietly cut any event past 16 KiB in half: a 24-entry `region_prepare` preview finished `succeeded`
  carrying `result = {}` and the screen reading it drew nothing beside a green badge. Candidate events now
  get `MAX_EVENT_BYTES`. The discipline that remains: a result whose size grows with the dataset is a
  finding about the result, and a handler exercised only by calling it directly says nothing about whether
  its answer reaches the parent.
- **A new method costs one entry** in `models/registry.py` and one module implementing `AnomalyModel`
  (ADR-0007). It must not need a route, a schema, or a line of TypeScript — the method picker and every
  configuration form are generated from the plugin's own JSON Schema. Keep heavy imports *inside* the
  plugin's functions: the registry is lazy so that opening the method picker does not cost three seconds
  of torch, and that only holds if every module cooperates.
- **Every method loads its pixels through `models/preprocessing.load_array`.** Preprocessing is
  configuration of the *experiment*, not of the model. A method that decodes an image any other way makes
  every comparison against it partly a measurement of its resize.
- **A metric that could not be computed is `None`, and renders as a dash.** Never 0.0. A subset with no
  defects has no ROC-AUC; a fabricated number on a results screen is worse than a visible gap.
- **Bound anything whose cost is linear in the dataset and whose value is not — and say what was
  dropped.** Reference-image counts, quantile-fit samples, per-image diagnostics. Use
  `models.base.evenly_spaced`, never the first N, and log the cap. A silent truncation reads as "this is
  all there was".
- **The deep-learning dependencies live behind the optional `dl` extra.** `pixel_reference`, the whole
  evaluation layer and every test but the `dl`-gated ones must work without torch installed. CI has
  **two** backend jobs for exactly this: `Backend` installs without the extra and is what *measures*
  the torch-free boundary, and `Backend (dl extra)` runs the `test_dl_*.py` files. Run the MPS smoke
  test (`scripts/mps-smoke-test.py`) before trusting the accelerator, and before writing wrapper code
  against a new library (ADR-0008) — it has already paid for itself once.
- **Type-check with bare `uv run mypy`, never `mypy --strict src`.** `pyproject` sets
  `files = ["src", "tests"]`; checking only `src` is how five type errors in a test file reached CI.
- **Exactly one resident inference worker may exist, and a lock is what keeps it off the device
  (ADR-0026).** `ResidentWorker.evict` and a request take the same lock, and `JobQueue` awaits
  `before_spawn` before it starts a worker. A job may therefore be delayed by one in-flight request —
  that delay *is* the guarantee, so the hook must not be made non-blocking. Wiring stays in
  `api/app.py`: the queue must never import the resident.
- **`tests/test_showcase_import.py` runs against the private tree only when
  `ANOMALY_LAB_SHOWCASE_ROOT` is set**, and is skipped everywhere else. It contains no path and no
  directory name; keep it that way.
- **An adapter's JSON Schema drives its import form** — no adapter needs frontend work. Adding an option
  means adding a pydantic field with a `description`; the form renders it, and a field whose default is
  empty is shown while one with a working default is folded away. If a new option needs a change in
  `SchemaForm.tsx`, that is a finding about the schema-to-control mapping, not a place to special-case.
- **Keep the vertical slice honest (ADR-0007).** Adding or changing a method means adding a module and a
  registry entry. If a change for a new method leaks into the jobs, evaluation, results, or UI layers, the
  plugin boundary is wrong — fix the boundary, not the caller.
- **`AGENTS.md` is this file's twin for Codex.** They differ only in the first three lines. A change
  to one of them belongs in both, in the same commit; guidance that holds for one agent and not the
  other does not exist here.
- Prefer the smallest change that satisfies the milestone's exit criteria. Small, understandable
  architecture beats premature generality.
