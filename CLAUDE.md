# CLAUDE.md

Guidance for Claude Code working in this repository.

`visual-anomaly-lab` is a local desktop workbench (React + TypeScript + Tauri UI, Python FastAPI sidecar,
SQLite + filesystem artifacts) for importing image datasets, training anomaly-detection methods, and
comparing them under one evaluation protocol.

## Steering: the goal is universal

- The target is a **universal anomaly-detection explorer for arbitrary image datasets**. The private
  showcase dataset is one reference dataset, not the scope.
- **Only** the `classical_circular` plugin may assume anything about the showcase dataset's geometry. Domain
  model, import layer, DL methods (`efficientad_anomalib`, `patchcore_anomalib`, `efficientad_custom`),
  evaluation layer, and UI must stay dataset-agnostic.
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

- `docs/system-design.md` — architecture, domain model, API surface, job and evaluation protocols. Use its
  **canonical entity names exactly**: `Dataset`, `Channel`, `Sample`, `Image`, `Split`, `SplitAssignment`,
  `Experiment`, `Job`, `ImageResult`, `SampleResult`, `MetricSet`.
- `docs/roadmap.md` — milestones M0–M7 with scope and exit criteria. Check which milestone is current
  before starting work.
- `docs/backlog.md` — task-level breakdown by epic.
- `docs/adr/` — **22 records (0001–0022); 0001 is superseded by 0022**. Records are immutable once accepted. A significant new
  decision gets a **new numbered ADR** that explicitly supersedes the old one — never silently contradict
  an existing record, and never edit an accepted one in place. Follow the format in `docs/adr/README.md`.
- `frontend/src/styles.css` — **the design tokens (ADR-0021)**. Colour, type and radius are defined
  there and nowhere else. Components name `surface`, `line`, `fg-muted`, `signal`, `normal`,
  `defect`, `warn` — **never a raw Tailwind ramp step** like `slate-500`. A raw colour will compile
  and look almost right, and quietly ignore the theme.

## Current status and working discipline

- **M0–M4.6 are done.** The loop closes and its output is reachable: import a directory tree or a
  public benchmark, browse and label it, split it, train, score, and read image- and pixel-level
  metrics — then browse every scored sample as a picture, filter to the model's mistakes, and lay its
  segmentation against the ground truth. Two methods ship: `pixel_reference` (numpy + Pillow, the
  floor) and `efficientad_anomalib` (MPS).
- **M4.7 — the workbench you can iterate in — is next**, taken before M5 because running the
  application showed that training restarts from zero, diagnostics cannot be asked for, and Overview
  opens on a job log. Read `docs/roadmap.md` before starting work; the completed milestones there are
  summaries that keep only what still constrains new work.
- **Controls come from `components/ui`.** There is an `Input`, `NumberInput`, `Select`,
  `SegmentedControl`, `Switch`, `Checkbox`, `Slider`, `Table`, `Dialog`, `Tooltip`, `Disclosure`,
  `Skeleton`, `ToggleChip`, `PageHeader`, `Section` and `ReadoutStrip`. Reach for one before writing a bare
  `<select>`, `<input type="range">` or `<table>` — those are what the pass removed. A raw
  `<details>` in particular now renders **with no caret**, because the base layer drops the UA
  marker; use `Disclosure`.
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
- **Follow the milestone order** in the roadmap: M1 walking skeleton → M2 import + browse → M3 vertical
  slice on the classical baseline → M4 EfficientAD → M5 PatchCore + comparison → M6 custom EfficientAD →
  M7 polish + full README. Do not build M4 machinery while M3 is unfinished.
- **A new job kind costs one entry** in `jobs/handlers.py` and one handler function. The queue, the
  JSON-lines protocol, cancellation, log tee-ing and WebSocket fan-out are kind-agnostic; if a new kind
  needs a change in any of them, that is a finding about the boundary. `train` and `infer` cost exactly
  that in M3.
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
  evaluation layer and every test but the EfficientAD ones must work without torch installed. Run the
  MPS smoke test (`scripts/mps-smoke-test.py`) before trusting the accelerator, and before writing wrapper
  code against a new library (ADR-0008) — it has already paid for itself once.
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
- Prefer the smallest change that satisfies the milestone's exit criteria. Small, understandable
  architecture beats premature generality.
