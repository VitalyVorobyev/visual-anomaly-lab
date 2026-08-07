# CLAUDE.md

Guidance for Claude Code working in this repository.

`visual-anomaly-lab` is a local desktop workbench (React + TypeScript + Tauri UI, Python FastAPI sidecar,
SQLite + filesystem artifacts) for importing image datasets, training anomaly-detection methods, and
comparing them under one evaluation protocol.

## Steering: the goal is universal

- The target is a **universal anomaly-detection explorer for arbitrary image datasets**. The private
  showcase dataset under `privatedata/` is one reference dataset, not the scope.
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

- **Never read, open, copy, move, or commit anything under `privatedata/`.** It is a read-only mount of
  proprietary images, referenced in place. Do not sample it "just to check the format".
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
- `docs/adr/` — **16 accepted ADRs (0001–0016)**. Records are immutable once accepted. A significant new
  decision gets a **new numbered ADR** that explicitly supersedes the old one — never silently contradict
  an existing record, and never edit an accepted one in place. Follow the format in `docs/adr/README.md`.

## Current status and working discipline

- **M0, M1 and M2 are done, and M3's import layer is done.** The app imports a directory tree — or a public
  benchmark, through `folder_classes` or `csv_table` — into a catalog of grouped samples with masks and an
  optional published split; browses and labels them in bulk; views one sample across its channels; and
  creates splits either seeded or adopted from the source. The ADR-0009 job machinery was built in M2, so
  M3 adds `train` and `infer` by writing one handler each.
- **M3 is the current milestone, and it was re-aimed after M2 (ADR-0015):** the vertical slice now runs on
  **EfficientAD via anomalib** plus a dataset-agnostic `pixel_reference` floor baseline, not on
  `classical_circular`, which moved to an optional M8. What remains in M3 is the model plugin layer, the
  two job handlers, the diagnostics contract, the methods themselves, pixel-level evaluation, and the
  results UI. Check `docs/roadmap.md` — it is current.
- **Schema v1 is frozen.** It was amended in place through M2, as the rule below allowed; the first real
  import has now landed, so every further change is a new numbered migration (ADR-0004).
- **Regenerate `frontend/src/api/generated.ts`** with `scripts/gen-api-types.sh` after any API change; CI
  fails on a stale file.
- **Follow the milestone order** in the roadmap: M1 walking skeleton → M2 import + browse → M3 vertical
  slice on the classical baseline → M4 EfficientAD → M5 PatchCore + comparison → M6 custom EfficientAD →
  M7 polish + full README. Do not build M4 machinery while M3 is unfinished.
- **A new job kind costs one entry** in `jobs/handlers.py` and one handler function. The queue, the
  JSON-lines protocol, cancellation, log tee-ing and WebSocket fan-out are kind-agnostic; if a new kind
  needs a change in any of them, that is a finding about the boundary.
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
