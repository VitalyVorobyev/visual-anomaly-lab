# CLAUDE.md

Guidance for Claude Code working in this repository.

`visual-anomaly-lab` is a local desktop workbench (React + TypeScript + Tauri UI, Python FastAPI sidecar,
SQLite + filesystem artifacts) for importing image datasets, training anomaly-detection methods, and
comparing them under one evaluation protocol.

## Steering: the goal is universal

- The target is a **universal anomaly-detection explorer for arbitrary image datasets**. The private
  showcase dataset under `privatedata/` is the first reference dataset, not the scope.
- **Only** the `classical_circular` plugin may assume anything about the showcase dataset's geometry. Domain
  model, import layer, DL methods (`efficientad_anomalib`, `patchcore_anomalib`, `efficientad_custom`),
  evaluation layer, and UI must stay dataset-agnostic.
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
- `docs/adr/` — **12 accepted ADRs (0001–0012)**. Records are immutable once accepted. A significant new
  decision gets a **new numbered ADR** that explicitly supersedes the old one — never silently contradict
  an existing record, and never edit an accepted one in place. Follow the format in `docs/adr/README.md`.

## Current status and working discipline

- **M0 and M1 are done** (repo safety + foundation docs; walking skeleton). The desktop app runs, spawns
  its sidecar and shows live health; the browser path works against a standalone backend. **There are no
  features yet** — M2 (import + browse) is the current milestone.
- **Schema v1 may still be edited in place** (`001_initial.sql`, delete the dev database and re-apply)
  until the first real import lands in M2. After that, migrations are strictly forward-only (ADR-0004).
- **Regenerate `frontend/src/api/generated.ts`** with `scripts/gen-api-types.sh` after any API change; CI
  fails on a stale file.
- **Follow the milestone order** in the roadmap: M1 walking skeleton → M2 import + browse → M3 vertical
  slice on the classical baseline → M4 EfficientAD → M5 PatchCore + comparison → M6 custom EfficientAD →
  M7 polish + full README. Do not build M4 machinery while M2 is unfinished.
- **Keep the vertical slice honest (ADR-0007).** Adding or changing a method means adding a module and a
  registry entry. If a change for a new method leaks into the jobs, evaluation, results, or UI layers, the
  plugin boundary is wrong — fix the boundary, not the caller.
- Prefer the smallest change that satisfies the milestone's exit criteria. Small, understandable
  architecture beats premature generality.
