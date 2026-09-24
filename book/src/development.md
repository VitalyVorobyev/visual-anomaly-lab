# Development and verification

## Setup

```bash
./scripts/setup-hooks.sh
uv sync --directory backend --extra dl
(cd frontend && bun install)
```

Use `uv` for Python, `bun` for the frontend and `cargo` for Rust, nothing else. Lockfiles are
committed. Code, identifiers, comments and documentation are in English.

## Standard gates

```bash
uv run --directory backend ruff check .
uv run --directory backend ruff format --check .
uv run --directory backend mypy
uv run --directory backend pytest
(cd frontend && bun run typecheck && bun run test && bunx vite build)
cargo fmt --manifest-path frontend/src-tauri/Cargo.toml --check
cargo clippy --manifest-path frontend/src-tauri/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path deployment/runner/Cargo.toml
uv run --directory backend python ../scripts/build-book.py --check
uv run --directory backend python ../scripts/check-doc-links.py
mdbook build book
```

- After an API model or route changes, run `scripts/gen-api-types.sh`. CI compares the generated
  TypeScript client.
- Run `scripts/mps-smoke-test.py` before trusting a new accelerator or library path.
- A deep-learning test file must be named `test_dl_*.py`. The torch-free and `dl` CI jobs
  deliberately measure that dependency boundary.

## Repository safety

- Private source images live outside this working tree. Never read them for casual inspection, and
  never copy them, symlink them or use them as fixtures.
- Public reference packs stay under the gitignored top-level `/datasets/`.
- Fixtures are tiny synthetic PNGs.
- Stage explicit files, never `git add .` or `git add -A`, then run:

```bash
./scripts/check-repo-safety.sh
```

It must pass before every commit and push.

## Working contracts

- **Current internals.** The handbook in `docs/architecture/` describes them, and is updated in
  the same change as the code.
- **Records.** A choice that had a live alternative gets a record in `docs/adr/`.
- **Forms.** Method and adapter forms come from Pydantic JSON Schema, so a new option needs no
  TypeScript.
- **Methods.**
  - A method stays behind the plugin interface.
  - Heavy imports stay inside functions.
  - Pixels come through `load_array`.
  - Ground truth reaches a method only through the training context's targets.
- **Data.** Channel count is data, never schema. A missing metric is `None`, never a fabricated zero.
- **Dataset-linear work.** Bound it before it runs, sample it evenly, and report what was dropped.
- **Frontend.**
  - Colours and controls come from `@vitavision/lab-ui`.
  - An empty schema control means unset, so Python stays the only authority on defaults.

## Documentation model

| Where | What |
|---|---|
| `book/` | This task-oriented user and extension guide |
| `docs/architecture/` | The current implementation handbook |
| `docs/adr/` | Consequential choices and their alternatives |
| `docs/roadmap.md`, `docs/backlog.md` | What stands and what remains |
| `docs/measurements.md` | Gate protocols and verdicts |
| `docs/benchmarks/results.json` | Checked evidence that the method and benchmark pages are generated from |

Run `scripts/build-book.py` after a change to model capabilities or benchmark data. Never hand-edit
`book/src/generated/`: CI regenerates it and refuses a diff.

Public quality gates are reproducible and write to isolated application directories:

```bash
./scripts/dinomaly-public-gate.py --data-dir /tmp/dinomaly-public-gate
./scripts/glass-public-gate.py --data-dir /tmp/glass-public-gate
./scripts/region-value-gate.py --data-dir /tmp/region-gate
```

The destination must be absent or empty, and source datasets stay read-only.

## Review discipline

- Validate in proportion to risk:
  - a focused test while iterating
  - the full relevant gates before a commit
  - a real walkthrough of the application for a UI or workflow change
- A public benchmark claim needs a recorded protocol, immutable result data and a predeclared
  decision rule.
