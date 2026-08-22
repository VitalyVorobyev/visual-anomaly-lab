# Development

Contributor reference for Visual Anomaly Lab. Users should start with the [README](../README.md) and
[book](book/introduction.md); extension walkthroughs live in
[Add a model](book/add-model.md), [Add preprocessing](book/add-preprocessing.md), and
[Add an import adapter](book/add-adapter.md).

## Setup and checks

Use `uv` for Python, `bun` for the frontend, and `cargo` for Rust. Lockfiles are committed.

```bash
./scripts/setup-hooks.sh
uv sync --directory backend --extra dl
(cd frontend && bun install)

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
mdbook build
```

Run `scripts/gen-api-types.sh` after changing an API route or response. Run
`scripts/mps-smoke-test.py` before trusting a new accelerator/library path. A torch-dependent test file must
be named `test_dl_*.py`; CI's optional-dependency job selects that exact pattern.

## Safety

Private source images live outside this repository and must never be read for casual inspection, copied,
symlinked, or used as fixtures. Public reference packs stay under the gitignored top-level `/datasets/`.
Fixtures are tiny synthetic PNGs.

Stage explicit paths—never `git add .` or `git add -A`—then run `scripts/check-repo-safety.sh` before every
commit and push. The guard rejects private/public dataset paths, model artifacts, and oversized files.

## Working contracts

- The canonical current system description is [`docs/architecture/`](architecture/README.md). Update the
  relevant page in the same change as code.
- ADRs record choices with live alternatives; amend an existing record for refinements and create a new one
  only for a consequential reversal/new decision.
- Method and adapter forms come from Pydantic JSON Schema. A new option should not require TypeScript.
- New methods stay behind `AnomalyModel`; heavy imports remain inside functions; pixels come through
  `load_array`; ground truth stays in evaluation.
- Channel count is data, never schema. Missing metrics are `None`, never fabricated zeroes.
- Bound dataset-linear work before execution, sample evenly, and report what was dropped.
- Frontend colours come from `frontend/src/styles.css`; controls come from `components/ui`.
- An empty schema control means unset so Python remains the only default authority.

## Evidence and docs

The evidence log at `docs/measurements.md` contains full protocols and limitations. Checked plot/table
inputs live in `docs/benchmarks/results.json`. Run `scripts/build-book.py`; do not edit
`docs/book/generated/` by hand. CI regenerates, rejects drift, and builds the mdBook.

Public quality gates are reproducible and write to isolated application directories:

```bash
./scripts/dinomaly-public-gate.py --data-dir /tmp/dinomaly-public-gate
./scripts/glass-public-gate.py --data-dir /tmp/glass-public-gate
./scripts/region-value-gate.py --data-dir /tmp/region-gate
```

The destination must be absent or empty. Source datasets remain read-only.
