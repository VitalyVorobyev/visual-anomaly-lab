# Development and verification

## Setup

```bash
./scripts/setup-hooks.sh
uv sync --directory backend --extra dl
(cd frontend && bun install)
```

Use `uv`, `bun`, and `cargo` only. Lockfiles are committed. Keep all code, identifiers, comments, and
documentation in English.

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
./scripts/build-book.py --check
mdbook build
```

After an API model or route changes, run `scripts/gen-api-types.sh`. CI compares the generated TypeScript
client. A deep-learning test file must be named `test_dl_*.py`; the torch-free and `dl` CI jobs deliberately
measure that dependency boundary.

## Repository safety

Never put private source images inside this working tree, even via symlink. Never stage the top-level
`datasets/` directory. Fixtures are tiny synthetic PNGs. Stage explicit files—never `git add .` or
`git add -A`—then run:

```bash
./scripts/check-repo-safety.sh
```

It must pass before every commit and push.

## Documentation model

- This book is task-oriented user and extension documentation.
- `docs/architecture/` is the current implementation handbook.
- `docs/adr/` records consequential choices and alternatives.
- `docs/roadmap.md` and `docs/backlog.md` track delivery and remaining work.
- `docs/measurements-*.md` are evidence logs.
- `docs/benchmarks/results.json` is checked structured evidence used to generate the book's method and
  benchmark pages.

Run `scripts/build-book.py` after model capability or benchmark-data changes. CI regenerates and refuses a
diff, then builds the whole book. Do not hand-edit files under `docs/book/generated/`.

## Review discipline

Validate behavior in proportion to risk: focused test while iterating, full relevant gates before commit,
and a real application walkthrough for UI or workflow changes. Public benchmark claims require a recorded
protocol, immutable result data, and predeclared decision rule. A screenshot is useful visual evidence, not
an algorithm benchmark.

The compact contributor reference remains at [`docs/development.md`](../development.md).
