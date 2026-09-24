---
name: safe-commit
description: Commit or push in visual-anomaly-lab without leaking private data or leaving CI red. Use before every `git commit`, `git push` or PR in this repository — explicit staging, the repo-safety script, the CLAUDE.md/AGENTS.md twin rule, regenerated API types, the same checks CI runs, and a commit message that never names the showcase product.
---

# Safe commit

Every step here exists because skipping it once cost something. Run them in order; stop at the
first failure and fix it rather than working around it.

## 1. Look at what changed

```bash
git status --short
git diff --stat
```

Anything you did not mean to touch — `docs/dino3-methods.md`, `papers/`, a scratch file, a
screenshot — stays unstaged. Screenshots and scratch output belong in the session scratchpad, never
in the tree.

## 2. Stage explicit paths

```bash
git add path/one path/two          # never `git add -A`, never `git add .`
```

The private showcase images live outside the tree (ADR-0022), and explicit staging is the second
wall that keeps them out. Nothing under `/datasets/` is ever staged — the public reference data is
gitignored for size.

## 3. Keep the generated and twinned files in step

| If you changed…                                    | then also…                                                          |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| a route, a request/response model, an enum the API returns | `scripts/gen-api-types.sh` and stage `frontend/src/api/generated.ts` |
| `CLAUDE.md`                                        | the same edit in `AGENTS.md` (they differ only in lines 1–3)          |
| how a subsystem works                              | its page in `docs/architecture/` in the **same** commit              |
| something that shipped from `docs/backlog.md`      | delete the item; the handbook describes it now                       |
| a measured verdict                                 | `docs/measurements.md`, cited rather than restated elsewhere          |
| the database schema                                | a **new** numbered migration — schema v1 is frozen (ADR-0004)         |

Check the twins with `diff <(tail -n +4 CLAUDE.md) <(tail -n +4 AGENTS.md)` — it must print nothing.

## 4. Run what CI runs, for the parts you touched

Backend (from `backend/`):

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

Bare `uv run mypy` — `pyproject` checks `src` **and** `tests`; `mypy --strict src` is how type
errors in tests reached CI. A test that needs torch is named `test_dl_*.py`, or no CI job runs it.

Frontend (from `frontend/`):

```bash
bun run typecheck && bun run test && bunx vite build
```

Docs: `uv run --directory backend python ../scripts/check-doc-links.py` and
`uv run --directory backend python ../scripts/build-book.py --check` when `docs/` or `book/` changed.
Rust, when it changed: `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` under
`frontend/src-tauri/` (the shell) or with `--manifest-path deployment/runner/Cargo.toml` (the
handoff runner), then `uv run --directory backend pytest tests/test_rust_deployment_handoff.py`.
Clippy runs with `-D warnings` in CI, so a new lint in a toolchain update fails the build.

## 5. The safety gate

```bash
scripts/check-repo-safety.sh      # must print "Repo safety check passed." and exit 0
```

It inspects the index: no `privatedata`, nothing under `datasets/`, no camera or array formats, no
PNG/SVG over 256 KB. `scripts/setup-hooks.sh` installs it as a pre-commit hook — run it once per
clone — but invoke it explicitly anyway; a hook can be bypassed, a habit cannot.

## 6. Write the message

- The subject says **what is now true**, as a sentence ("The threshold chosen on Overview reaches
  the Samples tab"), not what you did ("fix threshold bug").
- The body says why, and what a reviewer should look at.
- **Never name or describe the showcase dataset's product.** Say "showcase dataset" or "circular
  part". This holds for the branch name and PR text too.
- End with the attribution line the session's system reminder gives.

## 7. Push and PR

Branch off `main` first if you are on it. `gh pr create` with a body that lists the verification
you ran and anything deliberately left out. Wait for `gh pr checks <n>` to be green before merging;
merge only when the user has said you may.
