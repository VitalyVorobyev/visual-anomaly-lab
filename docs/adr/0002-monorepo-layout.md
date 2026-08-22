# ADR-0002: Monorepo layout

**Status:** Accepted (2026-08-06)

## Context

The workbench spans three toolchains: Python for training and inference, TypeScript/React for the
user interface, and Rust for the Tauri desktop shell. They are developed together by one person,
change together, and are released together as a single desktop application. There is no scenario
in which the frontend ships against an older backend, or a third-party consumes either half.

The brief is explicit that this is a research prototype and asks us to "prefer a small,
understandable architecture over premature scalability". Multi-repo layouts, or a monorepo with
build orchestration (Nx, Turborepo, Bazel, Pants), would buy cross-project caching and dependency
graphs that a solo developer with three top-level components does not need, at the cost of a
configuration layer that must itself be learned and maintained.

## Decision

One Git repository, with a flat top level:

```
backend/     Python service, managed by uv (FastAPI app, models, jobs, evaluation)
frontend/    React + TypeScript + Vite app, with the Tauri Rust shell embedded at
             frontend/src-tauri/
docs/        Design docs, roadmap, backlog, and these ADRs
scripts/     Dev and safety scripts (dev launcher, check-repo-safety.sh)
privatedata/ Gitignored. Source images, referenced in place (see ADR-0022)
data/        Gitignored. Application-managed state (see ADR-0004)
```

`data/` holds `app.sqlite3`, `manifests/`, `thumbnails/`, `artifacts/`, and `exports/`. It is
**repo-local by default** — a research tool should let you `ls` its state and `rm -rf` it to start
clean — and relocatable via the `ANOMALY_LAB_DATA_DIR` environment variable for larger disks or
packaged installs.

`src-tauri/` sits inside `frontend/` rather than at the top level because the Tauri CLI expects it
adjacent to the web app it wraps, and because the shell is a delivery mechanism for the frontend,
not a peer component.

**No monorepo orchestration tooling.** Each stack keeps its native tool — `uv` for Python, `bun`
for the frontend, `cargo` for the shell — and shell scripts in `scripts/` glue them together for
common workflows (start backend + frontend in dev, run all checks). Lockfiles (`uv.lock`,
`bun.lock`) are committed.

## Consequences

Cross-cutting changes (a new API field touching the Python route, the TypeScript client, and the
UI) land in a single commit and a single reviewable diff. Each toolchain is used the way its
documentation describes, so upstream instructions apply directly and onboarding cost is limited to
tools the developer already knows.

Negative consequences, accepted honestly:

- **No unified command.** There is no single `build` or `test` that covers everything; the dev
  scripts are hand-rolled, undocumented outside `scripts/`, and will drift from reality unless
  maintained. Contributors must know which tool to invoke where.
- **No dependency graph, no incremental caching.** CI, when added, will re-run whole-stack checks
  rather than only what changed. At this size that is a few minutes, but it does not scale.
- **Version coupling is implicit.** Because backend and frontend are assumed to move in lockstep,
  nothing detects an accidental API contract break at build time — only at runtime (see ADR-0003).
- **A repo-local `data/` invites accidents.** Runtime state lives next to source, so an aggressive
  `git clean -xdf` will delete experiment results. The `ANOMALY_LAB_DATA_DIR` override exists
  partly as a mitigation for anyone who wants that separation.
- **Migrating out is a rewrite of the glue.** If any component later needs independent release,
  the scripts and relative paths that assume co-location must all be revisited.
