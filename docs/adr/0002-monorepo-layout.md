# ADR-0002: Monorepo layout

**Status:** Accepted (2026-08-06)

## Context

The workbench spans three toolchains: Python for training and inference, TypeScript/React for the
user interface, and Rust for the Tauri desktop shell. They are developed together by one person,
change together, and are released together as a single desktop application. There is no scenario
in which the frontend ships against an older backend, or a third party consumes either half.

The alternatives were one repository per component, or a monorepo with build orchestration (Nx,
Turborepo, Bazel, Pants). Both buy cross-project caching and dependency graphs that a solo
developer with three components does not need, at the cost of a configuration layer that must
itself be learned and maintained.

## Decision

One Git repository, with a flat top level:

```
backend/     Python service, managed by uv (API, models, jobs, evaluation)
frontend/    React + TypeScript + Vite app; the Tauri shell sits at frontend/src-tauri/
deployment/  The reference runner for exported model bundles (see ADR-0034)
book/        The mdBook user and extension guide, built to gitignored site/
docs/        Handbook, roadmap, backlog, measurements, and these records
scripts/     Dev, safety and probe scripts
data/        Gitignored. Application-managed state (see ADR-0004)
datasets/    Gitignored. Public reference datasets, downloaded (see ADR-0015)
```

Private source data is **not** in the tree at all; it is referenced by absolute path from outside
it (see ADR-0022).

`data/` is **repo-local by default** — a research tool should let you `ls` its state and `rm -rf` it
to start clean — and relocatable via `ANOMALY_LAB_DATA_DIR`.

`src-tauri/` sits inside `frontend/` because the Tauri CLI expects it adjacent to the web app it
wraps, and because the shell is a delivery mechanism for the frontend, not a peer component.

**No monorepo orchestration tooling.** Each stack keeps its native tool — `uv` for Python, `bun`
for the frontend, `cargo` for the Rust crates — and scripts in `scripts/` glue them together.
Lockfiles are committed.

## Consequences

A cross-cutting change (a new API field touching the Python route, the TypeScript client and the
UI) lands in one commit and one reviewable diff. Each toolchain is used the way its documentation
describes, so upstream instructions apply directly.

- **No unified command.** There is no single `build` or `test`; contributors must know which tool
  to invoke where, and the glue scripts drift unless maintained.
- **No dependency graph, no incremental caching.** CI re-runs whole-stack checks rather than only
  what changed. At this size that is minutes; it does not scale.
- **Version coupling is assumed, not checked by layout.** Backend and frontend move in lockstep;
  the generated API client (see ADR-0012) is what catches a contract break.
- **A repo-local `data/` invites accidents.** An aggressive `git clean -xdf` deletes experiment
  results. `ANOMALY_LAB_DATA_DIR` exists partly as the mitigation.
- **Migrating out is a rewrite of the glue.** A component that later needs independent release
  takes every script and relative path that assumes co-location with it.
