# Repository structure

```
visual-anomaly-lab/
├── README.md, CLAUDE.md, AGENTS.md  # setup and dataset credits; agent guidance (twins)
├── .gitignore                      # data/, /datasets/, /papers/, *.bmp, build output, site/
├── scripts/                        # dev-*.sh, check-repo-safety.sh (ADR-0022), gen-api-types.sh,
│                                   # build-book.py, check-doc-links.py, smoke tests, public gates
├── book/                           # mdBook user and extension guide
│   ├── book.toml                   # builds book/src into the gitignored site/
│   └── src/                        # SUMMARY.md, workflow and extension chapters
├── docs/
│   ├── architecture/               # the handbook — how the system works now
│   ├── adr/                        # decision records — why it is shaped this way
│   ├── roadmap.md                  # what works today and what is open
│   ├── backlog.md                  # the open task list
│   ├── measurements.md             # predeclared gates, protocols, verdicts
│   ├── papers.md                   # reading list
│   └── benchmarks/results.json     # public benchmark results rendered into the book
│
├── backend/                        # Python, uv-managed (pyproject.toml, uv.lock)
│   ├── src/anomaly_lab/
│   │   ├── api/                    # app factory, routers, websockets; errors.py maps refusals to statuses
│   │   ├── annotations/service.py  # draft lifecycle, revisions, scope, interchange
│   │   ├── experiments/            # service.py (create, preconditions, deletion); train/infer/diagnose work
│   │   ├── errors.py               # domain refusals: NotFound, Conflict, StaleVersion, InvalidInput, …
│   │   ├── domain/                 # pydantic entities and enums — no I/O
│   │   ├── db/                     # SQL migrations (NNN_*.sql), connection + transaction(), repositories
│   │   ├── datasets/               # import adapters, manifest model, scan/commit/verify, reference packs
│   │   ├── regions/                # region extractors, transforms, prepared-image builds
│   │   ├── media/                  # decode, thumbnail/preview cache, map rendering, prewarm
│   │   ├── models/                 # base.py (interface), registry.py, one module per method
│   │   ├── model_assets/           # fixed catalogue, integrity checks, licensed downloads
│   │   ├── deployment/             # ONNX bundle schema, export and parity
│   │   ├── jobs/                   # queue, worker entrypoint, event protocol, resident worker
│   │   └── eval/                   # metrics, channel→sample aggregation, thresholds
│   ├── research/                   # sweeps run outside the app (ADR-0038); ruff- and mypy-checked, never packaged
│   └── tests/                      # pytest: unit + API-level with a temp data dir; test_dl_*.py need torch
│
├── frontend/                       # React + TypeScript + Vite (package.json, bun.lock)
│   ├── src/                        # api client (generated.ts), hooks, routes, components
│   └── src-tauri/                  # Rust desktop shell: sidecar spawn, port handoff, teardown
├── deployment/runner/              # Rust reference runner for ONNX bundles
│
├── datasets/                       # GITIGNORED — public reference datasets (ADR-0015)
└── data/                           # GITIGNORED — all app-managed state
    ├── app.sqlite3                 # metadata, scores, paths
    ├── manifests/                  # committed import manifests (dataset-<id>-*.json)
    ├── thumbnails/{thumb,preview}/ # 256 px and 1024 px WebP
    ├── artifacts/exp-<id>/         # method state, maps/ (float32 .npy), logs/<job>.log, exports/
    ├── jobs/logs/                  # logs of jobs that belong to no experiment
    ├── annotations/image-<id>/     # revision-<n>.png — immutable app-owned binary truth
    ├── region-profiles/profile-<id>/build/  # lossless prepared PNGs + transforms
    └── model-cache/assets/         # verified shared weights + external-source metadata
```

Source images are not in this tree. They live outside the working directory and are reached by absolute
path (ADR-0022), so `git add` cannot reach them.

Monorepo layout (ADR-0002): one repository, two build systems, no shared build tooling — `uv` owns
`backend/`, `bun` + `cargo` own `frontend/`. The two halves are coupled only by the HTTP contract.

## Data directory

`data/` is repo-local by default so a fresh clone works with zero configuration. `ANOMALY_LAB_DATA_DIR`
overrides it — for tests (a temp dir each), packaged builds (which have no checkout to infer a root from
and must set it) and external disks. All backend code resolves paths through the one `Settings` object; no
module builds a path from `__file__` or the working directory.

## Storage split

SQLite stores *metadata, configuration, scores and paths only*. Pixel data — source images, anomaly maps,
thumbnails, checkpoints — lives on the filesystem, referenced by path, which keeps the database small,
large binaries out of transactions, and artifacts deletable by directory (ADR-0004).

## Transactions

Connections open in autocommit mode, and every multi-statement write goes through
`db.connection.transaction(conn, immediate=...)`: it commits when the block finishes, rolls back on any
exception, then runs the rollback callbacks registered on it. `immediate=True` takes the write lock up
front, which a read-check-write needs (a draft's version check, a deletion's blocker check). Because the
filesystem cannot join a transaction, a file written inside one (a completed annotation's mask, an accepted
manifest) is registered with `tx.remove_on_rollback(path)` and removed with the rows that would have
referenced it. Nothing else issues `BEGIN`, `COMMIT` or `ROLLBACK`, except the migration runner, whose
`executescript` carries its own transaction.

## Model assets

Model assets are executable inputs, not casual downloads. `model_assets/catalog.py` pins every accepted
asset to an immutable upstream revision, exact byte count, SHA-256 and licence. Acquisition streams to a
job-specific partial file, reports progress, honours cancellation, verifies size and digest, then
atomically renames into `model-cache/assets/`. A user may instead select an external file; it passes the
same checks, is recorded by absolute path, and is never copied or deleted by the application. Listing the
catalogue hashes each distinct `(path, size, mtime)` state once, so the UI can poll cheaply.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
