# Repository structure

```
visual-anomaly-lab/
├── README.md                       # setup, dataset conventions, how to add a method
├── .gitignore                      # data/, *.bmp, venvs, build output (and privatedata/, belt and braces)
├── scripts/check-repo-safety.sh    # pre-push guard: fails if private data is staged (ADR-0001)
├── docs/
│   ├── architecture/               # the handbook — how the system works now (this document)
│   ├── adr/                        # decision records: why it is shaped this way
│   ├── roadmap.md                  # milestones M0–M10
│   ├── backlog.md                  # task breakdown by epic
│   ├── measurements-efficientad.md # the append-only evidence log behind ADR-0029
│   ├── development.md              # how to run, test and check the thing
│   └── papers.md                   # method references
│
├── backend/                        # Python, uv-managed
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/anomaly_lab/
│   │   ├── api/                    # FastAPI app factory, routers, websockets, schemas
│   │   ├── domain/                 # pydantic entities, enums — no I/O
│   │   ├── db/                     # SQL migrations (NNN_*.sql), connection, repositories
│   │   ├── datasets/               # import adapters, manifest model, scan/verify
│   │   ├── media/                  # BMP decode, thumbnail/preview cache, map rendering
│   │   ├── models/                 # base.py (interface), classical/, anomalib_adapters/, registry
│   │   ├── model_assets/           # fixed catalogue, integrity checks, licensed downloads
│   │   ├── jobs/                   # queue, subprocess worker entrypoint, event protocol
│   │   └── eval/                   # metrics, channel→sample aggregation, thresholds
│   └── tests/                      # pytest: unit + API-level with a temp data dir
│
├── frontend/                       # React + TypeScript + Vite
│   ├── package.json, bun.lock, vite.config.ts, tsconfig.json
│   ├── src/                        # api client, hooks, screens, components
│   └── src-tauri/                  # Rust desktop shell: sidecar spawn, port handoff, teardown
│       ├── Cargo.toml, tauri.conf.json
│       └── src/main.rs
│                                # NOTE: source images are NOT here. They live outside this
│                                # tree entirely and are reached by absolute path (ADR-0022).
│
└── data/                           # GITIGNORED — all app-managed state
    ├── app.sqlite3                 # metadata, scores, paths
    ├── manifests/                  # committed import manifests (reproducibility)
    ├── thumbnails/
    │   ├── thumb/                  # 256 px WebP
    │   └── preview/                # 1024 px WebP
    ├── artifacts/
    │   └── exp-<id>/
    │       ├── checkpoints/        # model weights / memory banks
    │       ├── references/         # classical baseline per-channel reference statistics
    │       ├── maps/               # float32 .npy anomaly maps
    │       └── logs/               # <job>.log — full worker stdout stream
    ├── annotations/
    │   └── image-<id>/revision-<n>.png # immutable app-owned binary truth
    ├── model-cache/
    │   └── assets/                 # verified shared weights + external-source metadata
    └── exports/                    # CSV / JSON exports of results and metrics
```

Monorepo layout (ADR-0002): one repository, two build systems, no shared build tooling — `uv` owns
`backend/`, `bun` + `cargo` own `frontend/`. The two halves are coupled only by the HTTP contract.

**Data directory resolution.** `data/` is repo-local by default so a fresh clone works with zero configuration
and everything stays inside the ignored tree. It is overridable via the `ANOMALY_LAB_DATA_DIR` environment
variable — needed for tests (temp dir per test), for packaged builds (OS app-data directory), and for putting
artifacts on an external disk. All backend code resolves paths through a single settings object; no module
constructs a path from `__file__` or the current working directory.

**Storage split.** SQLite stores *metadata, configuration, scores and paths only*. Pixel data — source images,
anomaly maps, thumbnails, checkpoints — always lives on the filesystem, referenced by path. This keeps the
database small enough to be trivially inspectable with `sqlite3`, keeps large binaries out of transactions,
and lets artifacts be deleted or archived by directory (ADR-0004).

**Model assets are executable inputs, not casual downloads.** `model_assets/catalog.py` pins every accepted
asset to an immutable upstream revision, exact byte count, SHA-256 and licence. Acquisition streams to a
job-specific partial file, reports progress, honours cancellation, verifies size and digest, then atomically
renames into `model-cache/assets/`. A user may instead select an external file; it must pass the same checks,
is recorded by absolute path, and is never copied or deleted by the application. Listing the catalogue hashes
each distinct `(path, size, mtime)` state once, so the UI can poll without repeatedly reading tens of MB.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
