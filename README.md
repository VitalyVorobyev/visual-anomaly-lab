# visual-anomaly-lab

**A universal, local-first anomaly-detection explorer for image datasets — import your images, train
several methods on them, and compare the results under one evaluation protocol.**

---

## Why

Visual anomaly detection has plenty of published methods and very few honest, side-by-side answers about
which one works on *your* images. Papers report numbers on public benchmarks; real inspection datasets
look nothing like them. `visual-anomaly-lab` is a desktop workbench for closing that gap: point it at a
folder of images, and train, score, and compare a classical baseline against modern deep methods on the
same samples, the same split, and the same metrics.

Two commitments shape the whole design:

- **Universal by intent.** The goal is an anomaly-detection explorer for arbitrary image datasets. The
  domain model, import layer, deep-learning methods, evaluation layer, and UI are dataset-agnostic —
  the number of acquisition channels a dataset has, for example, is per-dataset *data*, never a constant
  in the code. The project is developed against a private industrial inspection dataset (multi-illumination
  images of a manufactured part), which serves as the **first showcase dataset**: an initial focus that
  keeps the design honest against real data, not the final scope.
- **Local and private by design.** Everything runs on one machine — no cloud, no accounts, no telemetry.
  Source images are read in place and never copied into the repository.

## What you'll be able to do

The product is one loop, end to end:

| Stage | Capability |
| --- | --- |
| **Import** | Point the app at a directory; review a proposed manifest (detected channels, grouping, warnings) before anything is written; commit it into the catalog. |
| **Label & split** | Mark logical samples normal / defect / unlabeled; create seeded, sample-level train/val/test splits so no two views of the same object straddle a subset. |
| **Train** | Create an experiment (dataset + split + method + config), run it as an async job, and watch live progress and streaming logs. |
| **Infer** | Score samples with a trained experiment, producing per-image scores and anomaly maps. |
| **Evaluate** | Threshold-independent metrics, an interactive threshold slider driving a live confusion matrix, FP/FN lists, and ranked most-normal / most-anomalous views. |
| **Compare** | Several experiments side by side under one protocol — ROC curves overlaid, timing, config diff, and where the methods disagree. |

Throughout: **grouped multi-view samples** (one physical object, many images, one label), **anomaly-map
overlays** with an opacity slider, and **experiments that persist** — close the app, reopen it, and a past
experiment comes back with identical configuration, metrics, results, and logs.

> ### Project status — walking skeleton
>
> **M0** (repository safety guards + foundation documentation) and **M1** (walking skeleton) are complete:
> the desktop app opens, spawns its Python sidecar, and displays live backend health over HTTP, with a
> WebSocket echo proving the streaming path. The same UI runs in a plain browser against a standalone
> backend — see [Getting started](#getting-started).
>
> **There are no features yet.** Importing a dataset starts at **M2**; training and evaluating a method
> starts at **M3**. Everything under [Methods](#methods) below describes the designed system, not shipped
> software. See [`docs/roadmap.md`](docs/roadmap.md) for the milestone sequence and honest sizing.

## Methods

All methods sit behind one plugin interface and are selected by a stable registry key. Adding a method
means adding a module and a registry entry — nothing else in the application changes.

| Registry key | What it is | Scope |
| --- | --- | --- |
| `classical_circular` | Geometry-aware classical baseline tailored to the showcase dataset: circle fit → polar unwrap → orientation alignment → robust per-channel reference comparison (median/MAD z-map). Fast, CPU-only, interpretable. | **Showcase-dataset-specific** (circular parts) |
| `efficientad_anomalib` | EfficientAD via Intel's [anomalib](https://github.com/open-edge-platform/anomalib) | Dataset-agnostic |
| `patchcore_anomalib` | PatchCore via anomalib | Dataset-agnostic |
| `efficientad_custom` | From-scratch EfficientAD reimplementation, for direct comparison against the library version | Dataset-agnostic |

The classical baseline is the *only* component allowed to assume anything about the showcase dataset's
geometry. Everything else must work on a dataset it has never seen.

## Architecture at a glance

- **Desktop app:** React + TypeScript + Vite UI running inside a **Tauri** shell. The shell is thin — it
  spawns the backend, hands over the port, and tears it down on exit.
- **Backend:** a **Python FastAPI sidecar** (HTTP + WebSocket) bound to `127.0.0.1`. It has no Tauri
  dependency, so the same UI also runs in a plain browser against a manually started backend — which is
  how most development and debugging actually happens.
- **Jobs:** training, inference, and import each run as a subprocess worker off a single FIFO queue,
  streaming progress events to the UI over WebSocket, with cancellation and crash recovery.
- **Storage:** SQLite for metadata, scores, and paths; filesystem artifacts (checkpoints, anomaly maps,
  thumbnails, logs) — all under a gitignored `data/` directory.
- **Compute target:** Apple Silicon / MPS. No CUDA assumption, no cluster, one job at a time.
- **Toolchain:** [`uv`](https://docs.astral.sh/uv/) for Python, [`bun`](https://bun.sh/) for the frontend,
  `cargo` for the Tauri shell. Lockfiles are committed.

Full detail — domain model, API surface, job protocol, evaluation protocol — is in
[`docs/system-design.md`](docs/system-design.md).

## Private data

> **Warning.** `privatedata/` holds proprietary source images that must never be committed or pushed.
> `.gitignore` excludes `privatedata/` and all `*.bmp` / `*.BMP` files as defense in depth, and
> `scripts/check-repo-safety.sh` fails if anything private — or any oversized file — is staged or tracked.
> **Run it before every commit and push.** Stage explicit paths; never `git add -A`.

Images are referenced **in place** and treated as a read-only mount: the application decodes them for
display and training and never writes to that tree, never copies them into a tracked path, and never
sends them anywhere off the machine.

### The showcase dataset

The first dataset is private: multi-illumination photographs of a manufactured circular part, with
**98 normal + 91 defect logical samples**. Each sample is captured under several illumination channels
(bright-field / dark-field / dome) — but one group has only two, a small irregularity with a large design
consequence: **channel count is data, not schema**, and no component may hard-code it. The dataset carries
no pixel masks, so evaluation is image-level, with sample-level ROC-AUC as the headline metric.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/system-design.md`](docs/system-design.md) | Architecture, domain model, canonical terminology, API surface, job and evaluation protocols |
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones M0–M7, scope, exit criteria, sizing |
| [`docs/backlog.md`](docs/backlog.md) | Task-level breakdown by epic |
| [`docs/adr/`](docs/adr/) | Architecture decision records 0001–0011 — what was decided, why, and what it costs |

## Getting started

### Prerequisites

[`uv`](https://docs.astral.sh/uv/) for Python, [`bun`](https://bun.sh/) for the frontend, and a Rust
toolchain for the desktop shell. On macOS the Xcode command line tools are also needed
(`xcode-select --install`). Nothing else — `uv` fetches its own Python 3.12.

### First run

```bash
git clone git@github.com:VitalyVorobyev/visual-anomaly-lab.git
cd visual-anomaly-lab

./scripts/setup-hooks.sh          # private-data guard runs on every commit
uv sync --directory backend
cd frontend && bun install && cd ..

./scripts/dev-app.sh              # the full desktop app
```

The window opens once the sidecar reports ready and shows its version, schema version and database path.

### The browser workflow

The backend has no dependency on the desktop shell, and **the browser path is first class** — it is how
debugging actually happens. In two terminals:

```bash
./scripts/dev-backend.sh          # FastAPI on :8000, with reload
./scripts/dev-frontend.sh         # Vite on :5173
```

Then open <http://localhost:5173>. With no shell to inject a base URL, the app falls back to
`http://127.0.0.1:8000`. The API is equally usable from `curl` or pytest; interactive docs are at
<http://127.0.0.1:8000/docs>.

### How the two processes fit together

The shell starts the sidecar with `ANOMALY_LAB_PORT=0`, so the OS picks a free port and the **sidecar
announces it back** on stdout; the shell then injects that URL into the page. On exit it signals the
child's process group, and the sidecar independently exits if its parent disappears — so a crash or a
force-quit leaves no stray Python process. Data lives in a gitignored `data/`, relocatable with
`ANOMALY_LAB_DATA_DIR`; deleting that directory resets the application.

### Checks

```bash
uv run --directory backend pytest         # backend tests
uv run --directory backend ruff check .   # lint
uv run --directory backend mypy           # types, strict
cd frontend && bun run test && bun run typecheck
./scripts/check-repo-safety.sh            # never commit private data (ADR-0001)
```

After changing any API route or response model, regenerate the typed client and commit the result — the
diff is the API contract changing:

```bash
./scripts/gen-api-types.sh
```
