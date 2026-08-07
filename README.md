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
| **Import** | Point the app at a directory; review a proposed manifest (detected channels, grouping, warnings) before anything is written; commit it into the catalog. Re-importing the same directory updates it rather than duplicating it. |
| **Label & split** | Mark logical samples normal / defect / unlabeled; create seeded, sample-level train/val/test splits so no two views of the same object straddle a subset. |
| **Train** | Create an experiment (dataset + split + method + config), run it as an async job, and watch live progress and streaming logs. |
| **Infer** | Score samples with a trained experiment, producing per-image scores and anomaly maps. |
| **Evaluate** | Threshold-independent metrics, an interactive threshold slider driving a live confusion matrix, FP/FN lists, and ranked most-normal / most-anomalous views. |
| **Compare** | Several experiments side by side under one protocol — ROC curves overlaid, timing, config diff, and where the methods disagree. |

Throughout: **grouped multi-view samples** (one physical object, many images, one label), **anomaly-map
overlays** with an opacity slider, and **experiments that persist** — close the app, reopen it, and a past
experiment comes back with identical configuration, metrics, results, and logs.

> ### Project status — the loop closes
>
> **M0**–**M3** are complete: the whole loop above runs. Point the app at a directory or a public
> benchmark, review the proposed manifest, commit it, adopt the benchmark's own split, train a method,
> score it, and read image- **and** pixel-level metrics with a working anomaly-map overlay. Experiments
> persist; reopening one gives back identical numbers.
>
> Two methods ship: `pixel_reference`, a dataset-agnostic floor baseline that needs numpy and Pillow and
> trains in seconds, and `efficientad_anomalib`, which trains on Apple Silicon via MPS. The remaining rows
> of the [Methods](#methods) table are designed, not shipped.
>
> **Next is M4**, which makes the method *legible* — architecture view, teacher inspector, live training
> charts, benchmark charts, diagnostic overlays — all built on the diagnostics contract M3 established.
> See [`docs/roadmap.md`](docs/roadmap.md) for the milestone sequence and honest sizing.

## Methods

All methods sit behind one plugin interface and are selected by a stable registry key. Adding a method
means adding a module and a registry entry — nothing else in the application changes.

| Registry key | What it is | Scope |
| --- | --- | --- |
| `pixel_reference` | Dataset-agnostic floor baseline: per-pixel median + MAD over the training normals → z-map → smoothing → high-percentile score. numpy and Pillow only, trains in seconds, gives every deep result something to beat. | Dataset-agnostic |
| `efficientad_anomalib` | EfficientAD via Intel's [anomalib](https://github.com/open-edge-platform/anomalib) | Dataset-agnostic |
| `efficientad_custom` | From-scratch EfficientAD reimplementation, for direct comparison against the library version | Dataset-agnostic |
| `patchcore_anomalib` | PatchCore via anomalib | Dataset-agnostic |
| `classical_circular` | Geometry-aware classical baseline tailored to the showcase dataset: circle fit → polar unwrap → orientation alignment → robust per-channel reference comparison. Deferred to a later, optional milestone (ADR-0015). | **Showcase-dataset-specific** (circular parts) |

The classical baseline is the *only* component allowed to assume anything about the showcase dataset's
geometry. Everything else must work on a dataset it has never seen — which is why the vertical slice is
proved on a dataset-agnostic method against public benchmarks, and not on the classical one.

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
no pixel masks, so evaluation of *this* dataset is image-level, with sample-level ROC-AUC as the headline
metric. Pixel-level metrics come from the public datasets below, which do ship masks.

## Reference datasets

Methods are developed and validated against public benchmarks, not only against the private tree. That is
what keeps the tool universal: a number computed here can be compared against a number someone else
published, on the same data and — through the `csv_table` adapter — on the same official split.

**These datasets are not committed.** `/datasets/` is gitignored: they are large, freely available, and
adding gigabytes to a source repository to duplicate a public download buys nothing.
`scripts/check-repo-safety.sh` fails if anything under `datasets/` is ever staged. Download them yourself
and point the import screen at the directory.

| Dataset | What it is | Adapter | Licence |
| --- | --- | --- | --- |
| [**VisA**](https://github.com/amazon-science/spot-diff) (Visual Anomaly), Zou et al. | 12 object classes, ~1000 normal + 100 anomalous images each, **with pixel-level ground-truth masks** and official one-class split tables in `split_csv/1cls.csv`. | `csv_table` | CC BY 4.0 (licence file ships with the download) |
| [**GKN Blade Surface Defect Dataset**](https://doi.org/10.17632/3bh998k78g.1), Qianyu Zhou, University of Connecticut, 22 May 2023 | 203 good, 48 nick, 149 scratch photographs of blade surfaces. No masks. | `folder_classes` | CC BY 4.0, DOI [10.17632/3bh998k78g.1](https://doi.org/10.17632/3bh998k78g.1) |

Both are used for training and as the comparison baseline. To import one, point the import screen at its
root directory and fill in the adapter options — for GKN, `normal_dirs = Data_GKN/Good` and
`defect_dirs = Data_GKN/Nick, Data_GKN/Scratch`; for one VisA class,
`csv_path = split_csv/1cls.csv` with `filter_column = object` and `filter_value = candle`. Then create a
split with the **`imported`** strategy to adopt the published partition rather than drawing your own,
which is what makes the resulting figure comparable to the paper's.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/system-design.md`](docs/system-design.md) | Architecture, domain model, canonical terminology, API surface, job and evaluation protocols |
| [`docs/roadmap.md`](docs/roadmap.md) | Milestones M0–M9, scope, exit criteria, sizing |
| [`docs/backlog.md`](docs/backlog.md) | Task-level breakdown by epic |
| [`docs/adr/`](docs/adr/) | Architecture decision records 0001–0018 — what was decided, why, and what it costs |

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

The deep-learning methods live behind an optional dependency group, so a checkout that only wants the
baseline never pays for torch. Install it when you want EfficientAD, and check the accelerator before
trusting it:

```bash
uv sync --directory backend --extra dl
./scripts/mps-smoke-test.py               # is MPS actually usable here? (ADR-0008)
```

A handful of backend tests assert the exact composition of the private showcase tree. They are skipped
unless you point them at it, and CI never does:

```bash
ANOMALY_LAB_SHOWCASE_ROOT=/path/to/tree uv run --directory backend pytest -k showcase
```

After changing any API route or response model, regenerate the typed client and commit the result — the
diff is the API contract changing:

```bash
./scripts/gen-api-types.sh
```
