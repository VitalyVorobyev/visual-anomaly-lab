# Frontend

The frontend is a professional engineering tool, not a debug panel: dense, keyboard-friendly, responsive, with
no configuration screens beyond what an experiment actually requires.

Its composition follows the data: images, maps, charts and result tables are the focal surface;
navigation, filters and configuration are subordinate. The visual direction is a calm technical
instrument at balanced density — hierarchy comes from alignment, type and spacing before borders or
colour.

## Stack

A deliberately small one — `react-router`, TanStack Query, Tailwind — with the API client generated
rather than written (**ADR-0012**).

- **Routing is `react-router` in its `HashRouter` form.** Not a stylistic preference: the bundle is
  served from three places — Vite's dev server at `/`, the desktop WebView at `…/index.html`, and
  `tauri://localhost` once packaged — and a path-based router matches no route at the second,
  rendering an empty document indistinguishable from a crash.
- **Server state is TanStack Query.** This application is almost entirely server state, with polling
  and invalidation on top.
- **The API client is generated.** `scripts/gen-api-types.sh` starts a throwaway backend, reads
  `/openapi.json`, and emits `frontend/src/api/generated.ts`. The file is **committed**, so `tsc` and
  CI need no running backend, and a CI job regenerates it and fails on any diff — which turns
  contract drift from a runtime failure into a type error.

## The token layer

Colour, type and radius are defined in `frontend/src/styles.css` and nowhere else (**ADR-0021**).

- **The direction is *instrument*: the chrome is grey so the data can be loud.** Anomaly maps,
  colormaps and grayscale sensor images are what this application exists to show, so chrome carries
  no saturation. Exactly one accent, `signal`, meaning "you can act here". `normal` / `defect` /
  `warn` are reserved for verdicts and are never decoration — a colour that sometimes means nothing
  teaches a reader to ignore it.
- **Greys are true neutral, not Tailwind's blue-cast `slate`.** A blue-tinted grey beside a viridis
  or inferno map shifts how the map's cool end reads, on the one screen that matters most.
- **Components name `surface`, `line`, `fg-muted`, `signal` — never a ramp step.** Nothing enforces
  this. A `text-slate-500` added later will compile, look almost right, and quietly ignore the theme.
- **Light and dark both ship**, with a three-state choice (`light` / `dark` / `system`) applied by an
  inline script before first paint, so "system" is a real state that survives a reload as itself.
- **Four Radix primitives and only four** — `Select`, `Dialog`, `Tooltip`, `Slider` (plus `Checkbox`
  and `Switch` from the same family). Everything else is hand-built on native elements, because
  native `<details>`, `<table>` and a radio group already have the semantics and the keyboard model.
  One trap this created: the base layer drops the UA `<summary>` marker, so a raw `<details>` outside
  the `Disclosure` primitive renders with no caret and reads as a dead panel.
- **Every interactive element carries `focus-visible:outline-2 outline-signal`.** An outline rather
  than a ring, because it follows the element's own border-radius and needs no offset colour.

## Shell capabilities are injected, not imported

**Nothing under `frontend/src/` imports a Tauri API.** The shell injects `window.__ANOMALY_LAB__`
before the page loads, carrying the sidecar's base URL and every capability a browser genuinely
cannot provide — the directory picker that must return an absolute path the backend can open,
reveal-in-Finder. One module, `src/api/shell.ts`, declares that global's shape and exposes a
`has<Capability>()` / `<capability>()` pair; nothing else touches it.

**A missing capability is a different affordance, not a broken one.** The directory picker's absence
means a text field, which is a perfectly good way to enter a path. The UI must never render a
disabled control whose only explanation is "you are not in the desktop app". Anything expressible
over HTTP belongs in the sidecar, which both hosts reach identically.

The cost, accepted: the contract is a hand-written global rather than a type-checked interface. A
renamed Rust command fails at runtime, in the desktop build only, where the browser tests never look.

## Scroll and layout ownership

The shell never scrolls. Every route belongs to one of three explicit layouts, so scroll ownership
is visible in the route table rather than emerging from nested `overflow` declarations:

- **`ReadingLayout`** owns one outer vertical scroller for catalogues, forms and tables.
- **`WorkspaceLayout`** owns the remaining viewport but does not scroll; the route gives its main
  data surface the single vertical scroller. The dataset browser uses this layout, with filters in a
  256 px supporting rail and the virtual grid filling the rest.
- **`CanvasLayout`** does not scroll; an image canvas fills the viewport and any supporting pane
  scrolls only when its own content requires it.

The main navigation contains `Datasets`, `Experiments` and `Compare`. Import is an action in the
dataset catalogue, not a peer destination; backend health remains visible in the shell and the full
health route remains directly addressable. Inside a dataset, a quiet local strip moves between
`Browse`, `Splits` and `Experiments`; annotation joins that strip when its route exists. A same-axis
scroller may not be placed inside another same-axis scroller.

## Screens

Each screen from the brief maps onto the API surface as follows.

| Screen | Purpose | Primary API |
| --- | --- | --- |
| **Dataset browser + import** | List datasets; discover local VisA/GKN packs and register every missing class in one job; native folder picker → scan → **manifest review** (edit channel mapping, fix labels, inspect warnings) → commit. The browser keeps label/channel/split/subset filters in a left rail and makes its virtual image grid the only vertical data scroller. Dataset deletion requires its exact name and previews the row cascade plus app-owned manifest, thumbnails, job logs and experiment artifacts; source images and masks are immutable. | `GET /api/reference-packs`, `POST /api/reference-packs/register`, `POST /api/import/scan`, `POST /api/import/commit`, `GET /api/datasets`, `GET /api/datasets/{id}/samples`, `GET /api/datasets/{id}/deletion-preview`, `DELETE /api/datasets/{id}`, `GET /api/images/{id}/thumb` |
| **Sample viewer (grouped)** | One part, all its channels side by side, channel count driven by data. Label editing (normal / defect / unlabeled) with keyboard shortcuts for fast passes over unlabeled data. Full-resolution zoom. | `GET /api/datasets/{id}/samples/{sid}`, `PATCH …/label`, `GET /api/images/{id}/preview`, `…/full` |
| **Annotation queue + editor** | A dataset-local image queue opens a flush `CanvasLayout`: source image central, narrow tool rail, supporting region inspector. The controlled Konva scene emits source-pixel polygons and cropped bitmap brush/eraser layers; selection, vertices, add/subtract and undo/redo never become a second truth store. Select-mode left-drag and universal right-drag pan without a dedicated tool; Fit, source-pixel 1:1 and reversible double-click Fit make view state explicit. A rough bitmap mark can be traced into simplified editable contours, while image-aware refinement remains MobileSAM's separate responsibility. Drafts autosave after idle, carry the response `ETag` in `If-Match`, and retain local work on a visible conflict; completion freezes a revision and advances across prefetched queue pages. | `GET /api/datasets/{id}/samples`, `GET /api/datasets/{id}/annotation-labels`, `POST/PUT /api/images/{id}/annotations/draft`, `POST …/complete`, `GET /api/images/{id}/full`, `…/mask` |
| **Split management** | Create a seeded, stratified split; per-subset counts by label; splits are immutable once created. | `POST /api/splits`, `GET /api/splits?dataset_id=` |
| **Experiment catalogue** | Dataset-scoped history is the primary view; the global catalogue is a secondary cross-dataset view. Name/notes, exact method and status filters plus ordering live in the URL, are applied in SQLite, and survive reload/back/forward. A deletion preview counts generated files and bytes, reports active-work blockers and resident eviction, and confirms that source dataset files are untouched. | `GET /api/experiments?dataset_id=&q=&model_type=&status=&sort=`, `GET /api/experiments/{id}/deletion-preview`, `DELETE /api/experiments/{id}` |
| **Experiment creation** | A dedicated route separates the large form from history. Inside a dataset the dataset is fixed; the user picks split + model. The configuration form is **generated from the model's JSON Schema** ([methods](methods.md)), so new hyperparameters appear with no frontend change. Capability flags drive the UI. | `GET /api/experiments/model-types`, `POST /api/experiments` |
| **Progress & logs** | Live job progress bar, streaming log console, cancel button. Snapshot-then-subscribe on mount and on reconnect ([the job system](jobs.md)). | `GET /api/jobs/{id}`, `WS /ws/jobs/{id}`, `POST /api/jobs/{id}/cancel` |
| **Training charts** (M4) | Per-branch loss curves and learning rate, live and after the fact; a series reported once is printed as a value rather than plotted as one dot. Survives a reload mid-run because the history is re-read from the job's log (**ADR-0020**). | `GET /api/jobs/{id}/metrics`, `WS /ws/jobs/{id}` |
| **Benchmark charts** (M4) | ROC and PR curves at sample and image level, score histogram by class with the threshold drawn on it, confusion matrix, per-defect-type breakdown, timing. Every curve integrates to a number the metrics table already shows. | `GET /api/experiments/{id}/curves?subset=`, `GET /api/experiments/{id}/results`, `…/threshold` |
| **Results** | Per-sample scores, ranked; **threshold slider** recomputing the confusion matrix live; **TP / FP / TN / FN filter**; **anomaly-map overlay with an opacity slider** (CSS-composited, instant) and the ground-truth outline where a mask exists; timing summary. | `GET /api/experiments/{id}/results?subset=`, `GET /api/experiments/{id}/threshold?value=`, `GET /api/experiments/{id}/samples/{sid}/images`, `GET /api/images/{id}/anomaly-map?experiment_id=`, `GET /api/images/{id}/mask` |
| **Diagnostics** (M4) | Whatever the method recorded about itself, rendered by `kind` and never by method name (ADR-0018). Run-scoped entries split into an *Architecture* tab (`graph`, `table`) and an *Inspector* tab (`map`, `image`, `grid`); image-scoped entries render on the sample page beside the combined anomaly map, which is what makes a two-branch method legible. | `GET /api/experiments/{id}/diagnostics`, `…/diagnostics/payload?key=` |
| **Experiment comparison** (M5) | N experiments side by side under one protocol, on one dataset and one split. Threshold-independent metrics compare directly; everything threshold-dependent is resolved **per run by one shared rule**, each run's own cut printed beside its confusion matrix, because a score has no meaning outside its own run (**ADR-0028**). Overlaid ROC and PR curves, a config diff calling out preprocessing loudly, a per-sample table filtered to where the methods disagree, and a side-by-side map view sharing one cut *fraction* and one zoom across every pane. | `GET /api/compare?ids=&subset=&at=`, `GET /api/experiments/{id}/curves`, `GET /api/images/{id}/anomaly-map?experiment_id=` |

Three cross-cutting UI rules follow from the design above:

1. **Nothing in the frontend hard-codes a channel count.** Channel layouts are rendered from the dataset's
   channel dictionary, and a two-channel sample renders correctly with no special case.
2. **Nothing in the frontend names a method.** The picker, every configuration form and every capability-driven
   affordance come from `GET /api/experiments/model-types`. Adding a method is a Python module and a registry
   entry; if it ever needs a line of TypeScript, that is a finding about the boundary (ADR-0007).
3. **Opacity is client state; the threshold is a server round trip.** Opacity is genuinely a view property —
   applied in CSS over an already-fetched PNG, instant, no request. The threshold is not: deciding which rows
   are false positives is the *rule* `score >= threshold`, and holding that rule in TypeScript as well as
   Python would let the two drift. So the threshold endpoint returns the counts **and the classified rows
   together**, and the client renders what it is given. It is a read over a few hundred stored floats and
   writes nothing (ADR-0011).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
