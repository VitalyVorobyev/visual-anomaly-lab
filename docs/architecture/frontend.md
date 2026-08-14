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

**The document cannot scroll, by construction.** `html`, `body` and `#root` are pinned to `height:
100%; overflow: hidden` in `styles.css`, and the shell frame is `h-full` rather than `h-screen`.
`100vh` is measured against the initial containing block: it does not subtract a horizontal
scrollbar and does not track a fractional-device-pixel viewport, so with nothing clipping `body` the
document could grow a scrollbar *beside* the layout's, and the window would narrow by its width —
which is what put two bars on screen and shifted the shell header on one tab and not its neighbour.

Every route belongs to one of three explicit layouts, so scroll ownership is visible in the route
table rather than emerging from nested `overflow` declarations:

- **`ReadingLayout`** owns one outer vertical scroller for catalogues, forms and tables.
- **`DatasetLayout`** (`routes/dataset/`) owns the remaining viewport but does not scroll. It also
  renders the dataset's identity band — name, one run of facts, the one dataset-level action — and
  the section strip, **exactly once, above all five tabs**. The facts are counted in **samples**,
  never images: `label_counts` and split membership are stored per sample (ADR-0005), so the badges
  beside the count share its denominator. A channel count appears only when a sample is more than
  one image; the root path, the adapter and the import date are behind an information mark, because
  they are consulted rather than read. A tab renders no page heading, no
  strip and no back link of its own; it gives its main data surface the single vertical scroller,
  through `TabScroll` or, where the surface is full-bleed, its own. The dataset browser is the
  full-bleed case, with filters in a 256 px supporting rail and the virtual grid filling the rest.
- **`CanvasLayout`** does not scroll; an image canvas fills the viewport and any supporting pane
  scrolls only when its own content requires it. The single-sample viewer belongs here, not to the
  reading measure: it was four stacked panels in a 72 rem column with the image third, boxed at a
  fixed 384 px, so the one thing the screen exists for was the smallest element on it. It is now a
  thin band of identity and paging, the image taking every pixel that is left, and the label,
  channel and file controls in a 288 px rail beside it.

**Exactly one page-level scroller per screen.** A bounded pane may scroll on the same axis only when
it is a peer *column* beside the page's content — the dataset browser's filter rail and the sample
viewer's control rail are the two such cases — never when it is stacked inside the page's own flow. A log tail, a warnings list or a ranked
list in the flow is clipped and given a disclosure instead, so its lines land in the page's own
scroller rather than behind a second one. `data-scroll="tab"` marks a dataset tab's single region;
`frontend/src/routes/dataset/tabScroll.test.tsx` asserts there is one per tab and nothing scrolling
inside it.

**A control never nests inside a link.** The catalogue card's edit and delete buttons and the browse
tile's selection box are absolutely-positioned siblings of their `<Link>`, not children of it. This
is a correctness rule, not a styling preference: a control inside an anchor has to cancel the click
to stop the navigation, and cancelling a checkbox's click makes the browser restore its previous
state *after* React has written the new one — the tick then arrives one render late. Outside the
anchor there is nothing to cancel.

**A collection is created by naming it and filling it, in one dialog.** `collection` is a string on
each dataset ([the domain model](domain-model.md)), so a collection exists for exactly as long as
some dataset names it and an empty one cannot be stored. `CollectionDialog` therefore asks for the
name and the membership together, serves renaming and re-filing through the same form, and writes a
sequence of `PATCH /api/datasets/{id}` calls. Unticking clears the *override*, which returns a
reference dataset to its pack and a user's own to no group at all.

The main navigation contains `Datasets`, `Experiments` and `Compare`. Import is an action in the
dataset catalogue, not a peer destination; backend health remains visible in the shell and the full
health route remains directly addressable. Inside a dataset, the layout's strip moves between
`Browse`, `Annotate`, `Prepare`, `Splits` and `Experiments`; it is underlined rather than pilled,
because a pill in this application marks an in-page state switch and these are navigations.

## Screens

Each screen from the brief maps onto the API surface as follows.

| Screen | Purpose | Primary API |
| --- | --- | --- |
| **Dataset browser + import** | The catalogue is a grid of covers grouped by collection — a card carries a thumbnail, a name and a sentence, and nothing else. Discover local VisA/GKN packs and register every missing class in one job; native folder picker → scan → **manifest review** (edit channel mapping, fix labels, inspect warnings) → commit. The browser keeps label/channel/split/subset filters in a left rail and makes its virtual image grid the only vertical data scroller. Filtering by channel selects whole *samples* that have one, so each tile previews **that** channel and names it in place of the channel count, and opening a sample lands on it: previewing image one regardless made the rail look inert on a multi-channel dataset. A sample missing the filtered channel falls back to its first image rather than going blank. Dataset deletion requires its exact name and previews the row cascade plus app-owned manifest, thumbnails, job logs and experiment artifacts; source images and masks are immutable. | `GET /api/reference-packs`, `POST /api/reference-packs/register`, `POST /api/import/scan`, `POST /api/import/commit`, `GET /api/datasets`, `PATCH /api/datasets/{id}`, `GET /api/datasets/{id}/samples`, `GET /api/datasets/{id}/deletion-preview`, `DELETE /api/datasets/{id}`, `GET /api/images/{id}/thumb` |
| **Sample viewer (grouped)** | One part, all its channels side by side, channel count driven by data. Label editing (normal / defect / unlabeled) with keyboard shortcuts for fast passes over unlabeled data. Full-resolution zoom. | `GET /api/datasets/{id}/samples/{sid}`, `PATCH …/label`, `GET /api/images/{id}/preview`, `…/full` |
| **Annotation queue + editor** | A dataset-local image queue opens a flush `CanvasLayout`: source image central, narrow tool rail, supporting region inspector. The controlled Konva scene emits source-pixel polygons and cropped bitmap brush/eraser layers; selection, vertices, add/subtract and undo/redo never become a second truth store. Select-mode left-drag and universal right-drag pan without a dedicated tool; Fit, source-pixel 1:1 and reversible double-click Fit make view state explicit. The focusable canvas adds a visible source-pixel keyboard cursor: arrows move one pixel, Shift moves ten, Space applies the current drawing tool and Enter closes a polygon; with a region selected under Select the same arrows nudge *it*, which is what a region copied from another channel needs. J/K cross queue pages and C completes. A brush or eraser stroke extends the selected region rather than minting another one, a region is dragged with Select, a polygon closes on its own first vertex or on a double-click, and a dragged vertex reshapes the outline live. The eraser only ever removes: with nothing selected it takes paint off every region the stroke passes over, and a cut through a polygon is an explicit Subtract region rather than the eraser's side effect. A rough bitmap mark can be traced into simplified editable contours. MobileSAM takes source-pixel positive/negative points or a box and previews up to three ranked masks without mutating the draft; acceptance keeps the compact mask or immediately traces it to editable polygons. The verified checkpoint is downloaded from the fixed model-asset catalogue only after explicit licence acceptance, with ordinary job progress/cancel. Drafts autosave after idle, carry the response `ETag` in `If-Match`, and retain local work on a visible conflict; completion freezes a revision and advances across prefetched queue pages. A draft is created by the first save, so opening the editor writes nothing, and a persisted one can be discarded from the header with a force offered only after a conflict. The queue filters to one label and to samples still missing truth, and each card shows whether that sample's images all have truth, some do, or none. Where a sample has more than one image the canvas column carries a channel strip and three view modes — one channel, two side by side sharing a single controlled view, or a blend compositing a second channel at adjustable alpha, which is the only way to see a few pixels of misregistration. Under a dataset's `sample` annotation scope (ADR-0036) one document covers the whole part: switching channel is a display change rather than navigation, one completion writes every channel, and the queue collapses to one card per part. Under `image` scope switching channel saves first and then navigates rather than locking the tab; the side-by-side reference pane draws that channel's *own* draft, `Edit this channel` exchanges the two panes so editing always happens in the left one, and `Copy to…` appends the current regions to the sibling channels the reader picks, each copy independently editable because the exposures are milliseconds apart. How the reader is looking — pane mode, second channel, tool, brush size, mask weight, pan and zoom — lives above the keyed editor and survives changing channel; what they are looking at — history, selection, the pending polygon — is keyed by the draft's target and resets with it. The pan and zoom additionally carry a frame stamp, so they follow the reader across the channels of one part and reset on a part or frame that has nothing to do with them. Which scope a dataset uses, and every reason it cannot use `sample`, is stated on the queue screen — and where the reason is open drafts, each one is a link into the editor that holds it rather than a count of things nobody can find. | `GET /api/datasets/{id}/samples`, `GET /api/datasets/{id}/annotation-labels`, `GET/PUT /api/datasets/{id}/annotation-scope`, `GET/POST/PUT/DELETE /api/images/{id}/annotations/draft`, `POST /api/images/{id}/annotations/copy-regions`, `GET/POST/PUT/DELETE /api/samples/{id}/annotations/draft`, `POST …/complete`, `GET /api/segment-assist`, `POST /api/images/{id}/segment-assist`, `GET /api/model-assets`, `GET /api/images/{id}/full`, `…/mask` |
| **Region preparation** | A dataset-local one-scroller workspace keeps immutable profile configuration in a sticky supporting rail and the visual crop audit central. Extractor schemas drive their options. Preview samples 24 images across the dataset without writing; Build materialises every successful prepared PNG atomically. Source/crop and prepared views expose failures, coverage and storage rather than hiding them behind the experiment form. A revision can be deleted; the confirmation names the experiments pinning it, because "delete those first" is only actionable if you know which ones. | `GET /api/region-extractors`, `GET/POST /api/datasets/{id}/region-profiles`, `POST /api/region-profiles/{id}/preview`, `POST/GET /api/region-profiles/{id}/build`, `GET /api/region-profiles/{id}/prepared/{image_id}`, `GET/DELETE /api/region-profiles/{id}` |
| **Split management** | Create a seeded, stratified split; per-subset counts by label; splits are immutable once created. | `POST /api/splits`, `GET /api/splits?dataset_id=` |
| **Experiment catalogue** | Dataset-scoped history is the primary view; the global catalogue is a secondary cross-dataset view. Name/notes, exact method and status filters plus ordering live in the URL, are applied in SQLite, and survive reload/back/forward. A deletion preview counts generated files and bytes, reports active-work blockers and resident eviction, and confirms that source dataset files are untouched. | `GET /api/experiments?dataset_id=&q=&model_type=&status=&sort=`, `GET /api/experiments/{id}/deletion-preview`, `DELETE /api/experiments/{id}` |
| **Experiment creation** | A dedicated route separates the large form from history. Inside a dataset the dataset is fixed; the user picks a split, a completed prepared-input revision and a model. Missing/unbuilt profiles link back to the dataset's Prepare workspace rather than falling back invisibly. Method, colour and evaluation forms are **generated from JSON Schema** ([methods](methods.md)), so new hyperparameters appear with no frontend change. Capability flags drive the UI. | `GET /api/experiments/model-types`, `GET /api/datasets/{id}/region-profiles`, `GET /api/region-profiles/{id}/build`, `POST /api/experiments` |
| **Progress & logs** | Live job progress bar, streaming log console, cancel button. Snapshot-then-subscribe on mount and on reconnect ([the job system](jobs.md)). | `GET /api/jobs/{id}`, `WS /ws/jobs/{id}`, `POST /api/jobs/{id}/cancel` |
| **Training charts** (M4) | Per-branch loss curves and learning rate, live and after the fact; a series reported once is printed as a value rather than plotted as one dot. Survives a reload mid-run because the history is re-read from the job's log (**ADR-0020**). | `GET /api/jobs/{id}/metrics`, `WS /ws/jobs/{id}` |
| **Benchmark charts** (M4) | ROC and PR curves at sample and image level, score histogram by class with the threshold drawn on it, confusion matrix, per-defect-type breakdown, timing. Every curve integrates to a number the metrics table already shows. | `GET /api/experiments/{id}/curves?subset=`, `GET /api/experiments/{id}/results`, `…/threshold` |
| **Results** | Per-sample scores, ranked; **threshold slider** recomputing the confusion matrix live; **TP / FP / TN / FN filter**; **anomaly-map overlay with an opacity slider** (CSS-composited, instant) and the ground-truth outline where a mask exists; timing summary. | `GET /api/experiments/{id}/results?subset=`, `GET /api/experiments/{id}/threshold?value=`, `GET /api/experiments/{id}/samples/{sid}/images`, `GET /api/images/{id}/anomaly-map?experiment_id=`, `GET /api/images/{id}/mask` |
| **Diagnostics** (M4) | Whatever the method recorded about itself, rendered by `kind` and never by method name (ADR-0018). Run-scoped entries split into an *Architecture* tab (`graph`, `table`) and an *Inspector* tab (`map`, `image`, `grid`); image-scoped entries render on the sample page beside the combined anomaly map, which is what makes a two-branch method legible. | `GET /api/experiments/{id}/diagnostics`, `…/diagnostics/payload?key=` |
| **Experiment comparison** (M5) | N experiments side by side under one protocol, on one dataset and one split. Threshold-independent metrics compare directly; everything threshold-dependent is resolved **per run by one shared rule**, each run's own cut printed beside its confusion matrix, because a score has no meaning outside its own run (**ADR-0028**). Overlaid ROC and PR curves, a config diff calling out preprocessing loudly, a per-sample table filtered to where the methods disagree, and a side-by-side map view sharing one cut *fraction* and one zoom across every pane. | `GET /api/compare?ids=&subset=&at=`, `GET /api/experiments/{id}/curves`, `GET /api/images/{id}/anomaly-map?experiment_id=` |
| **Portable export** | The run bar offers `Export ONNX` only for a method whose capability declares it. The generic export job shows ordinary progress/cancel, and the resulting bundle appears under Jobs & files. An unsupported method has a disabled, explanatory affordance rather than a job that fails after loading the model. | `POST /api/experiments/{id}/export`, `GET /api/jobs/{id}` |

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
