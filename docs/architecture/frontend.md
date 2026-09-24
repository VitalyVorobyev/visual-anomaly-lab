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

## When the frontend crashes, it says so

React 19 unmounts the entire root when anything throws during render, and `index.html`
declares `color-scheme: dark` — so an unhandled error paints a **black window**, which is
indistinguishable from a hung shell, a sidecar that never started, or a dev server that
died. The desktop build has no console to check, so that failure mode has no next step.

`components/CrashScreen.tsx` closes it, in two halves wired up in `main.tsx`:

- **`CrashBoundary`** wraps everything, outside the router: a route that throws is caught,
  and so is a router that fails to construct. It renders the error's message with the
  component stack.
- **`installCrashHandlers(container)`** runs *before* the first render and listens for
  `error` and `unhandledrejection`, which is what catches a module that throws while being
  evaluated or a lazily-imported chunk that never arrives — neither of which any boundary
  can see. It paints the same panel straight into the DOM, and **only when the root is
  empty**: a live UI reporting its own failed request must not be replaced by it.

The file imports nothing — no `@vitavision/lab-ui`, no Tailwind class, no token, inline
styles only. A crash screen that needs the stylesheet is another black window on the day
the stylesheet is what failed.

**The same panel reports a backend that never started.** The shell builds its window either
way and injects `startupError` instead of the capabilities — on macOS its `setup` hook runs
inside `did_finish_launching`, an Objective-C callback an unwind may not cross, so returning
an error there aborts the process and shows nothing at all (that is exactly how an installed
build died before `uv` was resolved by absolute path). `main.tsx` reads it through
`shellStartupError()` **before** mounting anything and renders `CrashScreen` with its own
headline; the router, the query client and every fetch against a port nothing is listening on
are never constructed. A packaged app's stderr is written to nothing, so the panel carries the
detail the shell collected — the paths searched, and the backend's own last output.

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
| **Dataset browser + import** | The catalogue is a grid of covers grouped by collection — a card carries a thumbnail, a name and a sentence, and nothing else. Discover local VisA/GKN packs and register every missing class in one job; native folder picker → scan → **manifest review** (edit channel mapping, fix labels, inspect warnings) → commit. The scan job and the manifest under review are in the URL (`scan=`, `manifest=`), each step a history entry, so a reload or Back returns to the console or the review rather than orphaning them. The browser keeps label/channel/split/subset filters in a left rail and makes its virtual image grid the only vertical data scroller. Filtering by channel selects whole *samples* that have one, so each tile previews **that** channel and names it in place of the channel count, and opening a sample lands on it: previewing image one regardless made the rail look inert on a multi-channel dataset. A sample missing the filtered channel falls back to its first image rather than going blank. A dataset names the channel it is read in, set in the same edit dialog as its description and collection and shown there only when it has more than one; that name then answers wherever one image stands for a whole part — the card's cover, a grid tile, the sample viewer's opening tab, the annotation queue, the compare view — with the rail's channel filter still winning over it and the first image still answering when a part was not photographed that way. Dataset deletion requires its exact name and previews the row cascade plus app-owned manifest, thumbnails, job logs and experiment artifacts; source images and masks are immutable. | `GET /api/reference-packs`, `POST /api/reference-packs/register`, `POST /api/import/scan`, `POST /api/import/commit`, `GET /api/datasets`, `PATCH /api/datasets/{id}`, `GET /api/datasets/{id}/samples`, `GET /api/datasets/{id}/deletion-preview`, `DELETE /api/datasets/{id}`, `GET /api/images/{id}/thumb` |
| **Sample viewer (grouped)** | One part, all its channels side by side, channel count driven by data. It opens on the channel the rail is filtered to, else the dataset's default, else the first. Label editing (normal / defect / unlabeled) with keyboard shortcuts for fast passes over unlabeled data. Full-resolution zoom. | `GET /api/datasets/{id}/samples/{sid}`, `PATCH …/label`, `GET /api/images/{id}/preview`, `…/full` |
| **Annotation queue + editor** | A dataset-local image queue opens a flush `CanvasLayout`: source image central, narrow tool rail, supporting region inspector. The controlled Konva scene emits source-pixel polygons and cropped bitmap brush/eraser layers; selection, vertices, add/subtract and undo/redo never become a second truth store. Select-mode left-drag and universal right-drag pan without a dedicated tool; Fit, source-pixel 1:1 and reversible double-click Fit make view state explicit; zoom reaches 32 screen pixels per source pixel with smoothing off above 1:1, and a readout in each pane gives the source pixel under the pointer, the mask value the document resolves to there, the region on top and the zoom as a percentage of 1:1. A brush or eraser shows its own footprint, at its true diameter, as the cursor, and `?` opens a sheet of every key and gesture rendered from the same list the keymap resolves against. The focusable canvas adds a visible source-pixel keyboard cursor: arrows move one pixel, Shift moves ten, Space applies the current drawing tool and Enter closes a polygon; with a region selected under Select the same arrows nudge *it*, which is what a region copied from another channel needs. J/K cross queue pages and C completes. A brush or eraser stroke extends the selected region rather than minting another one, a region is dragged with Select, a polygon closes on its own first vertex or on a double-click, and a dragged vertex reshapes the outline live. The eraser only ever removes: with nothing selected it takes paint off every region the stroke passes over, and a cut through a polygon is an explicit Subtract region rather than the eraser's side effect. A rough bitmap mark can be traced into simplified editable contours. MobileSAM takes source-pixel positive/negative points or a box and previews up to three ranked masks without mutating the draft; acceptance keeps the compact mask or immediately traces it to editable polygons. The verified checkpoint is downloaded from the fixed model-asset catalogue only after explicit licence acceptance, with ordinary job progress/cancel. Drafts autosave after idle, carry the response `ETag` in `If-Match`, and retain local work on a visible conflict; completion freezes a revision and advances across prefetched queue pages. A draft is created by the first save, so opening the editor writes nothing, and a persisted one can be discarded from the header with a force offered only after a conflict. The queue filters to one label and to samples still missing truth, and each card shows whether that sample's images all have truth, some do, or none. Where a sample has more than one image the canvas column carries a channel strip and three view modes — one channel, two side by side sharing a single controlled view, or a blend compositing a second channel at adjustable alpha, which is the only way to see a few pixels of misregistration. Under a dataset's `sample` annotation scope (ADR-0036) one document covers the whole part: switching channel is a display change rather than navigation, one completion writes every channel, and the queue collapses to one card per part. Under `image` scope switching channel saves first and then navigates rather than locking the tab; the side-by-side reference pane draws that channel's *own* draft, `Edit this channel` exchanges the two panes so editing always happens in the left one, and `Copy to…` appends the current regions to the sibling channels the reader picks, each copy independently editable because the exposures are milliseconds apart. How the reader is looking — pane mode, second channel, tool, brush size, mask weight, pan and zoom — lives above the keyed editor and survives changing channel; what they are looking at — history, selection, the pending polygon — is keyed by the draft's target and resets with it. The pan and zoom additionally carry a frame stamp, so they follow the reader across the channels of one part and reset on a part or frame that has nothing to do with them. Both channel pickers are the same strip: the second pane lists every channel with the one already in the left pane disabled and carrying its reason, rather than a dropdown beside a row of buttons. `H` hides the drawn regions and shows them again, holding it shows the other state while held, and an eye beside the Mask weight does the same — the weight dims fills and cannot reach a bare photograph, because outlines and handles carry no opacity, and unlike the weight the visibility is deliberately not remembered between visits. A new region's class and operation are not chosen in advance: operation is set on the selected shape in Selection, and a class picker appears only on a dataset with more than one class. The header carries the **sample's own label** as a three-way control with `N`/`D`/`U` keys, its `imported`/`hand-set` provenance, and — only where the label and the open document contradict each other — a quiet mark naming what evaluation will do about it; the label is the part's, so it covers every channel, and setting it leaves an unsaved draft untouched. Which scope a dataset uses, and every reason it cannot use `sample`, is stated on the queue screen — and where the reason is open drafts, each one is a link into the editor that holds it rather than a count of things nobody can find. | `GET /api/datasets/{id}/samples`, `PATCH /api/datasets/{id}/samples/{sid}`, `GET /api/datasets/{id}/annotation-labels`, `GET/PUT /api/datasets/{id}/annotation-scope`, `GET/POST/PUT/DELETE /api/images/{id}/annotations/draft`, `POST /api/images/{id}/annotations/copy-regions`, `GET/POST/PUT/DELETE /api/samples/{id}/annotations/draft`, `POST …/complete`, `GET /api/segment-assist`, `POST /api/images/{id}/segment-assist`, `GET /api/model-assets`, `GET /api/images/{id}/full`, `…/mask` |
| **Region preparation** | A dataset-local one-scroller workspace keeps immutable profile configuration in a sticky supporting rail and the visual crop audit central. The open revision and the job being followed (`profile`, `job`, `mode`, `jobProfile`) live in the URL, so a reload mid-build returns to the same console. A new profile's name defaults to its extractor and size. Extractor schemas drive their options. Preview samples 24 images across the dataset without writing; Build materialises every successful prepared PNG atomically. Source/crop and prepared views expose failures, coverage and storage rather than hiding them behind the experiment form. A revision can be deleted; the confirmation names the experiments pinning it, because "delete those first" is only actionable if you know which ones. | `GET /api/region-extractors`, `GET/POST /api/datasets/{id}/region-profiles`, `POST /api/region-profiles/{id}/preview`, `POST/GET /api/region-profiles/{id}/build`, `GET /api/region-profiles/{id}/prepared/{image_id}`, `GET/DELETE /api/region-profiles/{id}` |
| **Split management** | Create a seeded, stratified split; per-subset counts by label; splits are immutable once created. | `POST /api/splits`, `GET /api/splits?dataset_id=` |
| **Experiment catalogue** | Dataset-scoped history is the primary view; the global catalogue is a secondary cross-dataset view. Name/notes, exact method and status filters plus ordering live in the URL, are applied in SQLite, and survive reload/back/forward. A checkbox column picks runs for Compare under the picker's own `refusalReason`, so a selection made here always opens. A deletion preview counts generated files and bytes, reports active-work blockers and resident eviction, and confirms that source dataset files are untouched. | `GET /api/experiments?dataset_id=&q=&model_type=&status=&sort=`, `GET /api/experiments/{id}/deletion-preview`, `DELETE /api/experiments/{id}` |
| **Experiment creation** | A dedicated route separates the large form from history. Inside a dataset the dataset is fixed; the user picks a split, a built region profile and a model. The dataset band says beforehand what is missing — a built region profile, a split — as links in the order they have to be done, using the form's own rule for "built" (`isUsableBuild`). Missing/unbuilt profiles link back to the dataset's Prepare workspace rather than falling back invisibly, and the unsent form is kept in `sessionStorage` (`api/experimentDraft.ts`) so following those links costs nothing typed. The method cards are the ones whose `capabilities.tasks` list the chosen task, and a task picker appears only when the methods offer more than one (ADR-0039). A lone profile or split is selected on its own, an empty name takes the suggested `<method> on <dataset>`, and what is still missing appears beside its field once Create is pressed. Method, colour and evaluation forms are **generated from JSON Schema** ([methods](methods.md)), so new hyperparameters appear with no frontend change. Capability flags drive the UI. | `GET /api/experiments/model-types`, `GET /api/datasets/{id}/region-profiles`, `GET /api/region-profiles/{id}/build`, `POST /api/experiments` |
| **Progress & logs** | Live job progress bar, streaming log console, cancel button. Snapshot-then-subscribe on mount and on reconnect ([the job system](jobs.md)). | `GET /api/jobs/{id}`, `WS /ws/jobs/{id}`, `POST /api/jobs/{id}/cancel` |
| **Training charts** (M4) | Per-branch loss curves and learning rate, live and after the fact; a series reported once is printed as a value rather than plotted as one dot. Survives a reload mid-run because the history is re-read from the job's log (handbook jobs.md). | `GET /api/jobs/{id}/metrics`, `WS /ws/jobs/{id}` |
| **Benchmark charts** (M4) | ROC and PR curves at sample and image level, score histogram by class with the threshold drawn on it, confusion matrix, per-defect-type breakdown, timing. Every curve integrates to a number the metrics table already shows. | `GET /api/experiments/{id}/curves?subset=`, `GET /api/experiments/{id}/results`, `…/threshold` |
| **Results** | Per-sample scores, ranked; **threshold slider** recomputing the confusion matrix live; **TP / FP / TN / FN filter**. **One threshold and one subset per experiment, in the URL**: Overview, Benchmark and Samples read and write the same `ResultsState` (`t`, `subset`), resolved once by `resolveSubset` to the last scored subset when the URL names none — an absent subset on the server ranks *every* scored subset together, which is a different population from the one the confusion matrix counts. The panel under the matrix does not browse verdicts itself; it links each outcome into the Samples tab at the cut in force, and the gallery prints which threshold its badges are at; **anomaly-map overlay with an opacity slider** (CSS-composited, instant) and the ground-truth outline where a mask exists; timing summary. The sample viewer stacks photograph, heatmap, segmentation and outline as layers inside one `ImageStage` per channel, sharing a single `StageView`, and the arrows are left to the sample list (`panKeys={false}`) while the stage keeps `0` / `1` / `+` / `-`. **The localization verdict rides beside the outcome, never inside it** ([evaluation](evaluation.md)): a second badge — `localized` / `off target`, and *nothing* for a `null` — on the sample header, on the gallery tile and on the classified row, plus `N of M` in the confusion strip whose `M` counts the rows carrying a verdict rather than the subset. That figure does not move with the slider beside it, which is what its `InfoHint` is there to say. A fifth `peak` layer draws the marker behind it: a cross at the stored argmax and the **square** Chebyshev window of `tolerance_px` it was tested against, as `MeasureOverlay` primitives inside the same stage transform, toned by the verdict. Off by default, and `null` draws in `signal` so an unchecked image never borrows the green of a passed check. | `GET /api/experiments/{id}/results?subset=`, `GET /api/experiments/{id}/threshold?value=`, `GET /api/experiments/{id}/samples/{sid}/images`, `GET /api/images/{id}/anomaly-map?experiment_id=`, `GET /api/images/{id}/mask` |
| **Diagnostics** (M4) | Whatever the method recorded about itself, rendered by `kind` and never by method name (ADR-0018). Run-scoped entries split into an *Architecture* tab (`graph`, `table`) and an *Inspector* tab (`map`, `image`, `grid`); image-scoped entries render on the sample page beside the combined anomaly map, which is what makes a two-branch method legible. **Those panes are in the prepared frame** — nothing projects a diagnostic, by design ([diagnostics](diagnostics.md)) — so their ground-truth outline is fetched with `frame=prepared`, while the combined map beside them keeps the source-frame outline because the stored map was projected before it was written. | `GET /api/experiments/{id}/diagnostics`, `…/diagnostics/payload?key=`, `GET /api/images/{id}/mask?frame=prepared&experiment_id=` |
| **Experiment comparison** (M5) | N experiments side by side under one protocol, on one dataset and one split. Reached from the top bar, from a scored run's "Compare with…", or from the catalogue's checkboxes; once one run is picked the picker shows only its dataset and split, refuses runs that are not trained, and prints each refusal on the row rather than in a tooltip. Threshold-independent metrics compare directly; everything threshold-dependent is resolved **per run by one shared rule**, each run's own cut printed beside its confusion matrix, because a score has no meaning outside its own run (**ADR-0028**). Overlaid ROC and PR curves, a config diff calling out preprocessing loudly, a per-sample table filtered to where the methods disagree, and a side-by-side map view sharing one cut *fraction* and one `StageView` across every pane. That view state is shared because the comparison is spatial, and its one caveat is stated where it lives: `StageView.scale` is absolute — CSS pixels per image pixel — so panes drawn from images of *different* pixel sizes would sit at different fractions of fit. Every pane here draws the same image of the same sample, so it does not arise; a fit-relative zoom would reintroduce exactly the frame-relative arithmetic `ImageStage` removed. The localization verdict and the `peak` layer are **per pane**, from each run's own `ImageScore`, because two methods agreeing on the label and disagreeing about where the defect is is exactly what this screen exists to show; the localization rows join the threshold-independent table for the same reason they sit beside ROC-AUC rather than beside a cut. | `GET /api/compare?ids=&subset=&at=`, `GET /api/experiments/{id}/curves`, `GET /api/images/{id}/anomaly-map?experiment_id=` |
| **Portable export** | The run bar offers `Export ONNX` only for a method whose capability declares it. The generic export job shows ordinary progress/cancel, and the resulting bundle appears under Jobs & files. An unsupported method has a disabled, explanatory affordance rather than a job that fails after loading the model. | `POST /api/experiments/{id}/export`, `GET /api/jobs/{id}` |

**One vocabulary.** A *region profile* is the saved crop-and-resize recipe built on Prepare; *Colour* is
the experiment's colour option (it was "Model input", which was also the default profile name). The
*threshold* is a run's image-score cut; the *map cut* is the fraction of a run's map range that draws its
segmentation; the *threshold rule* is how Compare chooses each run's threshold. A method is shown by its
title, with its registry key as secondary text where a reader may need it.

Five cross-cutting UI rules follow from the design above:

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
4. **Layer registration is structural, not a class every layer has to get right.** A result viewer stacks a
   photograph, a heatmap, a segmentation and a ground-truth outline, and they are only a comparison while
   they agree pixel for pixel. `ImageStage` lays its stage out at the image's **own pixel size** and carries
   the whole transform, so a layer at `inset-0` covers exactly the source frame at every viewport size —
   there is no box to fill and therefore no `object-*` class in either result viewer. Its predecessor,
   `ZoomPanCanvas`, composed a layout scale from the *frame's* size with a CSS transform, which held only
   while the frame's aspect ratio matched the picture's: clamped by `max-w-full` with its height left full,
   a column narrower than the image drew every layer squashed. Identically squashed, and so invisible as a
   fault — the picture was simply not the shape of the part. `frontend/src/routes/ExperimentSampleRoute.test.tsx`
   pins it. All three sample viewers — the dataset browser's, the results viewer and the compare panes — are
   one component, `components/viewer/SampleStage.tsx`: the photograph at `tierFor(view)`, the raster layers in
   order, then `VectorLayer` (boxes and polygons with a label, in image pixels, with non-scaling strokes and
   labels sized through the stage's scale, toned by lab-ui's `toneColor`), then whatever else a screen draws
   in image pixels, such as the peak marker. Boxes are drawn client-side rather than rendered into a PNG
   because a detection result is a few shapes whose colour is decided per shape. `ZoomPanCanvas` is no longer
   used anywhere in the app.
5. **A window shortcut goes through `useHotkeys`, never a bare `keydown` listener.** Every screen with
   shortcuts used to carry its own guard, and each missed a different case: the sample viewer relabelled on
   ⌘D, the experiment sample page paged away while its slider thumb (`span[role=slider]`, not an `<input>`)
   had focus, and the annotation editor completed the document behind an open dialog. `hotkeyBlocked`
   (`hooks/useHotkeys.ts`) is the one rule — text entry, lists and menus, navigation keys on a slider, tab
   strip or radio group, any open dialog, and held modifiers unless the screen opts in — and is unit-tested
   case by case.
6. **A screen's keys are data, and the list is also its documentation.** The annotation editor's keymap is
   `EDITOR_BINDINGS` (`components/annotation/editorKeys.ts`): each binding names its command, the keys a
   reader sees, a sentence and a matcher. `useEditorKeymap` resolves a keystroke by walking that list, the
   `?` shortcut sheet renders it, and the tool rail's tooltips read their keys from it — so a key cannot be
   bound and undocumented, or documented and dead, which is where the hand-written hint line under the
   queue buttons had drifted to. What a command *does*, including when it declines (J/K with unsaved work,
   Enter under three vertices), stays in the caller's action beside the state it reads.

## The annotation editor's shape

The editor is the largest screen, and it is split by seam rather than by panel.
`routes/AnnotationEditorRoute.tsx` only composes; each concern is a hook in `routes/annotation/` —
`useDraftSession` (history, `ETag`, the one shared save flight, autosave, completion, discard, the unload
guard), `useQueueNavigation`, `useChannelPanes` (reference pane, save-then-navigate channel switching, copy
to channels), `useDocumentCommands` (selection, the open polygon and every edit), `useSegmentAssistSession`
and `useEditorKeymap` — with each panel a component beside them.

**An async edit is built from the document as it will be, not as the render that started it saw it.**
`useDraftSession` runs the pure history reducer eagerly as well as in React, and `latest()` returns that
document; every edit in `useDocumentCommands` is built from it. Painting into a region awaits a PNG decode,
and a stroke that committed the `history.present` its closure captured discarded whatever was dispatched in
the meantime — six of eight back-to-back keyboard stamps were lost, and an undo pressed mid-decode came back.
Strokes are now queued, each reads the document and the selection when it starts, and one that finds the
document moved when its decode returns is painted again rather than committed over the move.

**The canvas re-renders what moved.** `AnnotationCanvas` wires three pieces: pure per-tool modules
(`components/annotation/tools/`) that turn pointer and key input into effects; a memoised static Konva
layer (`SceneLayer`: photograph, blended channel, base mask, committed regions); and a live layer
(`LiveLayer`: brush trail, open polygon, assist prompts, keyboard and brush cursors) fed by a small external
store, so a pointer move re-renders the live layer and nothing else. A brush trail is appended in place,
never copied per sample. The view arithmetic is `canvasView.ts`: zoom is stored as a multiple of fit, so one
view serves a reference pane beside the editor, but its ceiling is lab-ui's `MAX_SCALE` — 32 screen pixels
per *source* pixel — and smoothing is off above 1:1 so pixels render as squares.

The rules that a pattern can recognise — no raw `<table>`, no hex colour or Tailwind ramp step, no bare
`<select>`, range or checkbox input, no `<details>`, no `<Button>` directly inside a `<Link>`, no `keydown`
listener outside `useHotkeys` — are held by `frontend/src/uiRules.test.ts`. It is a ratchet rather than a
ban: the occurrences that remain are listed per file with their count, so a new one fails the suite and so
does a fixed one whose allowance was not lowered. What a pattern cannot see — hierarchy, states, whether the
next step is reachable — is the visual pass (`.claude/skills/lab-visual-pass`).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
