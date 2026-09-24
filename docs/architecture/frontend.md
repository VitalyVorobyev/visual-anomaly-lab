# Frontend

The frontend is an engineering tool: dense, keyboard-friendly, responsive, with no configuration beyond
what an experiment requires. Images, maps, charts and result tables are the focal surface; navigation,
filters and configuration are subordinate. Hierarchy comes from alignment, type and spacing before borders
or colour.

## Stack

A small one — `react-router`, TanStack Query, Tailwind, Konva for the annotation canvas — with the API
client generated rather than written (**ADR-0012**).

- **Routing is `react-router`'s `HashRouter`.** The bundle is served from Vite's dev server at `/`, the
  desktop WebView at `…/index.html` and `tauri://localhost`; a path-based router matches no route at the
  second and renders an empty document.
- **Server state is TanStack Query**, with polling and invalidation on top.
- **The API client is generated.** `scripts/gen-api-types.sh` starts a throwaway backend, reads
  `/openapi.json` and writes `frontend/src/api/generated.ts`. The file is committed, so `tsc` needs no
  backend, and CI regenerates it and fails on any diff. The generator runs in its own throwaway project on
  TypeScript 5, because openapi-typescript needs the TS 5 compiler API the frontend's TS 7 lacks.

## Design system

Colour, type, radius and the controls come from **`@vitavision/lab-ui`** (**ADR-0021**), shared by every
lab app. `frontend/src/styles.css` only imports it, plus this app's one rule: `html`, `body` and `#root`
are `height: 100%; overflow: hidden`.

- **The chrome is grey so the data can be loud.** One accent, `signal`, means "you can act here".
  `normal` / `defect` / `warn` are reserved for verdicts. Greys are true neutral, so they do not shift how
  a colormap's cool end reads.
- **Components name tokens — `surface`, `line`, `fg-muted`, `signal` — never a Tailwind ramp step.** A raw
  colour compiles, looks almost right, and ignores the theme.
- **Light and dark both ship**, with `light` / `dark` / `system` applied by an inline script before first
  paint, so `system` survives a reload as itself. The choice is stored under `anomaly-lab-theme`
  (`src/themeStorageKey.ts`); the script in `index.html` repeats the literal and must agree with it.
- **Controls are lab-ui primitives** (`Select`, `Dialog`, `Tooltip`, `Slider`, `Checkbox`, `Switch` on
  Radix; `Disclosure`, `Table`, `SchemaForm`, `ImageStage` and the rest). A raw `<details>` renders with no
  caret, because the base layer drops the UA marker. A primitive that needs improving is improved upstream.
  Tests query ARIA roles: a Radix `Select` is a `button[role="combobox"]` with a portalled listbox, and a
  Radix `Checkbox` is a `button[role="checkbox"]` with no `.checked`.
- **Focus is `focus-visible:outline-2 outline-signal`** — an outline, so it follows the element's radius.

## Shell capabilities

**Nothing under `frontend/src/` imports a Tauri API.** The shell injects `window.__ANOMALY_LAB__` before
the page loads, carrying the sidecar's base URL and what a browser cannot provide — a directory picker
returning an absolute path, reveal-in-Finder. `src/api/shell.ts` alone declares its shape and exposes a
`has<Capability>()` / `<capability>()` pair.

**A missing capability is a different affordance, not a broken one**: no directory picker means a text
field. The UI never renders a control disabled only because it is not in the desktop app; anything
expressible over HTTP belongs in the sidecar. The accepted cost: the contract is a hand-written global, so
a renamed Rust command fails at runtime, in the desktop build only.

## Crash reporting

An unhandled render error unmounts the React root and paints a black window, indistinguishable from a
hung shell. `components/CrashScreen.tsx`, wired in `main.tsx`, has two halves:

- **`CrashBoundary`** wraps everything outside the router and renders the error with its component stack.
- **`installCrashHandlers(container)`** runs before the first render, listening for `error` and
  `unhandledrejection` (a module throwing at evaluation, a lazy chunk that never arrives), and paints the
  same panel **only when the root is empty**.

The file imports nothing — no lab-ui, no Tailwind, no token — so it works when the stylesheet is what
failed.

**A backend that never started uses the same panel.** The shell always builds its window and injects
`startupError` instead of capabilities (failing inside macOS's `did_finish_launching` would abort the
process with nothing shown). `main.tsx` reads it via `shellStartupError()` before mounting anything, so no
router, query client or fetch is constructed, and the panel carries the shell's detail: paths searched and
the backend's last output.

## Layout and scroll

**The document cannot scroll**: the root chain is clipped and the shell frame is `h-full`, not
`h-screen`. Every route belongs to one of three layouts, marked with `data-layout`, `data-band` and
`data-scroll`:

- **`ReadingLayout`** — one outer vertical scroller, for catalogues, forms and tables.
- **`DatasetLayout`** (`routes/dataset/`) — owns the viewport without scrolling and renders the dataset's
  identity band and section strip **once, above all five tabs**. Facts are counted in **samples**, never
  images (ADR-0005); a channel count appears only when a sample has more than one image; root path, adapter
  and import date sit behind an information mark. A tab renders no heading, strip or back link, and gives
  its main surface the single scroller through `TabScroll` or its own full-bleed surface (the browser).
- **`CanvasLayout`** — no page scroll; an image canvas fills the viewport and supporting panes scroll only
  on their own content. The sample viewer and the annotation editor live here.

**Exactly one page-level scroller per screen.** A bounded pane may scroll on the same axis only as a peer
**column** beside the content — the browser's filter rail and the sample viewer's control rail. A log tail
or ranked list in the flow is clipped behind a disclosure instead. `frontend/src/routes/dataset/tabScroll.test.tsx`
asserts it.

**A control never nests inside a link.** Card actions and grid selection boxes are absolutely-positioned
siblings of their `<Link>`: a control inside an anchor must cancel the click, and cancelling a checkbox's
click makes the browser restore its old state after React writes the new one.

**Navigation.** The main navigation is `Datasets`, `Experiments` and `Compare`; import is an action in the
catalogue, and backend health stays visible in the shell. Inside a dataset the strip is grouped by the stage
of the work, in the order it is done: **Data** (`Browse`, `Prepare`) · **Truth** (`Annotate`) · **Runs**
(`Splits`, `Experiments`). It stays one row, so the band keeps its height: the stage names are quiet labels
between the links, folded away below `md`, and each stage is a `role="group"`. The links are underlined
because pills mark in-page state.

**Readiness** (`hooks/useDatasetReadiness.ts`) sits in the band, per task that some method declares. Every
run needs a built profile; `anomaly` needs a split drawn or adopted for it; `few_shot_segmentation` needs a
class with a reference and something to test on (from the coverage read) and a split of references
(`splitServesTask`). With one task the band is a checklist. With two, the shared first step comes first,
then each task shows a check or the link to its next step.

**Vocabulary.** A *region profile* is the crop-and-resize recipe built on Prepare; *Colour* is the
experiment's colour option; the *threshold* is a run's image-score cut; the *map cut* is the fraction of a
run's map range that draws its segmentation; the *threshold rule* is how Compare picks each run's
threshold. A method is shown by its title, its registry key as secondary text.

## Screens

**Dataset catalogue and import** — a grid of covers grouped by collection. Local VisA/GKN packs register
in one job; a folder is scanned, its manifest reviewed (channel mapping, labels, warnings) and committed.
The scan job and manifest are in the URL (`scan=`, `manifest=`). A collection is a string on each dataset
([domain model](domain-model.md)), so `CollectionDialog` names and fills it in one form via
`PATCH /api/datasets/{id}`; unticking clears the override. Deletion requires the exact name and previews
the cascade; source images and masks are never touched.
`GET /api/reference-packs`, `POST /api/reference-packs/register`, `POST /api/import/scan`,
`POST /api/import/commit`, `GET /api/datasets`, `PATCH /api/datasets/{id}`,
`GET /api/datasets/{id}/deletion-preview`, `DELETE /api/datasets/{id}`.

**Browser** — label/channel/split/subset filters in a 256 px rail; the virtual image grid is the only
vertical scroller. A channel filter selects whole samples having that channel and previews it on each
tile; a sample lacking it falls back to its first image. A dataset's default display channel (set in its
edit dialog when it has more than one) answers wherever one image stands for a part, with the rail's filter
winning over it. `GET /api/datasets/{id}/samples`, `GET /api/images/{id}/thumb`.

**Sample viewer** — one part, every channel, count driven by data, opening on the filtered channel, else
the dataset default, else the first. Label editing with keyboard shortcuts; full-resolution zoom; the
label, channel and file controls in a 288 px rail. `GET /api/datasets/{id}/samples/{sid}`,
`PATCH /api/datasets/{id}/samples/{sid}`, `GET /api/images/{id}/preview`, `…/full`.

**Annotation queue and editor** — see [annotations](annotations.md) for the behaviour. MobileSAM takes
positive/negative points or a box and previews up to three ranked masks without mutating the draft; its
checkpoint downloads from the model-asset catalogue only after explicit licence acceptance.
`GET/PUT /api/datasets/{id}/annotation-scope`, `GET /api/datasets/{id}/annotation-labels`,
`/api/images/{id}/annotations/*`, `/api/samples/{id}/annotations/*`, `GET /api/segment-assist`,
`POST /api/images/{id}/segment-assist`, `GET /api/model-assets`.

**Region preparation** — immutable profile configuration in a sticky rail, the crop audit central.
Revision and followed job are in the URL (`profile`, `job`, `mode`, `jobProfile`). Preview samples 24
images without writing; Build materialises atomically ([methods](methods.md#region-extractors)). Deleting a
revision names the experiments pinning it. `GET /api/region-extractors`,
`GET/POST /api/datasets/{id}/region-profiles`, `POST /api/region-profiles/{id}/preview`,
`POST/GET /api/region-profiles/{id}/build`, `GET /api/region-profiles/{id}/prepared/{image_id}`,
`GET/DELETE /api/region-profiles/{id}`.

**Splits** — create a seeded, stratified split, adopt the published one, or **draw references for a
class** (`few_shot`: a class that some sample shows, a shot count and a seed, with the class's coverage
beside the picker). Per-subset counts by label; immutable once created. `POST /api/splits`,
`GET /api/splits?dataset_id=`, `GET /api/datasets/{id}/annotation-labels/coverage`.

**Experiment catalogue** — dataset-scoped history first, a global view second. Filters and ordering live
in the URL and are applied in SQLite. A checkbox column picks runs for Compare under the picker's own
`refusalReason`. The headline column is each run's own task metric, labelled (`api/headline.ts`: `0.912
AUROC`, `0.643 IoU`), because the evaluator names it (`headline_metric`, `headline_value`). Deletion
previews files, bytes, active-work blockers and resident eviction.
`GET /api/experiments?dataset_id=&q=&model_type=&status=&sort=`,
`GET /api/experiments/{id}/deletion-preview`, `DELETE /api/experiments/{id}`.

**Experiment creation** — a dedicated route, **task first**. When methods offer more than one task
(ADR-0039), step 1 is the task, because it decides everything after it: the split list offers only the
task's kind of split (`splitServesTask`), method cards are those whose `capabilities.tasks` include it,
and a targeted task adds a **Target class** select beside the split, defaulting to the class a `few_shot`
split was drawn for (ADR-0040). With one task the form starts at its inputs. The band lists what is
missing as links in order, using the form's rule for "built" (`isUsableBuild`); the unsent form is kept in
`sessionStorage` (`api/experimentDraft.ts`). A lone profile or split is preselected; an empty name becomes
`<method> on <dataset>`. Method, colour and evaluation forms are **generated from JSON Schema**.
`GET /api/experiments/model-types`, `POST /api/experiments`.

**Run bar** — a draft's primary action is **Train & score** (`then_score`, [jobs](jobs.md)), with **Train
only** beside it; a trained run offers Retrain (confirmed), Continue for a resumable method, Score &
evaluate, and `Export ONNX` only when the method's capability declares it (otherwise a disabled,
explanatory affordance). The page follows the live job; a scoring run landing on Overview moves the reader
to Samples. `POST /api/experiments/{id}/train`, `…/infer`, `…/export`.

**Progress, logs and training charts** — progress bar, streaming log, cancel; snapshot-then-subscribe on
mount and reconnect. Loss curves survive a reload because history is re-read from the job log; a series
reported once is printed as a value. `GET /api/jobs/{id}`, `WS /ws/jobs/{id}`,
`POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/metrics`.

**Results** — Overview, Benchmark and Samples share one `ResultsState` in the URL (`t`, `subset`),
resolved by `resolveSubset` to the last scored subset when none is named. Ranked per-sample scores, a
threshold slider recomputing the confusion matrix, TP/FP/TN/FN links into Samples at the cut in force.
Benchmark draws ROC and PR at sample and image level, the score histogram with the threshold, confusion
matrix, per-defect-type breakdown and timing; every curve integrates to a number in the metrics table.
The **localization verdict rides beside the outcome, never inside it** ([evaluation](evaluation.md)): a
`localized` / `off target` badge (nothing for `null`) and `N of M` in the confusion strip, where `M` counts
rows carrying a verdict. An optional `peak` layer draws the stored argmax and its square tolerance window,
`null` toned `signal`. `GET /api/experiments/{id}/results?subset=`, `…/threshold?value=`,
`…/curves?subset=`, `GET /api/experiments/{id}/samples/{sid}/images`,
`GET /api/images/{id}/anomaly-map?experiment_id=`, `GET /api/images/{id}/mask`.

**Results by task** (`routes/experiment/taskViews.tsx`). The shell is shared — run bar, tab set,
`ResultsState`, the gallery's grid, the sample page — and so is the data flow: `useVerdicts(experiment,
state, task)` hands every screen the same classified rows, from the threshold report for `anomaly` or
from `GET /api/experiments/{id}/segmentation-outcomes?subset=` for `few_shot_segmentation`, and asks
neither until the task is known. What the registry holds per task is the gallery's outcome strip and rank
words, the note saying what an outcome is measured against, and the bodies of Overview, the metric tables
and Benchmark. The outcome vocabularies are disjoint (`tp`/`fp`/`tn`/`fn`; `hit`/`low_iou`/`miss`/
`false_presence`/`correct_absence`), so one URL parameter and one "mistakes" set serve both. A few-shot
Overview promotes IoU, Dice, presence ROC-AUC and the rate flagged on absent images, and tallies outcomes
per sample. Its tables are `segmentationRows`, with present / absent / unlabelled image counts. Its
Benchmark shows present samples by IoU band and absent samples by outcome. How overlap moves with the
number of references is a question across runs, and is open.

**Reference studio** (`routes/StudioRoute.tsx`, `/datasets/{id}/studio/{class}`, a flush canvas) —
where a few-shot run's references are chosen by eye rather than drawn blind (ADR-0040). Reached from each
class in the class manager and from the Splits tab's reference draw.
- **Left rail:** the samples that show the class (`GET …/samples?class_key=&presence=present`), as the
  browser's own `SampleTile`, which takes an `onOpen` that focuses the stage instead of navigating. Its
  checkbox makes the sample a reference.
- **Centre:** the focused sample on `SampleStage`, with that class's outline
  (`GET /api/images/{id}/mask?class_key=`).
- **Right rail:** the references (one to ten), the method (the few-shot methods that are available,
  `fss_dino` first), the region profile, and **Freeze as experiment**. Freezing makes a `manual` split of
  the references, a `few_shot_segmentation` run on the class and its Train & score
  (`hooks/useStudio.ts`), then lands on the run. Why it cannot freeze yet is said in words beside the
  button.

**The preview** segments the open image with the chosen method fitted on the current references, through
the resident worker (`POST /api/datasets/{id}/studio/preview`, [jobs](jobs.md)). It draws the foreground
probability under the class outline, and reads the presence score, the share of the image at 0.5 and
whether the resident was warm. The left rail can also list the samples *without* the class, or with no
answer for it, which is where a preview is most worth reading; only samples that show the class can be
ticked as references.

**This image's truth.** From what the stage shows, the studio writes ordinary annotation revisions
(`POST /api/images/{id}/studio/region`):
- **accept** the preview's region as the class's truth;
- **fix** it, which opens the same region as a draft and lands in the editor;
- **mark the class absent**.

The region is read from the preview's own map on the server, never sent by the client
(`annotations.class_region_document`, [annotations](annotations.md)). Accepting a sample makes it
eligible as a reference; the pseudo-label loop is accept, then tick.

The session is the URL (`refs`, `focus`, `method`, `profile`, `show`). The studio evaluates nothing: the
run page is the one place a run is read. Both rails are `RailSection`s (`components/viewer/`), shared with the
sample viewer.

**Diagnostics** — rendered by `kind`, never by method name (ADR-0018): run-scoped entries in an
*Architecture* tab (`graph`, `table`) and an *Inspector* tab (`map`, `image`, `grid`); image-scoped entries
beside the combined map on the sample page. Diagnostic panes are in the prepared frame
([diagnostics](diagnostics.md)), so their outline is fetched with `frame=prepared`.
`GET /api/experiments/{id}/diagnostics`, `…/diagnostics/payload?key=`.

**Comparison** — N runs of one dataset and split. Reached from the top bar, a run's "Compare with…", or
the catalogue checkboxes; once one run is picked the picker shows only its dataset and split and prints each
refusal on the row. Threshold-independent metrics compare directly; threshold-dependent ones are resolved
**per run by one shared rule**, each run's cut printed beside its matrix (**ADR-0028**). Overlaid curves, a
config diff calling out preprocessing, a disagreement-filtered sample table, and a map view sharing one cut
*fraction* and one `StageView` across panes. `StageView.scale` is absolute (CSS pixels per image pixel),
which is correct because every pane draws the same image. Localization verdicts and `peak` layers are per
pane. `GET /api/compare?ids=&subset=&at=`.

**Few-shot comparison** (`routes/compare/FewShotCompare.tsx`, `GET /api/compare/few-shot?ids=`). The first run
picked decides which comparison it is. Few-shot runs of one dataset and one class compare **across
reference draws**, so their splits may differ where the anomaly comparison refuses it; the picker groups
them by class, and `refusalReason` refuses another task or another class by name. It shows three things:
- **Foreground IoU by method and shot count:** the mean over draws ± their spread, which is how much the
  answer depends on which references were chosen.
- **Every segmentation metric per run**, on the anomaly table's own grid, each column naming its reference
  count and seed.
- **Where they disagree:** the test queries every run scored (a run's own references are no one's query),
  with each run's outcome and IoU, linked to that run's sample page.

## Cross-cutting rules

1. **No channel count is hard-coded.** Layouts render from the dataset's channel dictionary; a two-channel
   sample needs no special case.
2. **No method is named.** Picker, forms and capability-driven affordances come from
   `GET /api/experiments/model-types` (ADR-0007). lab-ui's `SchemaForm` maps a pydantic JSON Schema to
   controls: `enum` is read before `type` and `$ref` is resolved through `$defs`, so `Literal` and
   `StrEnum` both become pickers — three or fewer values a segmented control, more a select; `minimum`,
   `maximum` and `multipleOf` reach the control, and a float without `multipleOf` steps `any`.
   **An empty control means unset:** `toOptions` sends nothing for an untouched field, so a default is
   defined in Python alone. A segmented control highlights the effective value and stores `""` when it is
   the schema default; a select carries an explicit `Default · <value>` entry. Do not pre-fill.
3. **Opacity is client state; the threshold is a server read.** Opacity is CSS over a fetched PNG. The
   rule `score >= threshold` lives in Python only, so the threshold endpoint returns counts **and**
   classified rows together (ADR-0011).
4. **Layer registration is structural.** `ImageStage` lays out at the image's own pixel size and carries
   the whole transform, so a layer at `inset-0` covers exactly the source frame. All sample viewers are
   `components/viewer/SampleStage.tsx`: the photograph at `tierFor(view)`, raster layers in order, then
   `VectorLayer` (boxes and polygons in image pixels, non-scaling strokes, toned by lab-ui's `toneColor`),
   then screen-specific overlays such as the peak marker. `frontend/src/routes/ExperimentSampleRoute.test.tsx`
   pins it.
5. **A window shortcut goes through `useHotkeys`.** `hotkeyBlocked` (`hooks/useHotkeys.ts`) is the one
   guard — text entry, lists and menus, navigation keys on a slider, tab strip or radio group, any open
   dialog, held modifiers unless opted in — unit-tested case by case.
6. **A screen's keys are data.** The editor's keymap is `EDITOR_BINDINGS`
   (`components/annotation/editorKeys.ts`): command, displayed keys, sentence, matcher. `useEditorKeymap`
   resolves against it, the `?` sheet renders it and tool tooltips read from it, so no key is bound and
   undocumented. What a command does, including when it declines, stays in the caller.

## Annotation editor structure

`routes/AnnotationEditorRoute.tsx` only composes; each concern is a hook in `routes/annotation/` —
`useDraftSession` (history, `ETag`, the shared save flight, autosave, completion, discard, unload guard),
`useQueueNavigation`, `useChannelPanes`, `useDocumentCommands`, `useSegmentAssistSession`,
`useEditorKeymap` — with each panel a component beside them. View state (pane mode, second channel, tool,
brush size, mask weight, pan/zoom) lives above the keyed editor and survives a channel change; document
state (history, selection, pending polygon) is keyed by the draft target. Pan/zoom carry a frame stamp, so
they follow the reader across one part's channels and reset elsewhere.

**An async edit is built from the document as it will be.** `useDraftSession` runs the pure history reducer
eagerly as well as in React, and `latest()` returns that document; every edit in `useDocumentCommands`
builds from it.

**The canvas re-renders what moved.** `AnnotationCanvas` wires pure per-tool modules
(`components/annotation/tools/`) that turn input into effects, a memoised static layer (`SceneLayer`:
photograph, blended channel, base mask, committed regions) and a live layer (`LiveLayer`: brush trail, open
polygon, assist prompts, cursors) fed by a small external store, so a pointer move re-renders only the live
layer. A brush trail is appended in place. `canvasView.ts` stores zoom as a multiple of fit, capped at
lab-ui's `MAX_SCALE` — 32 screen pixels per source pixel — with smoothing off above 1:1.

## UI rule ratchet

`frontend/src/uiRules.test.ts` holds what a pattern can recognise: no raw `<table>`, hex colour or ramp
step, no bare `<select>`, range or checkbox input, no `<details>`, no `<Button>` inside a `<Link>`, no
`keydown` listener outside `useHotkeys`. Remaining occurrences are listed per file with their count, so a
new one fails and so does a fix whose allowance was not lowered. What a pattern cannot see is the visual
pass (`.claude/skills/lab-visual-pass`).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
