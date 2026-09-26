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
  images (ADR-0041); a channel count appears only when a sample has more than one image; root path, adapter
  and import date sit behind an information mark. A tab renders no heading, strip or back link, and gives
  its main surface the single scroller through `TabScroll` or its own full-bleed surface (the browser).
- **`CanvasLayout`** — no page scroll; an image canvas fills the viewport and supporting panes scroll only
  on their own content. The sample viewer, the annotation editor and the guided run live here; the
  guided run's one scroller is its step (`data-scroll="step"`), between a band and a footer that stay.

**A failed read shows its error promptly** (`api/retry.ts`). A 4xx is never retried, since it is the
request's fault and says so. A 5xx or a dropped connection is retried once, in case the sidecar is
restarting. React Query's default of three retries with backoff meant about seven seconds of skeleton
before any error.

**Exactly one page-level scroller per screen.** A bounded pane may scroll on the same axis only as a peer
**column** beside the content — the browser's filter rail and the sample viewer's control rail. A log tail
or ranked list in the flow is clipped behind a disclosure instead. `frontend/src/routes/dataset/tabScroll.test.tsx`
asserts it.

**A control never nests inside a link.** Card actions and grid selection boxes are absolutely-positioned
siblings of their `<Link>`: a control inside an anchor must cancel the click, and cancelling a checkbox's
click makes the browser restore its old state after React writes the new one. A navigation that looks
like a button is lab-ui's `ButtonLink` — one anchor with the button's variants — never a `<Button>`
inside a `<Link>`.

**Navigation.** The main navigation is `Datasets`, `Experiments` and `Compare`; import is an action in the
catalogue, and backend health stays visible in the shell. Inside a dataset the strip is grouped by the stage
of the work, in the order it is done: **Data** (`Browse`, `Prepare`) · **Truth** (`Annotate`) · **Runs**
(`Splits`, `Experiments`). It stays one row, so the band keeps its height: the stage names are quiet labels
between the links, folded away below `md`, and each stage is a `role="group"`. The links are underlined
because pills mark in-page state.

**The front door is the guided run.** The band's primary action, at every width, is **Start a run**
(`/datasets/{id}/run`, below): it asks for nothing in advance, so the band carries no checklist of what
to do first. **New experiment**, the full form, sits beside it as a quiet link for a reader who knows
what they want, and the Prepare and Splits tabs stay the expert surface. The import's finished screen,
each dataset a reference-pack registration added, and a catalogue card's corner offer the same **Start
a run** beside **Browse**.

**Readiness** (`hooks/useDatasetReadiness.ts`) is still what decides whether a task can run, per task
that some method declares. No region profile is asked for — a run reads the dataset's "Full frame"
unless it names another, and its first job prepares it at its size; `anomaly` needs a split drawn or
adopted for it; `few_shot_segmentation` needs a class with a reference and something to test on (from
the coverage read) and a split of references; `semantic_segmentation` and `object_detection` each need
the same annotated class and a `class_stratified` or `manual` split (`splitServesTask`), one presence
rule serving both. The guided run reads it to tell a task that needs annotation first from one a preset
can start; a missing split is not a blocker there, since a preset is drawn when the run starts. A task is
named only when the dataset's truth can serve it (`truthServesTask`, `api/truth.ts`): a dataset of
classes alone is not offered `anomaly`, and the create form opens on the first task it is offered.

**Screens follow the dataset's truth** (ADR-0041). `DatasetSummary.truth` says whether a dataset holds
anomaly labels, classes, both or neither, and `api/truth.ts` turns that into one question per surface:

- **Counts** (`components/DatasetTruth.tsx`) — a dataset with labels is counted in verdicts
  (`1000 normal · 100 defect`); one with classes in classes (`20 classes`), each class with the samples
  that show it behind an information mark. A dataset of classes never shows a run of zero defects.
- **Anomaly labelling** (`labelsApply`) — the label rail, its `n`/`d`/`u` keys, the verdict badges and bulk
  labelling are there for a dataset with labels and for one with no truth yet, since labelling is how an
  import becomes an anomaly dataset. A dataset of classes alone hides them; the sample viewer offers
  *Label for anomaly detection* instead, which brings the rail back for the visit, and one verdict makes
  it a dataset with labels.

**Vocabulary.** A *region profile* is where a run looks — the crop recipe defined on Prepare, with no size
of its own; the *input size* is the run's; *Colour* is the
experiment's colour option; the *threshold* is a run's image-score cut; the *map cut* is the fraction of a
run's map range that draws its segmentation; the *threshold rule* is how Compare picks each run's
threshold. A method is shown by its title, its registry key as secondary text.

## The guided run

`/datasets/{id}/run` (`routes/GuidedRunRoute.tsx`, `routes/run/`) takes a dataset to a queued run as
five decisions, each made in front of images and each with its default already chosen, so pressing
Next through all of them starts a sensible run and **Review & run** jumps from any step to the last.
`useGuidedRun` resolves every choice: what the reader picked, else the default the rest of the
workbench already states.

1. **Goal** — a card per task some method runs and the truth can serve, each with its meaning and a
   strip of what it learns from: defect samples for anomaly detection, the target class for few-shot,
   one sample of each class for segmentation and detection, each with its truth's outline from
   `GET /api/images/{id}/mask`. The suggestion (`suggestedTask`, `api/guidedRun.ts`) follows the truth:
   verdicts ask anomaly detection, classes drawn as boxes (`DatasetDetail.class_geometry`,
   [domain model](domain-model.md)) ask detection, classes drawn as regions ask few-shot segmentation;
   it comes first and is badged. A task that needs an annotated class says so and links to Annotate
   instead of being offered. Few-shot adds the class beside its card, the most frequent by coverage.
2. **Look** — "Full frame" or any saved region profile, as cards, over Prepare's own `LiveStage`: one
   image, source and prepared frame, at the size the chosen method resolves to
   (`POST /api/experiments/input-size`), prepared by `POST /api/datasets/{id}/region-preview`; ←/→ step
   through Prepare's filmstrip images. **Adjust on Prepare** opens `…/prepare?profile=&return=run`, where
   a banner leads back to `…/run?step=look&profile=<the saved revision open there>`, and the run takes it.
3. **Split** — the task's presets first, the first chosen, then the dataset's own splits that serve the
   task (a few-shot split only for the chosen class), each with its `SplitComposition` bar; the chosen
   one adds a few thumbnails per subset from the composition's `examples`. A few-shot preset for another
   class than its own is a fresh dry run, as on the Splits tab. Nothing is created here.
4. **Method** — `MethodCard`s in the registry's order, with the methods this installation can run first:
   a front door that opens on a method missing its dependencies has chosen a run that cannot start, and a
   callout says so when the recommendation is the one missing. Beside the cards, held in view on a wide
   window, the method's `SchemaForm` — `x-primary` fields in front, the rest under Advanced — and the
   resolved input size.
5. **Run** — every choice in a summary whose lines link back to their step, a name defaulting to
   `<method> on <dataset>` (with the class for few-shot), and **Start run**: `hooks/useLaunchRun.ts`
   creates the split if it is still a preset, creates the experiment and queues Train & score
   (`then_score`, the run bar's first primary), then lands on the run page. A split made before a later
   stage failed is kept in the state and reused by the next press. What is missing is said beside the
   disabled button.

A rail across the top shows each step with what it is set to; every step up to the furthest reached is
a button, a later one is text. The step is in the URL (`?step=`), so the browser's Back walks the steps;
the choices are in `sessionStorage` per dataset (`anomaly-lab:guided-run:<id>`), so a detour comes
back to the same run, and they are cleared when the run starts. Enter continues and Shift+Enter goes
back, through `useHotkeys`, except where a focused button or link answers Enter itself. A read that
failed is one `ErrorBox` above the step, never "nothing fits". From the dataset page a first run is three
presses — Start a run, Review & run, Start run — and six by Next alone.

## Screens

**Dataset catalogue and import** — a grid of covers grouped by collection, each group headed by its name
and a muted `N datasets` badge, so a count never reads as part of the name. A card is its cover, its name,
its description and one quiet line of counts in the unit of its truth (`200 samples · 20 classes`); its
corner, revealed on hover and focus and a sibling of the card's link, holds Start a run, edit and delete. Local VisA/GKN/FSS-1000/PKU-Market-PCB packs register
in one job, whose success stays on screen listing each dataset it added with Browse and Start a run
(from the job's `dataset_ids`) until dismissed; a folder is scanned, its manifest reviewed (channel
mapping, labels, warnings) and committed, and the finished screen offers Start a run beside Browse.
The scan job and manifest are in the URL (`scan=`, `manifest=`). A collection is a string on each dataset
([domain model](domain-model.md)), so `CollectionDialog` names and fills it in one form via
`PATCH /api/datasets/{id}`; unticking clears the override. Deletion requires the exact name and previews
the cascade; source images and masks are never touched.
`GET /api/reference-packs`, `POST /api/reference-packs/register`, `POST /api/import/scan`,
`POST /api/import/commit`, `GET /api/datasets`, `PATCH /api/datasets/{id}`,
`GET /api/datasets/{id}/deletion-preview`, `DELETE /api/datasets/{id}`.

**Browser** — class/label/channel/split/subset filters in a 256 px rail; the virtual image grid is the only
vertical scroller. The class filter is offered for a dataset with classes and lists each with its sample
count (`class=` in the URL, sent as `class_key` with `presence=present`, and carried into bulk labelling
as `class_key`); the label filter only for a dataset with labels. A filter already in the URL stays on
screen so it can be cleared. A channel filter selects whole samples having that channel and previews it on each
tile; a sample lacking it falls back to its first image. A dataset's default display channel (set in its
edit dialog when it has more than one) answers wherever one image stands for a part, with the rail's filter
winning over it. `GET /api/datasets/{id}/samples`, `GET /api/images/{id}/thumb`.

**Sample viewer** — one part, every channel, count driven by data, opening on the filtered channel, else
the dataset default, else the first. Label editing with keyboard shortcuts where anomaly labelling applies
(above); full-resolution zoom; the label, channel and file controls in a 288 px rail. `GET /api/datasets/{id}/samples/{sid}`,
`PATCH /api/datasets/{id}/samples/{sid}`, `GET /api/images/{id}/preview`, `…/full`.

The viewer **draws each image's truth**, on by default (the View section's *Truth* switch and its opacity
slider). `GET /api/images/{id}/truth` answers per image by the presence rule every class read shares
([annotations](annotations.md)) — the newest completed revision, else an imported mask, else a normal
label, else nothing — so a sample-scoped annotation, materialised on every channel, draws on each pane
with no channel count anywhere. A revision's drawn `add` boxes come back as rectangles and are drawn as
tagged `VectorLayer` boxes in their class's colour; every other region is one raster,
`GET /api/images/{id}/truth/regions.png`, rasterised without those boxes and filled in each class's colour,
or — for an imported mask, which is anomaly truth — outlined. Colours and names are the dataset's own
taxonomy (the editor's), so they travel as data and the legend beside the switch lists the classes any
channel shows with their chips, and says where the truth comes from. Layer order is fixed
(`routes/sample/truthLayers.ts`): truth, then Explore above it; while Explore is on, truth keeps
`EXPLORE_DIM` of its weight and the switch still hides it. Completing an annotation or recolouring a class
invalidates the image's truth.

**Explore** — the sample viewer's rail section for *what a frozen encoder sees*, for intuition rather than
truth (`routes/sample/`). Off until switched on; then a click on the picture — `SampleStage`'s `onPick`,
a background click that did not become a pan, projected through the stage's own transform — asks one
question. **Similar**: a click adds a positive patch, shift-click a negative, and the similarity heatmap
and its thresholded mask are two raster layers; the heatmap is coloured over this image's own range and
the rail prints it (*colour scaled to this image · median → top 1%*), while the mask threshold stays in
absolute cosine. **Clusters**: K (2–12) is a slider committed on release;
a click picks the cluster under it from the cells the answer carried (`api/explore.ts` replays the
transform), and only that cluster is drawn. **PCA**: false colour. **SAM**: MobileSAM through the editor's
own `POST /api/images/{id}/segment-assist`, its ranked candidates tinted as suggestions. **Text**: a
phrase submitted from an `Input` asks SAM 3 (`routes/sample/useExploreTextSession.ts`); its instances are
one label-map layer, one series colour each, listed as `ToggleChip`s with their scores and swatches, and
picking one draws its whole mask alone, as picking a cluster does. The first phrase on an image says it is
encoding the image and may be loading SAM 3; the answer prints whether the image was cached and how many
weaker instances were not kept; nothing found says so. Until the checkpoint is installed the mode is where
it is installed: the capability's reason (the gated message when no Hugging Face token is present), the
licence and access links, *Accept licence & download*, then the download job's progress with a cancel.
Every overlay is
a source-sized PNG in `layers` with the opacity slider's weight, and points are `MeasureOverlay` dots.
The prompt belongs to one image: clicking another channel's pane moves Explore there, so no channel
count is special. The encoder picker lists only the capability's usable encoders, defaulting to DINOv2
ViT-B/14; the first request on an image names the encoder it is encoding with, a 409 says a job holds the
device and offers to ask again, and a missing `dl` extra is said instead of offering controls.
**Send to editor** turns the thresholded similarity, the picked cluster, the chosen SAM candidate or the
picked SAM 3 instance into a `BitmapShape` (`POST /api/explore/maps/{id}/shape` for all but MobileSAM's)
and navigates to the editor with it
in the navigation's state; the editor's Contour assist shows it as a suggestion to accept — as a mask or
an editable contour, in the class selected there — or discard, and consumes the state so a reload does
not bring it back. `GET /api/explore/capability`, `POST /api/images/{id}/explore`,
`POST /api/images/{id}/explore/text`, `GET /api/explore/maps/{id}.png`,
`POST /api/explore/maps/{id}/shape`, `GET /api/model-assets`.

**Annotation queue and editor** — see [annotations](annotations.md) for the behaviour. MobileSAM takes
positive/negative points or a box and previews up to three ranked masks without mutating the draft; its
checkpoint downloads from the model-asset catalogue only after explicit licence acceptance.
`GET/PUT /api/datasets/{id}/annotation-scope`, `GET /api/datasets/{id}/annotation-labels`,
`/api/images/{id}/annotations/*`, `/api/samples/{id}/annotations/*`, `GET /api/segment-assist`,
`POST /api/images/{id}/segment-assist`, `GET /api/model-assets`.

**Region preparation** — preview first. The profile form sits in a sticky rail; the main column is a
**live stage**: the source image with the extractor's own box dashed and the padded crop solid, beside the
prepared frame exactly as a build writes it (the resized crop outlined, so the contain-pad bands show).
Every control change re-runs a synchronous preview of the *unsaved* form on the image on the stage,
debounced 250 ms (`useDebounced` holds back the request, not the controls); the last answer stays up,
dimmed and marked "Preparing…", until the next lands, and an extractor failure or a refused request is an
`ErrorBox` with its message. A filmstrip of 24 images spread over the dataset and every channel
(`preview_selection`, one per sample under a shared crop) is stepped with ←/→ through `useHotkeys`, and
**Random** puts any image of the dataset on the stage. The form opens on a saved revision (the newest
until one is chosen); **Save profile** appears only when the form differs from it — each option compared
at its effective value, typed or default — under an editable name made from the form ("Foreground
threshold · pad 5%"). **Check 24** runs the sampled `region_prepare` preview job on the unsaved form and
lists the crops failures first, each opening on the stage; a check whose form has since changed says so.
**Build all** prepares the *saved* revision and says why it is disabled (unsaved edits, or a size already
built). The preview size is a width and height pair, 448 × 448 until changed — the frame every DINO method
and most measured gates read — with a line saying a run prepares its own size when it trains, so building
only saves that run the wait; built sizes are buttons that select one. Revision and followed job are in
the URL (`profile`, `job`, `mode`, `jobProfile`); opened from a guided run's Look step, `return=run`
stays in it through every write and a banner leads back with the saved revision open. Deleting a
revision names the experiments pinning it.
`GET /api/region-extractors`, `GET/POST /api/datasets/{id}/region-profiles`,
`POST /api/datasets/{id}/region-preview`, `GET /api/datasets/{id}/region-preview/images?alignment=`,
`GET /api/datasets/{id}/region-preview/random`, `POST /api/datasets/{id}/region-check`,
`POST /api/region-profiles/{id}/build` (body `{width, height}`), `GET /api/region-profiles/{id}/builds`,
`GET /api/region-profiles/{id}/prepared/{image_id}?width=&height=`, `GET/DELETE /api/region-profiles/{id}`.

**Splits** — presets first, tuning optional. The top is a card per preset the dataset can serve
(`GET /api/datasets/{id}/split-presets`, [evaluation](evaluation.md#splits)): its task badges, a
one-line meaning, a composition bar with counts from the dry run, the name Create will give, and one
**Create** press; a few-shot card adds a class `Select` on the most frequent class, and another class
is a fresh dry run. `components/SplitComposition.tsx` draws every composition on the screen: the bar
divided by subset and within it by what the task reads — verdicts for an anomaly split, the drawn class
against the rest for `few_shot`, class colours for `class_stratified` (a range past eight classes).
**Custom split** is a `Disclosure` holding the strategy form, offering only the strategies the
dataset's truth can feed, each labelled with its task — the seeded draw, the published partition
(only with a published preset), **draw references for a class** (`few_shot`, defaulting to the most
frequent class, coverage beside the picker) and **draw annotated samples by class**
(`class_stratified`). Name and seed are optional and their placeholders are what the server will
derive; a debounced `POST /api/datasets/{id}/splits/preview` shows the composition as the form
changes. `?strategy=` opens the disclosure on a strategy, which is how a prerequisite link lands on
the one its task needs. Each existing split shows its tasks, its composition with a `browse` link per
subset, the experiments that ran on it, and **Delete**, whose `ConfirmDialog` reads the deletion
preview and is disabled, naming the runs, while any experiment holds it. With no split the empty
state points at the presets; with no preset a `Callout` points at labelling and annotation.
`POST /api/splits`, `GET /api/splits?dataset_id=`, `GET /api/splits/{id}/deletion-preview`,
`DELETE /api/splits/{id}`, `GET /api/datasets/{id}/annotation-labels/coverage`.

**Experiment catalogue** — dataset-scoped history first, a global view second. Filters and ordering live
in the URL and are applied in SQLite: a search that also matches a run's number (`12`, `#12`), any set of
methods as chips, a status, and a created-on day range. The server orders — by a header button (Created
flips newest/oldest; Name, Method, Status) — and pages by an opaque keyset cursor of `[sort value, id]`,
fifty rows per "Load more", so a run is never skipped or repeated while runs are added. The panel title
is the matching total, not the loaded count. Readers that need one dataset's or split's runs rather than
a catalogue (the Compare picker, readiness) take one page of up to 500 and say when there are more. A
checkbox column picks runs for Compare under the picker's own
`refusalReason`. The headline column is each run's own task metric, labelled (`api/headline.ts`: `0.912
AUROC`, `0.643 IoU`), because the evaluator names it (`headline_metric`, `headline_value`). Deletion
previews files, bytes, active-work blockers and resident eviction.
`GET /api/experiments?dataset_id=&q=&model_type=…&status=&created_from=&created_to=&sort=&cursor=&limit=`
→ `{items, total, next_cursor}`,
`GET /api/experiments/{id}/deletion-preview`, `DELETE /api/experiments/{id}`.

**Experiment creation** — a dedicated route, **task first**. When methods offer more than one task
(ADR-0039), step 1 is the task, because it decides everything after it: the split list offers only the
task's kind of split (`splitServesTask`), method cards are those whose `capabilities.tasks` include it,
and a targeted task adds a **Target class** select beside the split, defaulting to the class a `few_shot`
split was drawn for (ADR-0040). With no split its task can use, the Split field offers the task's
first preset in place — "Use Standard · 60/20/20, normals only — create it", with the target class for
a few-shot preset — which creates it in one press and is then preselected, and links to the Splits tab
opened on that task's strategy — `few_shot` for a targeted task, `class_stratified` for segmentation
and detection.
With one task the form starts at its inputs. The unsent form — the task and the target class
included, so a detour to Splits or the studio comes back to the same task — is kept in
`sessionStorage` (`api/experimentDraft.ts`). The
region profile defaults to the dataset's "Full frame" (`api/inputSize.fullFrameProfile`) and needs no
build. **Input size** is an optional width and height pair (`api/inputSize.ts`): empty means the method's
own, and the caption says what that resolves to for the current configuration — "448 × 448 · from
dino_memory", from `POST /api/experiments/input-size` — which follows the configured backbone. A typed
size snaps on blur to the multiple the method reads (a DINO backbone's patch) and the caption says why;
one side alone, or a size the multiple does not divide, is refused beside the fields. A lone split is
preselected; an empty name becomes `<method> on <dataset>`. Method, colour and evaluation forms are
**generated from JSON Schema**.
Method cards come in the registry's order of standing (`api/methodChoice.ts`): the task's recommended
method first, then `supported`, `experimental`, and the `floor` last, registry order within each; the form
starts on the first of them, so a task with a recommended method starts there. A card badges
`recommended`, `experimental` and `floor` from the listing's `status` and `recommended_for` — the verdicts
live in the registry ([methods](methods.md#status-and-the-task-default)), never here. The card is
`components/MethodCard.tsx`, shared with the guided run. Every method's schema marks its decisions
`x-primary`, and the method tab puts them in front and folds the rest under Advanced. The evaluation
tab shows only the fields the task's evaluator reads: a field carrying `x-tasks` is dropped for any
other task (`schemaForTask`), and the tab is absent when none remains. The primary is **Create & run**
— create, queue Train & score and land on the run, through the guided run's `useLaunchRun` — with
**Create only** beside it for a run that should wait as a draft.
`GET /api/experiments/model-types`, `POST /api/experiments/input-size`, `POST /api/experiments`.

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
resolved by `resolveSubset` to the last scored subset when none is named. Its overlay defaults are the
task's (`resultsDefaults`): heatmap and truth on for `anomaly` and `few_shot_segmentation`, prediction
and truth on with the map off for `semantic_segmentation` and `object_detection`. `readResultsState` and
`writeResultsState` both take the task, so the URL carries only departures from that task's defaults
and an untouched view's URL is clean; before the experiment loads the task is unknown and the anomaly
defaults stand in. Ranked per-sample scores, a
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
from `GET /api/experiments/{id}/segmentation-outcomes?subset=` for both segmentation tasks, or from
`GET /api/experiments/{id}/detection-outcomes?subset=` for detection, and asks none until the task is
known. What the registry holds per task is the gallery's outcome strip and rank
words, the note saying what an outcome is measured against, and the bodies of Overview, the metric tables
and Benchmark. The outcome vocabularies are disjoint (`tp`/`fp`/`tn`/`fn`; `hit`/`low_iou`/`miss`/
`false_presence`/`correct_absence`, `false_class` for a supervised run, and `mixed` for a detection
run), so one URL parameter and one "mistakes" set serve every task. A few-shot
Overview promotes IoU, Dice, presence ROC-AUC and the rate flagged on absent images, and tallies outcomes
per sample. Its tables are `segmentationRows`, with present / absent / unlabelled image counts. Its
Benchmark shows present samples by IoU band and absent samples by outcome. How overlap moves with the
number of references is a question across runs, and is open. A `semantic_segmentation` run's verdict
is per sample, pooled over its labelled images ([evaluation](evaluation.md)): `miss`, `false_class`,
`false_presence`, `correct_absence`, or `hit` / `low_iou` by mean IoU. Overview promotes mean IoU,
pixel accuracy, mean class accuracy and frequency-weighted IoU, then a lab-ui `Table` of each class's
IoU across the scored subsets (background and the mean beside them), the stored confusion matrix of the
current subset drawn as a `Table` whose cells are each true row's share, toned `normal` on the diagonal
and `defect` off it at five token strengths, and the outcome tally. Its tables are `semanticRows` — the
summary, then one IoU per pinned class, with labelled / unlabelled image counts — and its Benchmark shows
samples by mean-IoU band and absent samples by outcome. Its split list is every strategy but `few_shot`.
On the sample page its layers are label maps, not a cut and a mask: `GET
/api/experiments/{id}/images/{iid}/labels[?truth=true]` serves the method's map or the truth over the
pinned classes as a value plane of class indices, and `LabelLayer` paints it on a canvas inside the
`SampleStage` (`components/viewer/labelPaint.ts`) — prediction solid (faint fill, full outline), truth
dashed (outline only), background and ignored pixels clear. Class `i` is lab-ui's `seriesColour(i - 1)`,
by pinned position, never the dataset's own label colour, and the overlay row carries the legend.
The Samples tab is the same set: `GalleryTab` passes the run's `classes` to `OverlayControls`, so its
toggles are prediction / truth / foreground with the class legend, and each tile lays
`GET /api/experiments/{id}/images/{iid}/label-map?colours=[&truth=true]` over its thumbnail — the server
drawing `labelPaint`'s rule at the thumbnail's size, because a tile cannot afford a value plane each.
The colours travel in the URL, one per pinned class, from `classColour` (`labelMapUrl`), so lab-ui stays
the palette's only home. An anomaly or few-shot tile draws the cut and the outline.
An `object_detection` run's verdicts are read at its subset's **confidence cut**, resolved by the
evaluator by one rule and stored beside the metrics ([evaluation](evaluation.md#object-detection)); every
screen that draws a verdict prints that rule and value (ADR-0028). Overview promotes AP@[.5:.95], AP50,
AP75 and recall, then a lab-ui `Table` of each class's AP, AP50, AP75 and recall across the scored
subsets (the mean beside them, class swatches from `classColour`), the cut per subset with the F1,
precision and recall it reaches and the rule printed, and the outcome tally (`hit`, `miss`,
`false_presence`, `mixed`, `correct_absence`; `mistakes` is the middle three). Its tables are
`objectDetectionRows`, with labelled / unlabelled image counts, and its Benchmark counts true and
predicted boxes per class and subset. On the sample page its layers are boxes: `GET
/api/experiments/{id}/images/{iid}/boxes` feeds `boxShapes` (`components/viewer/boxTones.ts`) into
`SampleStage`'s `VectorLayer` — truth dashed, kept detections solid and tagged with class and confidence
in the class's colour, each outline toned `normal` for a match, `defect` for a false positive and `warn`
for a missed truth box, a class the run does not pin `muted`; detections below the cut are not drawn.
The overlay row (`OverlayControls`' box mode) is prediction / truth / map with the tone key, the class
legend and the cut's rule, and a line under each image counts the kept detections and the found truth.
Each gallery tile lays `GET …/images/{iid}/box-map?colours=` over its thumbnail — one server-drawn SVG,
because a tile cannot afford a box request each; the three tones travel in the URL, read once per grid
from the theme's custom properties (`boxToneColours`).

**Reference studio** (`routes/StudioRoute.tsx`, `/datasets/{id}/studio/{class}`, a flush canvas) —
where a few-shot run's references are chosen by eye rather than drawn blind (ADR-0040). Reached from each
class in the class manager and from the Splits tab's reference draw.
- **Left rail:** the samples that show the class (`GET …/samples?class_key=&presence=present`), as the
  browser's own `SampleTile`, which takes an `onOpen` that focuses the stage instead of navigating. Its
  checkbox makes the sample a reference.
- **Centre:** the focused sample on `SampleStage`, with that class's outline
  (`GET /api/images/{id}/mask?class_key=`).
- **Right rail:** the references (one to ten), the method (the few-shot methods that are available,
  `proto_seg` first, the gate's default), the region profile ("Full frame" by default), and **Freeze as
  experiment**. The preview reads the method's own size and needs the profile built at it; when it is
  not, the blocker names the size to build on Prepare. Freezing makes a `manual` split of
  the references, a `few_shot_segmentation` run on the class and its Train & score
  (`hooks/useStudio.ts`), then lands on the run. Why it cannot freeze yet is said in words beside the
  button.

**The preview** segments the open image with the chosen method fitted on the current references, through
the resident worker (`POST /api/datasets/{id}/studio/preview`, [jobs](jobs.md)). It draws the foreground
probability under the class outline, and reads the presence score, the share of the image at 0.5 and
whether the resident was warm. The left rail can also list the samples *without* the class, or with no
answer for it, which is where a preview is most worth reading; only samples that show the class can be
ticked as references. **Least certain first** orders the rail's current page by how close each sample's
presence score is to 0.5 under the current references (`POST …/studio/preview-batch`, one resident
request of at most one page, `MAX_BATCH` = 48). The rail says it ordered a page, not the dataset.

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

**Diagnostics** — rendered by `kind`, never by method name ([diagnostics](diagnostics.md)): run-scoped entries in an
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

**Detection comparison** (`routes/compare/DetectionCompare.tsx`, `GET /api/compare/detection?ids=&subset=`).
Object detection runs of one dataset, split and class list compare on **threshold-free metrics alone**:
each run's AP family, recall, AP per class and timing on the anomaly table's grid, with a subset picker.
A run's confidence cut is its own, so the server strips it and what it reaches from the report and it
stays on that run's Overview (ADR-0028).

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
   **The schema also says which fields matter:** `"x-primary": true` on a property shows it in front,
   `false` folds it, and an unmarked field keeps the default rule — folded when it is optional with a
   working default.
3. **Opacity is client state; the threshold is a server read.** Opacity is CSS over a fetched PNG. The
   rule `score >= threshold` lives in Python only, so the threshold endpoint returns counts **and**
   classified rows together ([evaluation](evaluation.md)).
4. **Layer registration is structural.** `ImageStage` lays out at the image's own pixel size and carries
   the whole transform, so a layer at `inset-0` covers exactly the source frame. All sample viewers are
   `components/viewer/SampleStage.tsx`: the photograph at `tierFor(view)`, raster layers in order, then
   `VectorLayer` (boxes and polygons in image pixels, polygons filled even-odd, non-scaling strokes, toned
   by lab-ui's `toneColor`), then screen-specific overlays such as the peak marker.
   `frontend/src/routes/ExperimentSampleRoute.test.tsx` pins it.
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
