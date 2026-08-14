# Annotation truth

Imported masks and editable annotations are deliberately different things (ADR-0032). A `Mask` points at
the source dataset and is never rewritten. An `AnnotationDraft` and its completed `AnnotationRevision`s live
under application ownership. This is what lets a benchmark remain reproducible while a person improves its
truth.

## The raster contract

**A bitmap shape's mask channel is luminance.** The backend defines it — `decode_png` is
`convert("L") > 0`, `encode_png` writes mode `L` on an opaque black ground — and every producer
follows: a brush stroke, an accepted MobileSAM candidate, an imported PNG / LabelMe / COCO mask.

The frontend converts luminance to *alpha* only when painting, which is what lets one contract
serve both an editor overlay and an opaque stored mask. It did not always: rendering relied on
transparency and tracing read the alpha byte, which agreed with the backend only because a brush
stroke happens to be white on transparent black. Anything the backend produced arrived fully
opaque, so it drew as a grey rectangle over its whole crop and traced as its bounding box.

Regions are painted in their label's colour, cuts (`operation: "subtract"`) in one fixed colour
with a dashed outline so an eraser layer never looks like a brush layer, and the whole scene
resolves its colours from the design tokens at runtime — Konva cannot take a class name, so
`scenePalette.ts` reads `styles.css` and repaints on a theme change rather than hardcoding one
palette (ADR-0021). Mask weight is a persisted per-reader preference; the label colour is dataset
taxonomy and is edited through the existing `PUT .../annotation-labels/{key}`.

The seeded `defect` class is magenta (`#c026d3`), not red. A mask sits over the photograph at partial
opacity for minutes at a time while somebody works: red reads as an error state, and over a metal part
in dark field it turns muddy brown. Magenta stays legible on metal, on plastic and in dark field, and
cannot be mistaken for the teal `signal` accent drawing the selection outline on top of it. Migration
017 moved existing datasets, touching only labels still carrying the old default — a colour somebody
chose through the swatch is theirs.

## Coordinate and document contract

An annotation document is JSON schema version 1 in **source-image pixel coordinates**. It pins
`image_width`, `image_height`, a base (`empty` or `source_mask`), and an ordered list of shapes. The first
shape is a polygon with a stable id, dataset taxonomy key, `add` / `subtract` operation and three or more
points. A bitmap shape is a cropped binary PNG with the same identity, taxonomy and operation fields plus
its integer source-frame rectangle. Duplicate shape ids, geometry outside the source frame, malformed bitmap
bytes, unknown label keys, dimension changes and base-layer changes are rejected before persistence.

The source frame is load-bearing. A future crop/localisation profile may change what a method sees, but an
annotation never moves into model-input or canvas coordinates. The UI transform and the spatial input
pipeline must map back to this frame.

## Scope: what one annotation is *of*

A dataset annotates either each image or each whole sample, and `Dataset.annotation_scope` says which
(ADR-0036). `image` is the default and the original behaviour. `sample` is for a multi-shot rig whose
channels are exposures of one registered part: one document is edited once and materialised as **one
ordinary `AnnotationRevision` per image of the sample**. Truth below that boundary is unchanged —
`resolve_ground_truth_masks`, pixel metrics, `has_mask`, the `MetricSet` digest and all three
interchange formats stay image-keyed and never learn that scope exists.

`GET`/`PUT /api/datasets/{id}/annotation-scope` reads and moves it, and the read reports **every**
reason sample scope is unavailable rather than failing on the first: imported source masks (pinned per
image, so a shared document cannot carry one image's provenance), samples whose images differ in
dimensions (a shared document pins one source frame), and open drafts. Leaving sample scope is refused
while any sample draft is open. Completed revisions are untouched in either direction. Under `sample`
scope the per-image write routes return `409` naming the sample route, because two writers editing one
part through two scopes would each hold a valid ETag for a different document.

A sample-scoped document is always `base="empty"`, enforced rather than assumed, and its draft lives in
its own `annotation_sample_draft` table with its own ETag namespace
(`annotation-sample-draft-{sample}-v{n}`).

## Draft lifecycle

**A draft records work, not a page view.** It is created by the first save and by nothing else, which is
what makes "how many drafts are open" the same question as "how much unfinished work is there" — the
question `annotation_scope` has to answer (ADR-0036).

`GET /api/images/{id}/annotations/draft` is **read-or-seed** and never writes. It returns the draft in
progress with its ETag, or — with `persisted: false`, a null `version` and no ETag — the document a new
draft *would* open on: the newest completed revision, the imported mask, or an empty canvas. One helper
produces that seed for both the read and the create, so the two cannot disagree about what a draft starts
from. The read deliberately does not hash or record the imported mask; that belongs to the create, which
is a write, and to completion, which verifies what the create pinned.

`POST .../draft` **creates and only creates**, requiring `If-None-Match: *` (`428` without it) and taking
the document as its body. `412` if a draft already exists. Create-only rather than an upsert is a
correctness choice, not tidiness: an upsert hands a second window holding the same seed the *first*
window's saved draft together with a currently-valid token for a document it never read, and its next save
then overwrites work no precondition can protect.

`PUT .../draft` requires the ETag in `If-Match`. Missing preconditions receive `428`; a stale one receives
`412`. The successful save increments the version and returns a new ETag. There is one writer in normal use,
but this contract also handles a second window and an autosave racing a manual save without silent loss.

`DELETE .../draft` discards without completing, `If-Match` guarded the same way. `If-Match: *` is the
deliberate force, offered by the editor only after a `412` has told the user the draft moved under them, so
discarding another window's work is always a second, informed click. Every route above has a
`/api/samples/{id}/…` twin with its own ETag namespace.

Migration 016 deleted the drafts the previous design left behind, when opening the editor created one and
completing one recreated it. Its `version = 1` predicate meant "never saved" only under *that* write path
and is never valid again — which is why `count_open_image_drafts` counts rows with no predicate at all.

There is deliberately no bulk discard on the queue screen. Now that a draft means work, "discard the
eleven drafts blocking this scope change" would destroy eleven pieces of real annotation work behind one
click, with nothing on screen saying what was in them. Discarding stays per draft, in the editor, beside
the document it throws away.

What the queue does instead is **name them**. `AnnotationScopeState` carries `open_draft_units` beside
the count — sample key, channel and the image id to open the editor at, capped at 24 with the count
telling the whole truth. A blocker that says "2 images hold annotation work; finish or discard it first"
over a dataset of several hundred is accurate and unusable; each unit is now a link into the editor,
where Complete and Discard already live. A sample draft is named through its sample's first image,
because the editor is addressed by the pair in either scope. The prose says *not completed* rather than
*unsaved* for the same reason: a draft row exists precisely because somebody saved one, and telling them
otherwise sent them looking for an editor they had already saved.

## Copying regions between channels

`POST /api/images/{image_id}/annotations/copy-regions` takes `{target_image_ids}` and **appends** the
source draft's shapes to each target, with fresh ids. Image scope only: under sample scope one document
already covers every channel, so the route answers `409` pointing at the sample routes.

Appending is what lets the targets go unguarded — an operation that only adds cannot lose a target draft
another window saved. The *source* carries `If-Match`, because a copy is one action that writes to several
channels and an editor that has fallen behind must not spread a stale document across all of them. Targets
that are not images of the same sample, or that do not share its dimensions, are refused (`409`) rather
than rescaled: an annotation is in source-image pixels and never leaves that frame, so a rescale would
invent geometry nobody drew. One bad target fails the whole request, so a person is never left working out
which channels received the copy.

Ids are minted rather than carried. They are unique within one document, and each target already has its
own; copying them verbatim would turn the second copy into a duplicate-id `422` half way through a
fan-out. Each copy is then an ordinary, independently editable region — which is the point, because the
exposures are milliseconds apart and a copy usually needs a nudge.

## Completion and storage

`POST .../complete` also requires `If-Match`. It renders the base and ordered polygon/bitmap operations to a binary
PNG, atomically writes `data/annotations/image-<id>/revision-<n>.png`, hashes the canonical document and mask,
inserts an append-only revision, then removes the draft. A database trigger rejects `UPDATE` on revisions.
The mask endpoint verifies both its expected app-owned path and digest before serving immutable bytes.

`POST /api/samples/{id}/annotations/complete` does the same thing once and copies the rendered bytes to
every image's own revision path, returning the list. The `mask_sha256` is therefore identical across the
fan-out, which is what makes "these channels carry the same truth" checkable rather than merely intended;
each image keeps its own `revision_no`, so one with earlier image-scoped history continues counting from
where it stopped. Any failure removes every file it had written.

Dataset deletion includes annotation directories and rows in its previewed app-owned cascade. The imported
image and mask trees are outside that inventory and survive unchanged.

## Interchange

The current completed truth—not a mutable draft—exports losslessly in three forms:

- binary PNG is a source-sized 0/255 mask;
- LabelMe 7 JSON uses its native `shape_type: "mask"`: a two-point inclusive bounding box plus a cropped,
  base64-encoded PNG. Empty truth has no shapes;
- COCO JSON is a one-image dataset whose binary defect region is an uncompressed, column-major RLE
  annotation. Empty truth has no annotations.

Draft import is `ETag` / `If-Match` guarded like every other edit. PNG and native raster shapes become bitmap
layers; LabelMe and COCO polygons remain editable polygon layers. COCO compressed and uncompressed RLE are
both accepted. Labels resolve by stable taxonomy key or display name, and unknown classes or mismatched image
dimensions are refused. Import replaces the editable operations while preserving immutable source-mask
provenance: a source-backed document first subtracts the entire source base and then applies the imported
layers. Completing the result therefore creates another app-owned revision and never rewrites the source.

## Current boundary

The annotation API, taxonomy, renderer, revision store, evaluation integration and editor foundation are live. One resolver is
used by pixel metrics, image overlays and `has_mask` reads: the newest completed `AnnotationRevision` wins,
then the imported `Mask`, then no mask. Evaluation verifies pinned bytes before reading them and stores a
content digest of the subset's sample labels and resolved mask identities in each `MetricSet`. A later label
or revision change therefore makes the old metrics visibly stale; reevaluation refreshes them from persisted
scores without running the model again. Legacy metric rows have no digest and are intentionally stale.

The queue filters to one label and to samples still missing ground truth, and marks each card with
whether every image of that sample resolves to truth, some do, or none — resolved by the same SQL
predicate the filter uses, so the queue can never disagree with what evaluation will read. Under sample
scope a part is one card and one job however many times it was photographed.

The editor's canvas column carries a channel strip whenever a sample has more than one image. Under
sample scope switching channel is a pure display change — the document belongs to the part, so the
shapes stay on screen and visibly land on the new illumination; under image scope each channel owns its
own truth, so it is real navigation, and it saves first rather than refusing. Three view modes: one
channel, two side by side sharing a single controlled view, or a blend that composites a second channel
at adjustable alpha. The blend is the registration check — a few pixels of drift are invisible side by
side.

**A pane draws the document that is truth for its own image, never a neighbour's.** Under sample scope
that is the edited document by construction; under image scope the reference pane draws the reference
channel's **own draft**, read from the same prefetched cache the copy dialog counts. Drawing the active
channel's regions there would claim truth that does not exist; drawing nothing, which it did at first,
hid truth that does — an already-annotated channel looked untouched, and a pane that showed no mask and
took no strokes read as broken.

**Editing happens in one pane, always the left one**, so a stroke never has an ambiguous destination.
Wanting to draw on the right is answered by making it the left: `Edit this channel` exchanges the two,
writing the outgoing channel into the reference preference so a part with three channels swaps rather
than rotating. The reference pane is otherwise inert — its regions do not answer the pointer, because a
drag affordance that snaps back on release is an affordance that lies.

The second pane is chosen by channel *position*, not by image id: the preference has to outlive the
image it was expressed on, so that a reader who put `dark` beside `bright` still has a second pane on the
next part. Choosing the channel that is already active wraps to the next one, which is what stops a
two-channel sample from showing the same photograph twice. Under image scope the editor reads every
channel's draft ahead of being asked, so switching channel resolves from the cache, the reference pane is
populated before it is looked at and the copy dialog can say what each channel already holds — affordable
only because reading a draft no longer writes one.

The channel strip's tabs are the one thing in it that gives way when the row is over-subscribed: they
read perfectly well half-scrolled, and everything to their right is a control with a usable minimum
size. A slider narrower than its own thumb is a rendering fault, not a tight fit.

The dataset-local queue opens a full-height controlled Konva scene for polygon/vertex and brush/eraser
editing, add/subtract, gesture-based pan/zoom, undo/redo and `ETag`-guarded save/completion. There is no
separate pan mode: left-drag moves the scene while Select is active and right-drag moves it from every
tool. Fit and source-pixel 1:1 are explicit views; a Select-mode double-click toggles Fit and the previous
view. A brush gesture is cropped into a bitmap layer in source coordinates, and that layer can be traced
deterministically into simplified, editable outer and hole polygons; this raster-to-vector operation does
not claim the image-aware boundary refinement reserved for MobileSAM.
Dirty drafts autosave after a short idle period; `412` keeps the local edit visible and offers an explicit
server-draft reload rather than choosing a winner. Keyboard traversal prefetches adjacent queue pages so
their boundary is not a dead end.

## Direct manipulation

Four rules, and each replaced something that only looked like it worked.

**A stroke extends the selected region.** A brush or eraser gesture composites into the selected bitmap
region and re-crops it to what is actually painted, so a defect is drawn in as many touches as it takes and
stays one region. With nothing selected the brush starts a region and selects it. Both PNG shapes the
raster contract allows are normalised on the way in by compositing over an opaque black ground, and the
output is the opaque black-and-white the backend itself writes. A region erased to nothing is removed.

**A brush size is a diameter, and one means one pixel.** `rasterizeStroke` walks the pointer samples into
an 8-connected integer spine and stamps a disc of the given diameter on every pixel of it, one row span at
a time. Canvas2D path stroking did this until it could not: fractional coordinates and an antialiased edge,
thresholded afterwards, gave *one* setting *three* footprints — a new region kept every touched pixel, a
stroke continuing an existing region needed a quarter coverage, and the eraser needed three quarters. So
brush and eraser at the same setting did not undo each other, the smallest possible mark was a blob about
three pixels across, and the control would not go below a radius of 2 — four pixels — while calling itself
"Brush size" in `px`. The rasteriser is now shared by all three paths, so the eraser removes exactly what
the brush at that size would add, and the spine is what closes the gaps between `mousemove` samples that
`lineTo` used to close. The in-progress preview draws at true source size with a one-*screen*-pixel floor,
because a one-pixel stroke at fit zoom is otherwise invisible while it is being made.

**The eraser never creates.** It takes paint off the selected region, or — with nothing selected — off
every painted region the stroke passes over, in one commit so the gesture is one undo step. It briefly
appended a `subtract` layer instead, on the reasoning that cutting a hole through a *polygon* is the one
thing erasing pixels cannot do; but a `subtract` layer is a **region**, so the tool for removing things
added one, named it in the region list, and left an operator holding a document with more shapes than
before. Cutting a polygon is still available — as an explicit Subtract region in the New region panel,
where creating something is what the control says it does. A stroke over nothing painted says so and
changes nothing.

**A region moves.** Dragging with Select translates it and an arrow key nudges it — 1 px, or 10 with
Shift. The offset is clamped once against the shape's own extent, never per coordinate, because clamping
each vertex on its own deforms a polygon pushed against an edge. This is what a copied region needs: the
exposures of one part are milliseconds apart on a moving line, so a copy lands close and not right.

**A polygon closes itself.** A click within a screen-sized radius of the first vertex closes the ring; a
click on the *last* vertex is dropped as a duplicate, which is what lets a double-click anywhere add its
vertex and then close. The first vertex swells while the pointer is over it. Backspace removes the last
vertex, Escape discards the ring, and the tool stays active afterwards because most parts carry more than
one defect. There is no "Close" button.

**A vertex drag is live.** The dragged point is held as transient scene state and applied to the outline
during the gesture, then committed once on release. The scene still never becomes a second store of
annotation truth — it is the same kind of state as a brush stroke in progress — but the polygon now follows
the handle instead of jumping to it when the mouse comes up.

---

[← the handbook](README.md) · [domain model](domain-model.md) · [evaluation](evaluation.md)
