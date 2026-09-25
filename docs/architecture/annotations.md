# Annotations

Imported masks and editable annotations are different things (ADR-0032). A `Mask` points at the source
dataset and is never rewritten. An `AnnotationDraft` and its completed `AnnotationRevision`s are
app-owned. A benchmark therefore stays reproducible while a person improves its truth.

## Raster contract

**A bitmap shape's mask channel is luminance.** The backend defines it — `decode_png` is
`convert("L") > 0`, `encode_png` writes mode `L` on an opaque black ground — and every producer follows:
a brush stroke, an accepted MobileSAM candidate, an imported PNG / LabelMe / COCO mask. The frontend
converts luminance to alpha only when painting, so one contract serves both an editor overlay and an
opaque stored mask.

Regions are painted in their label's colour; cuts (`operation: "subtract"`) in one fixed colour with a
dashed outline. The scene resolves its colours from the design tokens at runtime — Konva cannot take a
class name, so `scenePalette.ts` reads `styles.css` and repaints on a theme change (ADR-0021). Mask
weight is a persisted per-reader preference; the label colour is dataset taxonomy.

The Annotate tab's **Classes** section (`routes/dataset/ClassManager.tsx`) adds a class — its key
derived once from the name, unique within the dataset, never changed — and renames, recolours and
reorders one through `PUT .../annotation-labels/{key}`. There is no delete: regions may still name the
class. The seeded `defect` class is magenta (`#c026d3`): legible over metal, plastic and dark field, not
read as an error state, and distinct from the teal `signal` selection outline.

**A class's truth includes its absence** (ADR-0040). `annotations_repo.presence_by_class` answers, per
sample, whether a class is present, absent or unlabelled:

- An image's newest completed revision decides it, for the classes that existed when it was completed. A
  class created later is unlabelled there: it was never in front of the annotator.
- A revision's pinned class table answers exactly: present when the class has pixels, absent when it has
  none. A revision completed before class tables existed answers from its document instead: present when
  it adds a region of the class (or, for `defect`, starts from the source mask).
- Without a revision, only `defect` has an answer: an imported ground-truth mask is present, and a sample
  labelled normal is absent. Every other class is unlabelled.
- A sample shows the class when any of its images does, and is absent only when all of them are.

**Every class at once**, for supervised segmentation (ADR-0039): an image is labelled for a run when the
rule above answers *every* class the run pinned, and `class_truth.load_label_map` then reads it as one
8-bit map in the run's numbering — `classes[i]` is `i + 1`, background 0. A revision's class mask is
renumbered from its pinned table; a pixel drawn in a class the run did not pin is `IGNORE_INDEX` (255).
An imported mask or a normal label answers for `defect` alone, so it labels an image only for a run whose
classes are `defect` alone.

### Detection truth

**Every class at once, as boxes**, for object detection (ADR-0039): the images the rule above labels for
a run, read by `class_truth.load_boxes` as the image's object instances in the source frame. Which
source answers is `resolve_box_truth`'s:

- a revision's instances file, when it has one and its document starts from an empty base;
- its document, re-rasterised in memory (`annotation_render.document_instances`), when it has no
  instances file or starts from the source mask. The instances completion would write come first, then
  one instance per 8-connected component of the `source_mask` base that no drawn instance owns, keyed
  `source-mask-<n>`. A document read this way is rasterised by today's shape rules, not those its
  revision was completed under ([what a revision wrote, it keeps](#completion-and-storage));
- an imported mask's 8-connected components, each one instance of `defect`;
- nothing, for an image of a sample labelled normal: a confirmed absence.

A component stands for an object because a binary mask records no instances: two touching defects are
one box and one broken defect is two. An instance's box is the tight box of the pixels it owns, so a drawn box with
integer corners is its own instance box. A reader keeps
the classes its run pinned; `load_boxes` returns every instance and verifies each source against its
digest.

**Boxes a dataset ships are a revision, not a source.** A pack annotated as boxes of several classes
(PKU-Market-PCB's Pascal VOC files) enters each image's boxes as its first completed revision, one `box`
shape per object on an empty base (`annotations/imported_boxes.py`, [import](import.md#box-truth-a-pack-ships)).
It is read by the first rule above, answers every class the dataset had when it was completed, and opens in
the editor like any drawn truth; a correction completes the next revision.

`GET /api/datasets/{id}/annotation-labels/coverage` counts those samples per class, and
`GET /api/datasets/{id}/samples?class_key=&presence=` lists them. `GET /api/images/{id}/mask?class_key=`
outlines one class's region from its own truth (`annotations/class_truth.py`), in the source frame; an
image whose truth does not answer for the class is a 404, and a confirmed absence is an empty outline.

**One class's region, set from outside the editor.** `service.class_region_document` rebuilds an image's
current truth with one class replaced, for the reference studio's accept, fix and mark-absent. It appends
rather than rewrites: a bitmap `subtract` over the class's old pixels (which hold that class alone), then a
bitmap `add` of the new region wherever no other class is. Every other class's shapes stay exactly as they
were, so the result is still a document the editor opens. It goes through the ordinary draft lifecycle —
create, then complete unless the reader will fix it — and is refused while the image has an open draft,
because that is someone's unfinished work. It is what a few-shot
split draws references from, and what a run on the class can test on.

## Document contract

An annotation document is JSON schema version 1 in **source-image pixel coordinates**. It pins
`image_width`, `image_height`, a base (`empty` or `source_mask`) and an ordered list of shapes:

- a polygon — stable id, taxonomy key, `add` / `subtract` operation, three or more points, in the
  same pixel-edge coordinates as a box. It owns the pixels whose centres it contains by the **even-odd
  rule** (`annotation_render.polygon_coverage`): a centre on a left or top edge is in, one on a right or
  bottom edge is out, a ring traced inside another is a hole, a self-intersection's doubly wound region
  is outside, and a polygon that contains no centre — sub-pixel, or with no area — owns nothing. It is
  a scanline at pixel centres in numpy, one crossing per edge and row, bounded by the polygon's own
  extent, and it is the editor's pixel readout (`pixelReadout.insidePolygon`) evaluated everywhere at
  once; `polygonCentres.fixture.json` pins the two against each other. The editor's canvas and the
  gallery's vector layer fill a polygon even-odd too;
- a box — the same identity, taxonomy and operation fields and a float `x`, `y`, `width`, `height`
  rectangle with positive area, in **pixel-edge coordinates**: pixel `i` spans `[i, i + 1)`, as it does
  on the editor's canvas, where a drag's corners land wherever the pointer is, and in
  `SpatialTransform.prepare_box`. A box owns the pixels whose centres it covers, half-open
  (`x <= i + 0.5 < x + width`, `BoxShape.owned_pixels`), so a box at `x = 1` of width 3 owns columns 1
  to 3 and its instance box runs from `x0 = 1` to `x1 = 4` — exactly what was drawn. It is the polygon
  rule on a rectangle, so a box and the polygon of its four corners own the same pixels;
- a bitmap — the same identity, taxonomy and operation fields, a cropped binary PNG and its integer
  source-frame rectangle.

Every shape may carry an optional `instance_id`, which groups `add` shapes into one object instance
(see [completion](#completion-and-storage)). Unset, a shape is its own instance. Add shapes that share
an instance must share a class.

Duplicate shape ids, geometry outside the source frame, malformed bitmap bytes, unknown label keys,
an instance holding two classes, dimension changes and base-layer changes are rejected before
persistence.

**The shape list grows without a new schema version.** `schema_version` stays `1`: box and
`instance_id` are additions to a discriminated union, so every stored document reads unchanged. An unset
`instance_id` is left out of the canonical JSON rather than written as null, so every stored document
keeps its `document_sha256`; `tests/test_annotation_boxes.py` pins one such digest. A change that would
alter how an existing document reads or hashes is what would need a version.

The editor draws a `source_mask` base from `GET /api/images/{id}/annotations/source-mask` — the pinned
import, binary, digest-checked — and **not** from `GET /api/images/{id}/mask`, which is the image's
*current* truth and would show a revision's regions underneath themselves.

**The source frame is load-bearing.** A region profile changes what a method sees, but an annotation
never moves into model-input or canvas coordinates; the UI transform and the spatial pipeline map back to
this frame.

## Scope

A dataset annotates either each image or each whole sample; `Dataset.annotation_scope` says which
(ADR-0036). `image` is the default. `sample` is for a multi-shot rig whose channels are exposures of one
registered part: one document is edited once and materialised as **one ordinary `AnnotationRevision` per
image of the sample**. Below that boundary truth is image-keyed — `resolve_ground_truth_masks`, pixel
metrics, `has_mask`, the `MetricSet` digest and all three interchange formats never learn that scope
exists.

`GET`/`PUT /api/datasets/{id}/annotation-scope` reads and moves it. The read reports **every** reason
sample scope is unavailable: imported source masks (pinned per image), samples whose images differ in
dimensions (a document pins one frame), and open drafts. Leaving sample scope is refused while a sample
draft is open; completed revisions are untouched either way. Under `sample` scope the per-image write
routes return `409` naming the sample route, so two scopes never hold valid ETags for different documents
of one part.

A sample-scoped document is always `base="empty"` (enforced), and its draft lives in
`annotation_sample_draft` with its own ETag namespace (`annotation-sample-draft-{sample}-v{n}`).

## Draft lifecycle

**A draft records work, not a page view.** It is created by the first save and nothing else, so "how many
drafts are open" equals "how much unfinished work is there" — the question `annotation_scope` asks.
`count_open_image_drafts` counts rows with no predicate.

| route | precondition | behaviour |
| --- | --- | --- |
| `GET .../draft` | — | read-or-seed, never writes. Returns the draft with its ETag, or `persisted: false`, null `version`, no ETag and the seed: newest completed revision, else imported mask, else empty |
| `POST .../draft` | `If-None-Match: *` (`428` without) | creates only; `412` if a draft exists |
| `PUT .../draft` | `If-Match` (`428` missing, `412` stale) | saves, increments version, returns a new ETag |
| `DELETE .../draft` | `If-Match`; `*` forces | discards without completing |
| `POST .../complete` | `If-Match` | renders and stores a revision, removes the draft |

Every route has a `/api/samples/{id}/…` twin. One helper produces the seed for both read and create, so
they cannot disagree; the read does not hash the imported mask — the create pins it and completion
verifies it. **Create-only, not upsert**, because an upsert would hand a second window a valid token for a
document it never read. The editor offers `If-Match: *` only after a `412` has told the user the draft
moved, so discarding another window's work is always a second, informed click.

**The twins are one implementation.** `annotations/service.py` writes the lifecycle once on `DraftUnit`
— read-or-seed, create-only, save-at-version, discard, and the locked prelude (resolve the dataset,
refuse the other scope, find the draft or `404`, match the ETag or `412`). `ImageDrafts` and
`SampleDrafts` supply only how the unit is found, its seed, its frame and its table. Completion stays per
unit. The routes in `api/routers/annotations.py` read precondition headers, call the service and set
`ETag`; every refusal below them is a domain error ([overview](README.md)).

There is no bulk discard: a draft is real work and discarding stays per draft, in the editor.
`AnnotationScopeState` instead **names** open drafts in `open_draft_units` — sample key, channel and the
image id to open the editor at, capped at 24 with the count telling the whole truth. A sample draft is
named through its sample's first image. The prose says *not completed*, never *unsaved*.

## Copying regions between channels

`POST /api/images/{image_id}/annotations/copy-regions` takes `{target_image_ids}` and **appends** the
source draft's shapes to each target with freshly minted ids (ids are unique per document). Image scope
only; under sample scope it answers `409` pointing at the sample routes.

The source carries `If-Match` so a stale editor cannot spread its document; targets need no guard
because an append cannot lose their work. Targets that are not images of the same sample or differ in
dimensions are refused (`409`), never rescaled, and one bad target fails the whole request.

## Completion and storage

Completion rasterises the base and the ordered polygon, box and bitmap operations once
(`annotation_render.py`). An `add` paints its class's index and a `subtract` clears whatever is there; a
`source_mask` base is drawn in `defect`. Three files are written atomically from that one pass:

- `data/annotations/image-<id>/revision-<n>.png`, the binary mask every anomaly consumer reads, which is
  exactly `index > 0`;
- `revision-<n>.classes.png`, the 8-bit class-index mask, with 0 as background;
- `revision-<n>.instances.json`, the object instances detection and instance segmentation read:
  `{"instances": [{instance_id, label_key, box, pixels}]}` ([detection truth](#detection-truth)).

An instance is keyed by `instance_id`, else by the shape's id, over `add` shapes in draw order. It owns
the pixels its shapes set that still carry its class in the final class mask: a later cut or a later
shape drawn over them takes them away. `box` is the tight `[x0, y0, x1, y1]` of those pixels, `x1`/`y1`
exclusive, and `pixels` their count. An instance left with no pixels is dropped, and the `source_mask`
base belongs to no instance. The pass keeps one `int32` owner raster beside the class canvas, painted
from the same coverage, so memory is bounded by the image, not by the number of shapes.

**What a revision wrote, it keeps.** Completed revisions are immutable, and their files verify against
their own digests, so a revision completed while a polygon or a box was filled outline-inclusive (one
pixel wider on its far edges) is read that way wherever a file answers: the binary mask for every
anomaly consumer and for export, the class mask for class and label truth, the instances file for
detection. What reads a document **in memory** follows the rule in [the document
contract](#document-contract) instead: the class and label truth of a revision older than class tables, the detection truth of one older than
instances files or drawn over the source mask, the reference studio's current region, and interchange's
`render_shapes`. Such an answer names the polygon rule in its identity when its document holds a
polygon, so a run scored against it reads its ground truth as stale rather than silently changed.

The revision pins `class_table`: every class the dataset had at completion, in taxonomy order, with its
index (from 1) and pixel count, and `instances_path` / `instances_sha256` (migration 024; null on a
revision completed before it). Completion then hashes the canonical document, both masks and the instances
file, inserts an append-only revision and removes the draft. A database trigger rejects `UPDATE` on revisions. The mask
endpoint verifies its expected app-owned path and digest before serving immutable bytes.

`POST /api/samples/{id}/annotations/complete` renders once and copies all three files to every image's own
revision path, so `mask_sha256`, `class_mask_sha256` and `instances_sha256` are identical across the fan-out and shared truth is checkable. Each image
keeps its own `revision_no`. Every written file is registered on the write transaction and removed if it
rolls back ([repository](repository.md)).

Dataset deletion includes annotation directories and rows in its previewed app-owned cascade; the imported
image and mask trees survive.

## Interchange

The current completed truth — never a draft — exports losslessly as:

- binary PNG — a source-sized 0/255 mask;
- LabelMe 7 JSON — `shape_type: "mask"`, a two-point inclusive bounding box plus a cropped base64 PNG;
  empty truth has no shapes;
- COCO JSON — a one-image dataset with an uncompressed, column-major RLE annotation; empty truth has no
  annotations.

Draft import is `If-Match` guarded. PNG and native raster shapes become bitmap layers; LabelMe and COCO
polygons stay editable polygons; COCO compressed and uncompressed RLE are both accepted. Labels resolve by
taxonomy key or display name; unknown classes or mismatched dimensions are refused. A source-backed
document first subtracts the whole source base, then applies the imported layers, so completing it creates
another app-owned revision and never rewrites the source.

## Resolution and evaluation

One resolver serves pixel metrics, overlays and `has_mask`: the newest completed `AnnotationRevision`,
else the imported `Mask`, else no mask. Evaluation verifies pinned bytes before reading them and stores a
digest of the subset's sample labels and resolved mask identities in each `MetricSet`, so a later label or
revision change makes the metrics visibly stale; reevaluation refreshes them from persisted scores
([evaluation](evaluation.md)). A row without a digest is stale.

The same resolver decides the localization question — did the map's peak land inside the drawn region? An
unannotated defect answers `null` (a dash) and is **excluded from numerator and denominator alike**; the
count is reported beside the verdicts. Under sample scope every channel resolves to the same truth, so a
part is judged by the image that *produced* its aggregate score, not by whichever channel landed.

## Editor

The dataset-local queue filters by label and to samples still missing ground truth, and marks each card
with whether every, some or none of the sample's images resolve to truth — by the same SQL predicate the
filter uses. Under sample scope a part is one card. Keyboard traversal prefetches adjacent queue pages.

The editor is a full-height controlled Konva scene: polygon/vertex, box and brush/eraser editing,
add/subtract, undo/redo, `ETag`-guarded save and completion. Dirty drafts autosave after a short idle; a `412` keeps the
local edit visible and offers an explicit reload of the server draft rather than choosing a winner.

**Navigation and view.** Left-drag pans while Select is active; right-drag pans from every tool. Fit and
source-pixel 1:1 are explicit views; a Select-mode double-click toggles Fit and the previous view. Zoom
tops out at 32 screen pixels per source pixel, and above 1:1 the image is drawn without smoothing. A
per-pane readout names the source pixel under the pointer, the mask value the document resolves to there
(base, then each shape in order) and the region on top — a readout, not truth; evaluation reads the
backend's renderer.

**Overlay visibility.** `H` toggles the drawn regions; holding `H` shows the other state while held. An eye
button beside Mask weight does the same. Hiding is separate from weight because outlines and handles carry
no opacity, and it does not persist, so an editor never opens with annotations invisible. The live brush
trail, pending polygon and assist points stay visible.

### Channels

A channel strip appears whenever a sample has more than one image. Under sample scope switching channel is
a display change — the shapes stay and land on the new illumination; under image scope it is navigation
and saves first. Three view modes: one channel, two side by side sharing one view, or a blend of a second
channel at adjustable alpha — the registration check.

- **A pane draws the truth of its own image.** Under image scope the reference pane draws the reference
  channel's own draft, from the same prefetched cache the copy dialog counts; the editor reads every
  channel's draft ahead of being asked.
- **Editing happens in the left pane only.** `Edit this channel` swaps the two panes, writing the outgoing
  channel into the reference preference. The reference pane's regions do not answer the pointer.
- The second pane is chosen by channel **position**, so the preference outlives the image; choosing the
  active channel wraps to the next.
- Both channels are chosen with the same strip control, the left pane's channel disabled with its reason
  rather than filtered out. It lives in the toolbar because Blend has no second pane. The first strip's tabs
  are what scrolls when the row is over-subscribed.

### Label

The editor header sets the sample's label through `PATCH /api/datasets/{id}/samples/{sid}` with the
sample viewer's `n` / `d` / `u` keys, beside the part's identity because **the label describes the part,
not the photograph** (ADR-0005). The edit is recorded `manual`, surviving re-import, and the header shows
`imported` or `hand-set`. `Sample.label` is part of the `ground_truth_digest`, so metric sets go stale.

Relabelling is not blocked by unsaved work (J/K are, because they navigate away): the mutation invalidates
the dataset's cache subtree, and drafts live under their own, so the document, its undo history and an
in-progress stroke survive. Where label and document contradict, a quiet mark names the consequence and
blocks nothing: a `defect` image resolving to no truth is skipped by pixel metrics, a `normal` sample's
mask is read as truth, an `unlabeled` sample is excluded. A document based on an imported mask counts as
carrying truth, so the mark never accuses somebody of leaving a drawn defect undrawn.

## Direct manipulation

- **A stroke extends the selected region.** Brush and eraser composite into the selected bitmap region and
  re-crop it to what is painted; with nothing selected the brush starts a region and selects it. Input PNGs
  are normalised over an opaque black ground; output is opaque black-and-white. A region erased to nothing
  is removed.
- **A brush size is a diameter, and one means one pixel.** `rasterizeStroke` walks pointer samples into an
  8-connected integer spine and stamps a disc of that diameter on every spine pixel. Brush, eraser and
  continuation share it, so the eraser removes exactly what the brush adds. The preview draws at true
  source size with a one-screen-pixel floor; the cursor is the stamped disc.
- **Strokes apply in order, each on the document the last one left.** Strokes are queued; each reads the
  current document and selection when it starts, and one whose document moved during its PNG decode is
  repainted rather than committed over the move.
- **The eraser never creates.** It takes paint off the selected region, or with nothing selected off every
  region the stroke crosses, in one undo step. Cutting a polygon is done by setting a region to Subtract in
  the Selection panel. A stroke over nothing painted says so and changes nothing.
- **A new region's class and operation are not chosen in advance.** Every new shape is selected, and
  operation is changed in Selection. The class picker appears only when the dataset has more than one class
  — label count is data. The inspector section shows the tool in hand (brush size, vertex readout) and is
  not drawn when empty.
- **A box is dragged corner to corner** with the box tool (`R`), normalised whichever corner the drag
  began from; a drag with no area draws nothing. Under Select a box moves like any region and its four
  corners are vertex handles: dragging one resizes the box against the opposite corner, and dragging past
  it flips the box rather than inverting it. `shapeOutline` is the one place a box becomes points, for
  the scene and the resize. The pixel readout tests a box half-open, and a polygon even-odd, at the
  pixel's centre — the rules completion owns their pixels by.
- **Class keys.** `2`–`9` pick the class for new regions, in the order the class picker lists them, which
  prints each class's key beside its name; `0` and `1` stay Fit and 1:1. They are one entry in
  `EDITOR_BINDINGS`, so the shortcut sheet lists them.
- **A region moves.** Select-drag translates; arrow keys nudge by 1 px, 10 with Shift. The offset is clamped
  once against the shape's extent, never per coordinate, so a polygon at an edge is not deformed.
- **A polygon closes itself.** A click near the first vertex closes the ring; a click on the last vertex is
  dropped as a duplicate, so a double-click adds and closes. Backspace removes the last vertex, Escape
  discards the ring, and the tool stays active. There is no Close button.
- **A vertex drag is live.** The dragged point is transient scene state applied during the gesture and
  committed once on release; the scene is never a second store of truth.
- **Tracing.** A bitmap layer can be traced deterministically into simplified editable outer and hole
  polygons; this is raster-to-vector only, not the image-aware refinement MobileSAM provides.

---

[← the handbook](README.md) · [domain model](domain-model.md) · [evaluation](evaluation.md)
