# Annotation truth

Imported masks and editable annotations are deliberately different things (ADR-0032). A `Mask` points at
the source dataset and is never rewritten. An `AnnotationDraft` and its completed `AnnotationRevision`s live
under application ownership. This is what lets a benchmark remain reproducible while a person improves its
truth.

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

`POST /api/images/{id}/annotations/draft` is idempotent. It returns the existing draft, copies the newest
completed document, or opens a new document on the imported mask / an empty canvas. Opening on a source mask
hashes and pins that file without changing it. Every draft response carries an ETag of the image id and
monotonic version.

`PUT .../draft` requires the ETag in `If-Match`. Missing preconditions receive `428`; a stale one receives
`412`. The successful save increments the version and returns a new ETag. There is one writer in normal use,
but this contract also handles a second window and an autosave racing a manual save without silent loss.

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
own truth, so it is real navigation and takes the dirty guard. Three view modes: one channel, two side
by side sharing a single controlled view, or a blend that composites a second channel at adjustable
alpha. The blend is the registration check — a few pixels of drift are invisible side by side. Shapes
are drawn on a pane only when the document is truth for that pane's image, so an image-scoped reference
pane shows the bare photograph.

The dataset-local queue opens a full-height controlled Konva scene for polygon/vertex and brush/eraser
editing, add/subtract, gesture-based pan/zoom, undo/redo and `ETag`-guarded save/completion. There is no
separate pan mode: left-drag moves the scene while Select is active and right-drag moves it from every
tool. Fit and source-pixel 1:1 are explicit views; a Select-mode double-click toggles Fit and the previous
view. A finished brush gesture is cropped into a bitmap layer in source coordinates. It can be traced
deterministically into simplified, editable outer and hole polygons; this raster-to-vector operation does
not claim the image-aware boundary refinement reserved for MobileSAM. Dirty drafts autosave after a short
idle period; `412` keeps the local edit visible and offers an explicit server-draft reload rather than
choosing a winner. Keyboard traversal prefetches adjacent queue pages so their boundary is not a dead end.

---

[← the handbook](README.md) · [domain model](domain-model.md) · [evaluation](evaluation.md)
