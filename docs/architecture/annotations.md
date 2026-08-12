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

The annotation API, taxonomy, renderer, revision store and evaluation integration are live. One resolver is
used by pixel metrics, image overlays and `has_mask` reads: the newest completed `AnnotationRevision` wins,
then the imported `Mask`, then no mask. Evaluation verifies pinned bytes before reading them and stores a
content digest of the subset's sample labels and resolved mask identities in each `MetricSet`. A later label
or revision change therefore makes the old metrics visibly stale; reevaluation refreshes them from persisted
scores without running the model again. Legacy metric rows have no digest and are intentionally stale.

Brush layers and the full-height editor extend this contract rather than introducing another annotation
store.

---

[← the handbook](README.md) · [domain model](domain-model.md) · [evaluation](evaluation.md)
