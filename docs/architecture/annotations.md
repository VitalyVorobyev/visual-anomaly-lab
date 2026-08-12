# Annotation truth

Imported masks and editable annotations are deliberately different things (ADR-0032). A `Mask` points at
the source dataset and is never rewritten. An `AnnotationDraft` and its completed `AnnotationRevision`s live
under application ownership. This is what lets a benchmark remain reproducible while a person improves its
truth.

## Coordinate and document contract

An annotation document is JSON schema version 1 in **source-image pixel coordinates**. It pins
`image_width`, `image_height`, a base (`empty` or `source_mask`), and an ordered list of shapes. The first
shape is a polygon with a stable id, dataset taxonomy key, `add` / `subtract` operation and three or more
points. Duplicate shape ids, points outside the source frame, unknown label keys, dimension changes and
base-layer changes are rejected before persistence.

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

`POST .../complete` also requires `If-Match`. It renders the base and ordered polygon operations to a binary
PNG, atomically writes `data/annotations/image-<id>/revision-<n>.png`, hashes the canonical document and mask,
inserts an append-only revision, then removes the draft. A database trigger rejects `UPDATE` on revisions.
The mask endpoint verifies both its expected app-owned path and digest before serving immutable bytes.

Dataset deletion includes annotation directories and rows in its previewed app-owned cascade. The imported
image and mask trees are outside that inventory and survive unchanged.

## Current boundary

The annotation API, taxonomy, renderer, revision store and evaluation integration are live. One resolver is
used by pixel metrics, image overlays and `has_mask` reads: the newest completed `AnnotationRevision` wins,
then the imported `Mask`, then no mask. Evaluation verifies pinned bytes before reading them and stores a
content digest of the subset's sample labels and resolved mask identities in each `MetricSet`. A later label
or revision change therefore makes the old metrics visibly stale; reevaluation refreshes them from persisted
scores without running the model again. Legacy metric rows have no digest and are intentionally stale.

PNG / LabelMe / COCO interchange, brush layers and the full-height editor extend this contract rather than
introducing another annotation store.

---

[← the handbook](README.md) · [domain model](domain-model.md) · [evaluation](evaluation.md)
