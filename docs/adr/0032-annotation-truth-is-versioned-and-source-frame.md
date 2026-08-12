# ADR-0032 — Annotation truth is versioned and source-frame

**Status:** Accepted (2026-08-12)

## Context

Imported benchmark masks are both data and evidence: changing one in place destroys the ability to
reproduce the benchmark as received. They are also not a workable editing format. A bitmap carries the
evaluation answer but not the vertices, labels, additions and subtractions a person needs to revise it.
The editor must autosave without silently overwriting a newer edit, and an experiment must be able to say
which ground truth its metrics used after the annotation changes.

The serious alternatives were to overwrite imported masks, keep only editable bitmaps, store only vector
documents and rasterise them whenever a consumer asks, or treat every edit as an event. Overwriting source
data breaks provenance. Bitmaps discard edit structure. On-demand rasterisation lets renderer changes alter
old evaluation truth. An event log preserves more history than this single-user workbench needs while making
the current state and interchange substantially harder to reason about.

## Decision

**Source masks are immutable provenance; application truth is a source-frame document with immutable,
materialised revisions.**

- Each dataset owns a taxonomy whose stable key is stored in shapes; display name, colour and order may
  change without rewriting annotations.
- Each image has at most one mutable draft. The document uses source-image pixel coordinates and pins its
  dimensions and base layer. `ETag` / `If-Match` is the write contract: a stale save receives `412` rather
  than winning by arrival order.
- Completing a draft appends an `AnnotationRevision`. Its canonical document and binary PNG mask are both
  SHA256-addressed. A SQLite trigger rejects revision updates; dataset deletion may still delete them.
- A draft may start from an imported source mask. Its id, path and digest are copied into provenance, the
  source bytes are verified before rendering, and the derived PNG is written under the app data directory.
  No annotation operation writes into the dataset tree.
- Evaluation resolves one ground-truth snapshot: the newest completed revision for each image when one
  exists, otherwise its imported source mask. A digest of that resolved set is stored with metrics, so a
  newer completed revision makes old metrics visibly stale rather than silently changing their meaning.
- Polygon operations are the first document primitive. Brush strokes, bitmap/RLE interchange and model
  proposals extend the versioned document; they do not change its coordinate frame or revision lifecycle.

## Consequences

- The editor can autosave aggressively and still surface a real conflict. This is slightly more API work
  than last-write-wins and deliberately does not attempt collaborative merging.
- Every completion costs a full-resolution PNG even when a compact polygon describes the same region. The
  duplicate is intentional: it freezes the exact binary truth used by evaluation and makes consumers
  independent of editor/rendering code.
- Taxonomy keys cannot be renamed casually. A rename is a migration across drafts and revisions, not a
  cosmetic update; the API therefore only edits label presentation.
- Source mask drift becomes detectable when a mask first participates in an editable draft. Old catalog
  rows have a nullable digest because migration 005 cannot truthfully invent hashes for files it has not
  read.
- A source mask and a completed revision can disagree. That is the point of preserving both; interfaces
  must label provenance clearly and never present the derived mask as the imported original.
- The document schema needs explicit versioning. Supporting a new shape without a reader for old documents
  would make historical revisions unreadable even though their materialised masks remain valid.
