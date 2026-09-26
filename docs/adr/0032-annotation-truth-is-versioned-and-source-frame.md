# ADR-0032: Annotation truth is versioned and source-frame

**Status:** Accepted (2026-08-12)

## Context

Imported benchmark masks are both data and evidence: changing one in place destroys the ability to
reproduce the benchmark as received. They are also not a workable editing format. A bitmap carries
the evaluation answer but not the vertices, labels, additions and subtractions a person needs to
revise it. The editor must autosave without silently overwriting a newer edit, and an experiment
must be able to say which ground truth its metrics used after the annotation changes.

## Decision

**Source masks are immutable provenance; application truth is a source-frame document with
immutable, materialised revisions.**

- Each dataset owns a taxonomy whose stable key is stored in shapes; display name, colour and order
  may change without rewriting annotations.
- Each image has at most one mutable draft, **created by its first save**, never by opening the
  editor, so an open draft always means unfinished work. The document uses source-image pixel
  coordinates and pins its dimensions and base layer.
- **Writes are conditional.** `ETag` / `If-Match` guards every save and discard, and creation is
  create-only (`If-None-Match: *`): a stale write receives `412` rather than winning by arrival
  order.
- Completing a draft appends an `AnnotationRevision`. Its canonical document and binary PNG mask are
  both SHA256-addressed, and the database rejects updates to a revision.
- A draft may start from an imported source mask. Its provenance and digest are copied, the source
  bytes are verified before rendering, and derived files are app-owned. No annotation operation
  writes into the dataset tree.
- Evaluation resolves one ground-truth snapshot: the newest completed revision per image when one
  exists, otherwise its imported source mask. A digest of that set is stored with metrics, so a newer
  revision makes old metrics visibly stale rather than silently changing their meaning.
- New shape kinds and proposals extend the versioned document; they never change its coordinate
  frame or revision lifecycle.

## Alternatives considered

- **Overwrite imported masks.** Simple, and it breaks provenance: the benchmark as received can no
  longer be reproduced.
- **Keep only editable bitmaps.** A bitmap discards the edit structure a person needs to revise it.
- **Store only vector documents and rasterise on demand.** A renderer change would then alter old
  evaluation truth.
- **An event log of edits.** It preserves more history than a single-user workbench needs, and makes
  current state and interchange much harder to reason about.
- **An upsert for draft creation.** It would hand a second window a valid token for a document it
  never read.

## Consequences

- The editor can autosave aggressively and still surface a real conflict. This is more API work
  than last-write-wins and deliberately does not attempt collaborative merging.
- Every completion costs a full-resolution PNG even when a compact polygon describes the same
  region. The duplicate freezes the exact binary truth evaluation used, independent of editor code.
- Taxonomy keys cannot be renamed casually. A rename rewrites drafts and revisions; the API only
  edits label presentation.
- Source-mask drift is detected only when a mask first participates in a draft or an evaluation.
- A source mask and a completed revision can disagree. That is the point of keeping both; interfaces
  must label provenance and never present a derived mask as the imported original.
- The document schema needs explicit versioning: a new shape without a reader for old documents
  would make historical revisions unreadable even though their masks stay valid.
