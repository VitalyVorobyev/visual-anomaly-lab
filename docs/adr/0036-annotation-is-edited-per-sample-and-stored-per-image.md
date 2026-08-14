# ADR-0036 — Annotation is edited per sample and stored per image

**Status:** Accepted (2026-08-14)

## Context

ADR-0005 makes `Sample` the unit of identity and `Channel` data, so one physical part photographed
under three illuminations is one sample owning three images. ADR-0032 then made annotation truth
versioned, source-frame and **image-keyed all the way down**: `annotation_draft.image_id` is a primary
key, `annotation_revision` is unique on `(image_id, revision_no)`, and the single resolver every
consumer reads — pixel metrics, image overlays, `has_mask`, the `MetricSet` ground-truth digest, and
all three interchange formats — is keyed by image.

Nothing connected the two. A defect that shows under dark field and is invisible under bright field
had to be traced once per illumination, by hand, on frames the import probe measures as registered to
within **0 px median offset** in every capture group. Three times the work to produce three copies of
one truth, with the copies free to drift apart.

Two shapes were live:

- **A sample-level annotation entity.** One `annotation_revision` per sample; the resolver learns
  that a sample's revision covers each of its images. Conceptually the cleaner model — truth about a
  part is stored once.
- **Sample-scoped *editing*, image-scoped *truth*.** One draft per sample; on completion, render once
  and append one ordinary `AnnotationRevision` per image.

The choice is not cosmetic. The first changes what "ground truth for image N" means for every reader
of it, and those readers are the evaluation layer.

## Decision

**`Dataset.annotation_scope` selects `image` (the default) or `sample`. Under `sample` scope, one
document is edited per sample and materialised as one revision per image of that sample.**

- **Truth stays image-keyed.** `resolve_ground_truth_masks`, pixel metrics, `has_mask`, the
  `MetricSet` digest and the PNG/LabelMe/COCO exports are **unchanged** and never learn that scope
  exists. Only the editing surface moves up.
- **One render, N files, one digest.** Completion renders the document once and copies those bytes to
  each image's own `revision-<n>.png`. The shared `mask_sha256` is what makes "these channels carry
  the same truth" checkable after the fact rather than merely intended. Each image keeps its own
  `revision_no`, so one carrying earlier image-scoped history continues counting from where it
  stopped.
- **A sample-scoped document is always `base="empty"`.** Source-mask provenance (`source_mask_id`,
  `_path`, `_sha256`) is pinned per image, and a document about to be written onto N images cannot
  carry image A's. This is why the draft is its own `annotation_sample_draft` table rather than a row
  stored against a nominated "carrier" image: those three columns would have to be permanently NULL
  and the ETag namespace would overlap a real image draft's.
- **The scope is a property of the data, not a preference.** A corpus whose channels are exposures of
  one registered part shares its truth; a corpus whose channels are unrelated views does not.
- **Entering `sample` scope is refused, with every reason at once**, when the dataset has imported
  source masks, when any sample mixes image dimensions (a shared document pins one frame), or when
  any draft is open. Leaving it is refused while any sample draft is open. Completed revisions are
  untouched in either direction — they are immutable and image-keyed, so they stay valid truth
  whichever scope produced them.
- **The per-image write routes return `409` under sample scope**, naming the sample route. Not a
  redirect: two writers editing one part through two scopes would each hold a valid ETag for a
  different document and neither would detect the other.
- **The queue collapses to one entry per sample.** A part is one job however many times it was
  photographed.

Rejected: the sample-level annotation entity. It buys storage of one PNG instead of three — ADR-0032
already accepted "every completion costs a full-resolution PNG" — and charges for it in the one place
the project cannot afford churn. Every consumer of ground truth would need a second resolution path,
and the `MetricSet` digest that makes stale metrics visible would have to hash a mixture of
sample-keyed and image-keyed identities. A method still receives images and still returns one
prediction per image (ADR-0007); truth that is not image-keyed would have to be joined back to them
at every use.

## Consequences

The measurement is what makes this safe rather than convenient: the probe reports a 0 px median
inter-channel offset with a sharp correlation peak, so a shared mask lands exactly on all three
channels. The editor shows the same fact — a blend mode composites a second channel at adjustable
alpha, which is the only way to see a few pixels of drift; side by side it is invisible.

Negative consequences, accepted honestly:

- **Storage is N× per completion.** Three identical PNGs per part on a three-channel dataset. Cheap,
  and the alternative was the coupling above.
- **The registration guarantee is not enforced, only measured.** Nothing stops an operator putting a
  badly-registered multi-shot dataset into sample scope; the shared mask would then be wrong on every
  channel but the one it was drawn on. The scan reports the offset and the editor makes it visible,
  and that is the whole defence. Geometric correction belongs to region profiles (ADR-0033), not to
  the annotation layer, where a document "never moves into model-input or canvas coordinates".
- **Two editing paths exist for one screen.** The branch is confined to one `DraftTarget` at the
  server-state boundary, but it is a branch, and a route added to one scope and not the other will
  read as a bug in the other.
- **Scope changes need an empty desk.** Refusing while any draft is open is the simplest honest rule,
  and on a large labelling pass it means finding and finishing the open one first.
- **A dataset re-imported with source masks after entering sample scope is in an odd state**: the new
  masks are ignored by the sample routes rather than refused. Opening a shared draft on a sample whose
  newest revision has a `source_mask` base is a `409`, which is the guard that matters, but the
  ordering itself is not prevented.

## Changelog

- **2026-08-14** — "Scope changes need an empty desk" stands, but the desk is now reachable. Drafts
  are created by the first save (see ADR-0032's changelog), so an open draft names real unfinished
  work rather than every image ever looked at, and the discard the blocker asks for is a route that
  exists. No untouched drafts are silently deleted by the transition: after the write-path fix there
  are none to delete.
