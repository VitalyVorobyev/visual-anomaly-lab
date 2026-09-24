# ADR-0036: Annotation is edited per sample and stored per image

**Status:** Accepted (2026-08-14)

## Context

ADR-0005 makes `Sample` the unit of identity and `Channel` data, so one part photographed under
three illuminations is one sample owning three images. ADR-0032 made annotation truth versioned,
source-frame and image-keyed all the way down: drafts and revisions are keyed by image, and the one
resolver every consumer reads — pixel metrics, overlays, `has_mask`, the `MetricSet` ground-truth
digest and every interchange format — is keyed by image. A defect visible in one channel of a
registered capture therefore had to be traced once per channel, by hand, producing copies of one
truth that were free to drift apart.

Two shapes were live:

- **A sample-level annotation entity**, one revision per sample, with the resolver learning that it
  covers each image. Conceptually cleaner: truth about a part is stored once.
- **Sample-scoped editing over image-scoped truth**: one draft per sample, materialised on
  completion as one ordinary revision per image.

The first changes what "ground truth for image N" means for every reader of it, and those readers
are the evaluation layer.

## Decision

**`Dataset.annotation_scope` is `image` (the default) or `sample`. Under `sample`, one document is
edited per sample and materialised as one revision per image of that sample.**

- **Truth stays image-keyed.** The resolver, pixel metrics, `has_mask`, the ground-truth digest and
  the exports never learn that scope exists; only the editing surface moves up.
- **One render, N files, one digest.** Completion renders once and writes those bytes to each
  image's revision, so a shared mask digest makes "these channels carry the same truth" checkable.
- **A sample-scoped document has an empty base.** Source-mask provenance is per image and cannot be
  carried onto N images, so the sample draft is its own table rather than a row borrowed from a
  "carrier" image.
- **The scope is a property of the data**: channels that are exposures of one registered part share
  truth; channels that are unrelated views do not.
- **Entering `sample` scope is refused, with every reason at once**, when the dataset has imported
  source masks, when a sample mixes image dimensions, or when any draft is open. Leaving it is
  refused while a sample draft is open. Completed revisions stay valid either way.
- **Per-image writes are refused under sample scope** rather than redirected: two writers editing
  one part through two scopes would each hold a valid ETag for a different document.

Rejected: the sample-level entity. It saves N−1 PNGs per part, which ADR-0032 already accepts
paying, and charges for it where churn is least affordable: every ground-truth consumer would need
a second resolution path, the staleness digest would hash a mixture of identities, and a method
still predicts per image (ADR-0007), so sample-keyed truth would have to be joined back at every use.

## Consequences

- A part is annotated once however many times it was photographed, and the channels provably share
  that truth.
- **Storage is N× per completion.** Cheap, and the alternative was the coupling above.
- **Registration is measured, not enforced.** Nothing stops a badly registered dataset entering
  sample scope; the shared mask is then wrong on every channel but the one it was drawn on. The
  import scan reports the offset and the editor's channel blend makes drift visible, and that is the
  whole defence. Geometric correction belongs to region profiles (ADR-0033).
- **Two editing paths serve one screen.** The branch is confined to one server-state boundary, but a
  route added to one scope and not the other will read as a bug in the other.
- **Changing scope needs an empty desk**: every open draft finished or discarded first.
- **A dataset re-imported with source masks after entering sample scope is in an odd state**: the
  new masks are ignored by sample routes rather than refused. Opening a shared draft over a
  source-mask revision is refused, but the ordering itself is not prevented.
