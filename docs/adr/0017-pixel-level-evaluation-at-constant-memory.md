# ADR-0017: Pixel-level evaluation, at constant memory

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-07.

> **Read [`architecture/evaluation.md`](../architecture/evaluation.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

Supersedes the "no pixel-level evaluation, possibly ever" consequence of **ADR-0011**, and the
"image-level metrics only" constraint in its Context. Every other decision in ADR-0011 — the
aggregation policy, the on-demand threshold, the treatment of unlabeled samples — stands unchanged.

## Context

ADR-0011 ruled out pixel metrics on a fact about the data, not a fact about the design: the showcase
dataset ships no masks, so nothing pixel-level was computable. It recorded the cost honestly — "a
method with a good score and a nonsensical map is indistinguishable from a good one by metrics
alone."

The public reference datasets adopted in **ADR-0015** change the fact. VisA carries a pixel-level
ground-truth mask for every anomalous image, and **ADR-0016** brought those masks into the catalog.
The `Mask` table has existed since schema v1 and the maps have always been stored as float32 `.npy`
(**ADR-0007**), precisely so that this would need no migration and no re-inference when it became
possible.

Two things made the decision non-obvious.

**Memory.** A hundred VisA test images at ~1.5 MPix in float32 is roughly 600 MB if the maps are
accumulated to compute a curve, and the M5 comparison view multiplies that by the number of
experiments being compared. The naive implementation is the one every tutorial shows, and it does not
fit the shape of this application.

**Which images count.** Ground truth is attached only to *anomalous* images. Evaluating over those
alone is the tempting reading and it is wrong: the negative pixels would then be drawn exclusively
from inside defective images, producing a false-positive rate that no published number is comparable
to. That would quietly defeat the entire point of ADR-0015, which is to be able to check our numbers
against someone else's.

## Decision

**Pixel ROC-AUC and AU-PRO are computed for every subset that has masks, streamed through fixed-bin
histograms, over normal and anomalous images alike.**

- **Nothing is accumulated.** Each image is folded into three arrays sized by the bin count and then
  discarded: a positive-pixel score histogram, a negative-pixel score histogram, and one accumulator
  holding, per bin, the sum over ground-truth regions of *that region's fraction* of pixels in the
  bin. Memory is constant in the number of test images and in their resolution.
- **The region accumulator is exact, not an approximation.** PRO at a threshold is the mean over
  regions of each region's covered fraction. Dividing each region's per-bin counts by its own size
  *before* summing is what collapses "one histogram per region" into one array while computing the
  same quantity. AU-PRO integrates to a false-positive rate of 0.3, the convention every paper
  reports, with the endpoint interpolated so the limit is exactly 0.3 rather than the nearest bin.
- **Normal images are included with an all-zero mask.** This is the MVTec and VisA convention and it
  is what makes the reported figure comparable. A *defective* image with no mask is skipped and
  **counted**, because its defect pixels are somewhere and assuming they are nowhere would reward a
  model for missing it.
- **The map is upsampled to the mask's resolution**, bilinearly, rather than the mask being
  downsampled to the map's. Interpolating a label map invents labels; only one image is in flight at
  a time, so the cost is bounded.
- **Bins adapt to the observed score range**, in one cheap pass over the map files, because a fixed
  `[0, 1]` assumption is wrong for every method that does not normalize its output.
- **Everything skipped is reported** in the metric set and surfaced in words on the results screen. A
  pixel metric computed over half the defects is not the metric it claims to be, and a reader must
  not have to dig for that.

**Ruled out:** scikit-learn's `roc_auc_score` over concatenated maps (the 600 MB problem, and it
would make the evaluation layer depend on a package that only arrives with the optional
deep-learning group); computing pixel metrics at the preprocessing resolution (cheaper, but produces
a number that cannot be compared to a published one); and persisting per-image pixel scores (a
derived value, and the same argument ADR-0011 makes against persisting per-threshold results).

## Consequences

Pixel-level quality is now measurable, which is what makes an anomaly *map* a result rather than a
picture. A method with a good image score and a map that lights up on the wrong thing is now
visible as such. The comparison view in M5 gets a second axis for free, and none of it required a
migration, a re-inference, or a change to any model.

Negative consequences, accepted honestly:

- **Quantization.** Scores are binned at 2^16 levels across the observed range. The error is far
  below the noise of a 100-image test set, but the number is not bit-identical to a direct
  computation, and a comparison against a paper's figure inherits that.
- **Two reads of every map file per evaluation** — one to find the range, one to accumulate. Cheap
  at this scale and linear in dataset size, like everything else in ADR-0011's on-demand model.
- **AU-PRO's connected-component labelling is Python-level union-find.** Fine for sparse defect
  masks, which is what these datasets have; a dataset whose masks cover most of the frame would make
  it the slowest part of evaluation.
- **`verify` still cannot detect mask drift.** Schema v1 has no `mask.sha256` and the schema is
  frozen, so a mask re-exported in place is invisible to it — and a pixel metric computed against a
  silently changed mask is wrong with no indication. This was already recorded in ADR-0016; pixel
  metrics raise its cost, and lifting it is a numbered migration rather than a patch.
- **The protocol choice is now load-bearing and invisible.** "Normal images count, with an empty
  mask" is a decision that moves the number substantially, and nothing in the UI says so. It is
  documented here and in the code, which is weaker than showing it.
