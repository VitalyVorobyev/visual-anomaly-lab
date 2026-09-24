# ADR-0011: Evaluation protocol for grouped samples

**Status:** Accepted (2026-08-06)

## Context

Methods emit **per-image** scores (see ADR-0007), but labels and splits belong to the **sample**
(see ADR-0005). A part photographed under three illuminations yields three scores and one label.
Something must reduce them to one, and that reduction is a detection decision, not formatting.

The evaluation layer must also stay independent of every method, so that all of them are compared
under one protocol, and it must handle unlabeled samples, datasets with and without pixel masks,
and a threshold the operator wants to move freely.

The alternatives for the reduction were `mean` (robust, and dilutes single-view evidence), a learned
fusion (needs labels the workbench cannot assume), and pushing the reduction into each method
(which would make every method's number partly a measure of its own fusion).

## Decision

**Evaluation is model-independent.** It reads stored image scores, sample labels and split
assignments, and never imports a model.

- **Channel-to-sample aggregation is `max` by default**, with `mean` available. A defect visible
  under *any* view makes the part defective; averaging halves exactly that evidence.
- **Per-channel normalization precedes the reduction**, because `max` assumes channels share a
  scale and a deep method's channels usually do not — one view's scores sit higher and win every
  maximum. It is off by default, and offered as `robust_z` or `rank`. It is fitted once over every
  image the experiment scored, **labels ignored**, so the metric cannot become a function of the
  answer. Aggregation and normalization are both recorded on each sample result.
- **The headline is sample-level ROC-AUC.** Image-level ROC-AUC is reported beside it on **raw**
  scores, to isolate model quality from the aggregation. Pixel-level metrics are computed wherever
  masks exist.
- **Everything threshold-dependent is derived on demand** — confusion matrix, precision, recall,
  false-positive and false-negative lists — and **nothing is persisted per threshold**. How a
  threshold is resolved across runs is ADR-0028's. A per-image localization verdict is stored
  because it is threshold-free, and it judges a part by the image that produced its aggregate score.
- **Unlabeled samples are ranked, never scored.** They appear in the ranked lists and in no metric.
- **A metric that cannot be computed is absent, never zero.**
- **Splits are seeded, sample-level, train on normals only, and stratified by capture group.** A
  benchmark's official partition is reproduced verbatim instead.
- **Re-evaluation rebuilds sample results and metrics from stored scores**, so changing the
  aggregation never requires re-inference.

## Consequences

A new method costs nothing here, and every method is read under one protocol. The threshold control
is instant. Recording the aggregation and normalization keeps old results interpretable.

- **`max` is the least robust aggregator.** One noisy view — a specular flare, a registration
  failure — sets the part's score. It maximizes sensitivity at the cost of false positives.
- **Normalization is opt-in.** A run left at the default is still exposed to scale artifacts
  between channels; `rank` removes them and discards magnitude, so a dramatic outlier and a marginal
  one score the same.
- **ROC-AUC hides operating-point behaviour**, and on a small dataset it is a noisy estimate; the
  tool offers no confidence intervals, so small differences are not significant.
- **Small validation and test sets.** After training normals are reserved, a threshold chosen on
  validation may not transfer, and one sample visibly moves a metric.
- **A good score with a nonsensical map** is only caught where masks or annotations exist.
- **On-demand derivation repeats work**, linear in the number of scored samples.
