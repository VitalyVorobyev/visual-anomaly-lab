# ADR-0011: Evaluation protocol for grouped samples

**Status:** Accepted (2026-08-06)

## Context

The brief requires image-level anomaly scores, ROC-AUC where labels permit, a
configurable-threshold confusion matrix, false-positive and false-negative lists, and per-sample
inference time — and insists the evaluation layer stay independent of individual model
implementations.

The domain model creates the central question: models emit **per-image** scores (ADR-0007), but
labels and splits belong to the **sample** (ADR-0005). A part photographed under three illuminations
yields three scores and one label. Something must reduce three numbers to one, and that reduction is
a substantive detection decision, not a formatting step.

The dataset constrains what can be measured: no pixel masks exist, so **image-level metrics only**;
and 113 samples in `unsorted/` are unlabeled, so they can be ranked but not scored.

## Decision

**The evaluation layer is model-independent.** It consumes only `ImageResult` scores, `Sample`
labels, and `SplitAssignment` rows. It never imports a model, and it works identically for the
classical baseline, the anomalib wrappers, and any future method.

- **Channel → sample aggregation: `max` by default.** A defect visible under *any* illumination
  makes the part defective — dark-field reveals scratches that bright-field cannot see. Averaging
  dilutes exactly that single-channel evidence: one strong signal among three views is halved or
  worse. `mean` is available as a configuration option, and the choice is **recorded with the
  experiment** so results remain interpretable.
  **Caveat:** `max` assumes per-channel scores are comparable. That holds for the z-scored classical
  baseline (handbook methods.md), whose scores are normalized against per-channel references by construction.
  It is *not* guaranteed for the deep-learning methods, where one channel's score distribution may
  simply sit higher and dominate every maximum. **Per-channel quantile normalization before
  aggregation is a backlog item**, not a shipped feature.
- **Headline metric: sample-level ROC-AUC.** Image-level ROC-AUC is also reported, since it isolates
  raw model quality from the aggregation choice. Both are stored in the `MetricSet`.
- **Threshold-dependent outputs are computed on demand.** Confusion matrix, precision, recall, and
  the FP/FN sample lists are derived from persisted scores whenever the user moves the threshold
  slider. **Nothing is persisted per threshold** — thresholds are a view concern, and storing them
  would multiply rows while fixing a decision that should stay explorable.
- **Unlabeled samples are ranked, never scored.** They appear in the most-normal / most-anomalous
  lists (that is where triage value lies for `unsorted/`) and are excluded from every metric.
- **Timing** comes from `ImageResult.inference_ms`, recorded per image at inference and aggregated
  for reporting.
- **Split guidance for the reference dataset:** train ≈ 60 **normal-only** samples; validation =
  held-out normals plus some defects, used for threshold selection; test = the remainder. Splits are
  seeded for reproducibility and stratified by capture group so no group lands entirely on one side.

## Consequences

Adding a method costs nothing in the evaluation layer, and every method is compared under exactly
one protocol — which is the point of the workbench. Because only raw scores are persisted, the
threshold slider is instant and re-thresholding never requires re-inference. Recording the
aggregation mode keeps old experiments interpretable after the default changes.

Negative consequences, accepted honestly:

- **`max` is the least robust aggregator.** One noisy channel — a specular flare, a registration
  failure — sets the sample score. It maximizes sensitivity and, on this dataset, will likely
  produce the false positives.
- **The comparability assumption is currently unenforced.** Until quantile normalization exists,
  cross-method comparisons under `max` may partly measure score-scale artifacts rather than
  detection quality. This is a real threat to the headline number's validity.
- **ROC-AUC hides operating-point behaviour.** With 98 normal and 91 defect samples it is also a
  noisy estimate; small AUC differences between methods will not be significant, and the tool offers
  no confidence intervals.
- **Tiny validation and test sets.** After reserving ~60 normals for training, the remaining
  splits are small enough that a threshold chosen on validation may not transfer, and single-sample
  changes visibly move metrics.
- **No pixel-level evaluation, possibly ever.** Anomaly maps can be looked at but not scored; a
  method with a good score and a nonsensical map is indistinguishable from a good one by metrics
  alone.
- **On-demand computation repeats work.** Every threshold change re-scans results; acceptable at
  this scale, but it grows linearly with dataset size.

## Changelog

- **2026-08-14:** Shipped this record's own named backlog item. Per-channel normalization
  (`EvalConfig.channel_normalization`, default `none`) now runs before the `max`/`mean` reduce and is
  recorded per row on `SampleResult.normalization`, so "the comparability assumption is currently
  unenforced" is no longer true by default and never true silently. Two transforms are offered: `robust_z`,
  and `rank` — which is scale-free but keeps only the ordering, so a dramatic outlier and a marginal one
  score identically. The transform is fitted **once over every image the experiment has scored, labels
  ignored**: `sample_result` has no subset column, so a per-subset fit is not representable, and fitting on
  labels would make the metric partly a function of the answer. Image-level ROC-AUC deliberately stays on raw
  scores, keeping its stated role of isolating model quality from the aggregation choice.
- **2026-08-14:** Fixed a gap this record's "re-thresholding never requires re-inference" claim depended on.
  `reevaluate` recomputed `MetricSet` rows without rebuilding `SampleResult`, so a changed aggregation mode
  appeared to apply and did not. Deriving sample scores moved out of the `infer` handler and into
  `evaluate_and_store`, which now owns the whole from-stored-scores path.
