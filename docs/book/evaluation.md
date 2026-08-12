# Inference, evaluation, and comparison

Inference produces one score per image and usually one raw prepared-frame anomaly map. Evaluation aggregates
images into samples, resolves labels and masks from the catalogue, and computes evidence without importing a
model implementation.

## From image to sample

Every `Prediction` belongs to an `Image`. For a multi-view sample, the evaluation layer aggregates image
scores into one sample score under the shared protocol. The model does not receive ground truth and cannot
change aggregation by method name.

Raw maps are persisted as numeric arrays. Colour, opacity, display cut, and segmentation outline are render
choices applied later. Moving them does not change stored metrics.

## Threshold-independent metrics

- **Image/sample ROC-AUC:** probability that a randomly chosen defect ranks above a normal.
- **Image/sample PR-AUC:** precision-recall area; more informative when defects are rare.
- **Pixel ROC-AUC:** ranking of source-frame defect versus non-defect pixels.
- **AU-PRO:** region-overlap quality integrated over false-positive rates, emphasizing connected defects.

Pixel metrics require aligned masks and inverse-projected source maps. If a metric cannot be computed, it is
`None` and renders as a dash—never zero.

## Operating points

Precision, recall, F1, confusion matrices, and FP/FN lists require a threshold. The workbench stores enough
scores to resolve these on demand. A named rule—such as F1-optimal on a declared subset—produces a different
numeric threshold for every run. The UI shows both rule and value.

The map segmentation cut is separate from the image-score threshold. It is a fraction of that run's map
range used for display and source overlay; it does not feed sample classification or pixel metrics.

## Reading the sample gallery

Rank by most anomalous, most normal, or closeness to the operating point. Filter to true positives, true
negatives, false positives, false negatives, or all mistakes. Inspect source image, map, predicted outline,
ground truth, and raw cursor values together. A metric tells you how often; these samples tell you why.

## Comparing experiments

Comparison requires the same dataset and split. Threshold-independent metrics appear side by side.
Threshold-dependent rows resolve the same rule separately for each run. The configuration diff highlights
preprocessing and region differences. The disagreement table is often more useful than a mean: it identifies
samples whose ranking or verdict depends on the chosen method.

Never subtract raw maps from different methods or place their scores on one numeric slider. Their units and
calibration are unrelated. Shared visualization uses per-run values and a shared *fraction* of each range.

See [Interpreting metrics](metrics.md) and the canonical
[evaluation handbook](../architecture/evaluation.md) for formulas and edge cases.
