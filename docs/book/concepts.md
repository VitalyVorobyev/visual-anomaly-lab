# Core concepts

The UI is dataset-centered, and the domain model follows the same hierarchy.

| Concept | Meaning |
|---|---|
| `Dataset` | One coherent inspection problem and its source provenance. |
| `Sample` | One labelled unit that must not be split across train and test. |
| `Image` | One view or channel belonging to a sample. Channel count is data. |
| `Split` | An immutable, seeded or imported assignment of samples to subsets. |
| `RegionProfile` | A versioned source-to-prepared transform for the dataset. |
| `Experiment` | Dataset + split + method + method config + preprocessing + pinned region revision. |
| `Job` | One asynchronous import, preparation, training, inference, or export execution. |
| `ImageResult` | A score, map path, and timing for one image in one experiment. |
| `SampleResult` | The evaluation-layer aggregation of a sample's image results. |
| `MetricSet` | Threshold-independent metrics and the data needed to resolve operating points. |

## Labels, annotations, and predictions are different

A sample label says whether the logical sample is normal or defective. A defect annotation says *where* a
known defect lies in source-image coordinates. A model prediction is evidence produced by one experiment:
an image score and, when supported, an anomaly map. Editing an annotation must not rewrite a past model
result, and moving a display threshold must not rewrite either.

## Source frame and prepared frame

Annotations belong to the immutable source image. Models consume a fixed prepared tensor. A region profile
defines how source pixels are cropped, masked, resized, and padded into that tensor and how a resulting map
is projected back. The experiment pins the exact profile revision and manifest hash; later region edits do
not silently alter an old run.

## One-class training

Most supported methods learn only from normal training images. Defects belong in validation or test data
for measurement, not in the normal reference. An imported benchmark split should be adopted rather than
redrawn when comparison with published evidence matters.

## Scores are local to a run

One method may score around `0.06` and another around `14` on the same images. Both can be correct. ROC-AUC,
PR-AUC, and AU-PRO compare rankings or integrate thresholds and can be compared under one protocol. A
confusion matrix needs an operating-point rule resolved separately for each run. The UI prints both the rule
and resulting value.

## Immutable evidence, deletable workspace

Experiments freeze their inputs so they remain interpretable. Jobs and results are append-like evidence.
The UI nevertheless lets the user delete experiments, datasets, drafts, and on-demand diagnostics to manage
the local workspace; dependencies are surfaced and destructive actions require confirmation.
