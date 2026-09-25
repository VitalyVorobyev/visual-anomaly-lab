# ADR-0039: A task is frozen on the experiment, and it chooses the evaluator

**Status:** Accepted (2026-09-24)

## Context

The workbench ranks images by how anomalous they are, trained on normals only. The next things it is
asked to do are **semantic segmentation** (a class per pixel) and **object detection** (boxes with a
class and a confidence), and after them instance segmentation.

Most of the machinery is already task-neutral: import, samples and channels, splits, region
profiles, the job queue, the resident worker, the method registry and its schema-driven forms, the
versioned annotation store whose shapes carry a label key, the per-dataset taxonomy and the image
viewer. The anomaly-shaped seams are few: a prediction is one score and one map, an image record
carries no ground truth, training filters to normals, the evaluator computes only anomaly metrics,
completing an annotation discards the class, and the result screens know only scores and heatmaps.

Three designs were live. **A separate application** would copy import, splits, jobs, annotation and
the viewer to change those seams. **The task as a property of the dataset** fails on the first public
dataset: VisA's pixel masks make it both an anomaly and a segmentation benchmark. **The task implied
by the method** leaves the evaluator and the training-set policy undecided until a plugin is loaded,
and a method can support more than one task.

## Decision

**A `Task` is chosen when an experiment is created and frozen on it**, like its split and its region
profile. A dataset has no task; whether its annotation is ready for one is a readiness check, not a
column.

- **A method declares the tasks it supports** in `Capabilities.tasks`, defaulting to `anomaly` so no
  existing plugin changes. Creation refuses a method that does not declare the task, and the picker
  filters by it. The plugin boundary stays one module and one registry entry (ADR-0007).
- **A prediction keeps an image-level `score` for every task**, so ranking, thresholds and the
  disagreement view keep working. A task adds optional outputs: a class-index label map for
  segmentation, and instances (box, class, confidence, optional mask) for detection.
- **Ground truth reaches training only through an explicit target provider** that is absent for
  `anomaly`, so an anomaly method cannot see a defect mask by construction. The training set is the
  task's policy: normals of the train subset for `anomaly`, annotated samples for supervised tasks.
- **Evaluation is chosen by task from a registry**, as job kinds and methods are. The anomaly
  evaluator is the existing one, unchanged. A task with no evaluator cannot be created.
- **ADR-0028 applies unchanged.** A confidence is not comparable across runs, so any cut on it is
  resolved per run by one shared rule, and comparison shows only threshold-free metrics side by side.
- **A supervised run — segmentation or detection — pins its class list at creation**: every class of
  the dataset, in taxonomy order, stored on the experiment. Class `i` is label index `i + 1` in its
  targets, its label maps and its confusion matrix; 0 is background. Deriving the list from the taxonomy at train
  time was the alternative, and it fails quietly: a class added or reordered between training and
  evaluation renumbers the labels under a stored model.
- **An image is labelled for a supervised run only when its truth answers every pinned class**, by the
  same presence rule a targeted task reads per class. An image with a gap is unlabelled, excluded from
  fit and metrics and counted, because a pixel of an unanswered class would otherwise read as
  background — a negative nobody asserted.
- **A supervised split draws annotated samples, stratified by the set of classes each one shows**
  (`class_stratified`). Only samples whose truth answers every class of the dataset are drawn; the
  rest go to `test`, scored and measured against nothing. Each class signature — `{}`, `{scratch}`,
  `{scratch, dent}` — takes its proportional share of `train`, with largest-remainder rounding so
  the total is exact; then a class two or more samples show gets a training sample if the draw left
  it none, and a test sample if that strands no other class. Three alternatives were live. **An unstratified draw**
  is simpler and lets a rare class land wholly on one side of the split, which reads as a method
  failing on it. **Iterative stratification per class** balances rare classes of a multi-label
  dataset more closely, but its assignment depends on the order it visits classes and samples and
  is hard to state on screen. **Reusing `normal_only_train`** trains on normals, which a supervised
  run cannot learn a class from. Signatures stay exact while a dataset has few classes; a taxonomy
  large enough to make most signatures singletons is when iterative stratification is worth its
  cost.
- **A segmentation prediction's `score` is the share of the image given a class other than
  background.** A mean maximum probability was the alternative; it needs a calibrated probability
  that a Gaussian classifier and a linear head do not share, while the share is defined by the label
  map alone and means the same thing for every method.
- **A detection prediction's `score` is its highest confidence**, 0 when it found nothing. The count
  of boxes was the alternative; it ranks a busy image above a clearly defective one, while the top
  confidence ranks by the one finding a reviewer would open the image for.
- **Detection truth is a revision's object instances**, one box per instance over the pixels it
  finally owns. An imported binary mask has no instances, so **each 8-connected component of it is
  one**, of the default class; so is each component of a `source_mask` base that no drawn instance
  owns. Treating mask-only images as unlabelled for detection was the alternative, and it would make
  every public benchmark whose truth is a mask unusable for the task until each image was redrawn.
  Components merge two touching objects and split one broken object, and the handbook says so.
- **Detection is read by COCO's protocol**: per class, detections ranked by confidence are matched
  greedily to the unmatched truth box of highest IoU at each of ten IoU thresholds from 0.50 to 0.95;
  AP is the 101-point interpolated area under the precision envelope, averaged over the thresholds
  and then over the classes that have truth. The headline is that average, AP@[.5:.95], with AP50
  beside it. Pascal VOC's AP at 0.5 alone was the alternative; it is what small defect benchmarks
  often report, but it cannot tell a box that grazes a defect from one that fits it. At most 100
  detections per image count, COCO's cap, and the write seam refuses more rather than dropping them.
  No metric cuts a confidence. A per-sample verdict and a drawn box do, so each subset resolves one
  cut by one rule — the confidence that maximises F1 at IoU 0.5, every class pooled — stored beside
  the metrics and printed wherever a verdict is drawn (ADR-0028). A fixed confidence was the
  alternative, and it would mean a different operating point for every method, since confidences are
  on each method's own scale; a cut per class was another, and it would print as many values as the
  run has classes beside one verdict. The cut never crosses runs: Compare reads the AP family alone.
- **The annotation document grows, it is not replaced** (ADR-0032): boxes and instance ids join the
  versioned document, completion keeps writing the binary mask the anomaly task reads, and a revision
  pins the class-to-index table it used so a renamed or reordered class cannot relabel old truth.

## Consequences

- One app, one catalogue and one annotation store serve every task, and a public dataset with masks
  is usable for anomaly and segmentation without being imported twice.
- **Every result screen branches on the task.** The shared shell is what keeps this from becoming N
  copies of each screen.
- **Grouped samples are undecided outside `anomaly`.** ADR-0011 aggregates image scores into a
  sample verdict; what a sample-level segmentation or detection result means is not settled here,
  and the first supervised tasks are evaluated per image.
- **The taxonomy becomes load-bearing**: it needs a management screen and class hotkeys that do not
  collide with the viewer's keys before a supervised task is usable.
- **`Sample.label` stays the anomaly verdict.** It is not a class, and a segmentation run ignores it.
- **Class-index materialisation on revisions, and the pinned class list, each cost a schema migration**
  (ADR-0004).
