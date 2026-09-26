# ADR-0039: A task is frozen on the experiment, and it chooses the evaluator

**Status:** Accepted (2026-09-24)

## Context

The workbench ranks images by how anomalous they are, trained on normals only. It is also asked to
do **semantic segmentation** (a class per pixel) and **object detection** (boxes with a class and a
confidence), and after them instance segmentation.

Most of the machinery is task-neutral: import, samples and channels, splits, region profiles, the
job queue, the resident worker, the method registry and its schema-driven forms, the versioned
annotation store whose shapes carry a label key, the per-dataset taxonomy and the image viewer. The
anomaly-shaped seams are few: a prediction is one score and one map, an image record carries no
ground truth, training filters to normals, the evaluator computes only anomaly metrics, and the
result screens know only scores and heatmaps.

## Decision

**A `Task` is chosen when an experiment is created and frozen on it**, like its split and its region
profile. A dataset has no task; whether its annotation is ready for one is a readiness check, not a
column.

- **A method declares the tasks it supports** in `Capabilities.tasks`, defaulting to `anomaly`.
  Creation refuses a method that does not declare the task. The plugin boundary stays one module and
  one registry entry (ADR-0007).
- **A prediction keeps an image-level `score` for every task**, so ranking, thresholds and the
  disagreement view keep working. A task adds optional outputs: a class-index label map for
  segmentation, instances for detection. A segmentation score is the share of the image given a
  class other than background; a detection score is its highest confidence.
- **Ground truth reaches training only through an explicit target provider** that is absent for
  `anomaly`, so an anomaly method cannot see a defect mask by construction. The training set is the
  task's policy: normals of the train subset for `anomaly`, annotated samples for supervised tasks.
- **Evaluation is chosen by task from a registry**, as job kinds and methods are. A task with no
  evaluator cannot be created. ADR-0028 applies unchanged: a confidence is not comparable across
  runs, so any cut on it is resolved per run by one shared rule, and comparison shows only
  threshold-free metrics side by side.
- **A supervised run pins its class list at creation**, in taxonomy order. Class `i` is label index
  `i + 1` everywhere; 0 is background. An image is labelled for such a run only when its truth
  answers every pinned class; one with a gap is unlabelled, excluded and counted.
- **A supervised split draws annotated samples stratified by the set of classes each one shows**
  (`class_stratified`).
- **Detection truth is a revision's object instances.** An imported binary mask has no instances, so
  each 8-connected component of it is one, of the default class. Boxes a dataset ships enter as each
  image's first completed revision.
- **Detection is read by COCO's protocol**, AP@[.5:.95] with AP50 beside it. A per-sample verdict
  resolves one confidence cut per subset by one printed rule, and the cut never crosses runs.
- **The annotation document grows, it is not replaced** (ADR-0032): boxes and instance ids join the
  versioned document, and a revision pins the class-to-index table it used.

The contracts — score definitions, the split's rounding, the matching rule, the cut — are in the
handbook's [evaluation](../architecture/evaluation.md), [methods](../architecture/methods.md) and
[annotations](../architecture/annotations.md) pages.

## Alternatives considered

- **A separate application per task.** It would copy import, splits, jobs, annotation and the viewer
  to change a handful of seams.
- **The task as a property of the dataset.** Fails on the first public dataset: VisA's pixel masks
  make it both an anomaly and a segmentation benchmark.
- **The task implied by the method.** Leaves the evaluator and the training-set policy undecided
  until a plugin is loaded, and a method can support more than one task.
- **Deriving the class list from the taxonomy at train time.** A class added or reordered between
  training and evaluation silently renumbers the labels under a stored model.
- **An unstratified supervised split**, which lets a rare class land wholly on one side and read as
  a method failing on it; or **iterative per-class stratification**, whose assignment depends on
  visiting order and is hard to state on screen. Signature stratification holds while a dataset has
  few classes.
- **A mean maximum probability as the segmentation score**, which needs a calibration the methods do
  not share; **a box count as the detection score**, which ranks a busy image above a clearly
  defective one.
- **Treating mask-only images as unlabelled for detection**, which would make every public benchmark
  whose truth is a mask unusable for the task. **Shipped boxes as a second kind of imported source**
  beside the mask, which would need a second presence rule and a second reader in every consumer.
- **Pascal VOC's AP at IoU 0.5 alone**, which cannot tell a box that grazes a defect from one that
  fits it; **a fixed confidence cut**, a different operating point per method; **a cut per class**,
  as many printed values as classes beside one verdict.

## Consequences

- One app, one catalogue and one annotation store serve every task, and a public dataset with masks
  is usable for anomaly and segmentation without being imported twice.
- **Every result screen branches on the task.** The shared shell is what keeps this from becoming N
  copies of each screen.
- **Grouped samples are undecided outside `anomaly`.** The anomaly evaluator aggregates image scores
  into a sample verdict; what a sample-level segmentation or detection result means is not settled
  here, and supervised tasks are evaluated per image.
- **The taxonomy becomes load-bearing**: a rename or reorder is a data change with consequences, and
  it needs a management screen and class hotkeys that do not collide with the viewer's keys.
- **`Sample.label` stays the anomaly verdict.** It is not a class, and a segmentation run ignores it.
- **Connected components are a guess at instances.** They merge two touching objects and split one
  broken object.
- **Shipped boxes are app-owned truth from the start**, so a correction is a new revision rather
  than a divergence from the source file.
