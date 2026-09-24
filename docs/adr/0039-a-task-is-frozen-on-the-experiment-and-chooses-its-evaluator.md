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
- **Class-index materialisation on revisions will cost a schema migration** (ADR-0004).
