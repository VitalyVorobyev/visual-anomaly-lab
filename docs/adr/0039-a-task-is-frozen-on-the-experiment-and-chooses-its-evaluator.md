# ADR-0039 — A task is frozen on the experiment, and it chooses the evaluator

**Status:** Accepted (2026-09-24)

Extends **ADR-0007** (a method costs one module and one registry entry) and **ADR-0011** (the
evaluation protocol) from one task to several. Constrained by **ADR-0004** (schema v1 is frozen, so
this is a numbered migration), **ADR-0028** (nothing is compared in score units) and **ADR-0032**
(annotation truth is versioned and source-frame).

## Context

The workbench does one thing: rank images by how anomalous they are, trained on normals only. The
next two things it is asked to do are **semantic segmentation** (a class for every pixel) and
**object detection** (boxes with a class and a confidence), and after them instance segmentation.

Most of the machinery is already task-neutral: import, samples and channels, splits, region
profiles, the job queue and its protocol, the resident worker, the method registry and its
schema-driven forms, the versioned annotation store — whose shapes already carry a `label_key` —
the per-dataset `AnnotationLabel` taxonomy, and the image viewer. Six seams are anomaly-shaped:
`Prediction` is one score and one map; `ImageRecord` carries no ground truth; training filters to
normals; `eval/runner.py` computes only anomaly metrics; completing an annotation renders a binary
mask and discards the class; and the result screens know only scores and heatmaps.

Three designs were live. **A separate application** would copy import, splits, jobs, annotation and
the viewer to change six seams. **The task as a property of the dataset** fails on the first public
dataset: VisA's pixel masks make it both an anomaly benchmark and a segmentation one, and the same
parts can be boxed for detection. **The task implied by the method** leaves the evaluator and the
training-set policy undecided until a plugin is loaded, and a method can support more than one task.

## Decision

**A `Task` is chosen when an experiment is created and frozen on it**, like its split and its region
profile: `anomaly`, `semantic_segmentation`, `object_detection` (and `instance_segmentation` when it
is needed). A migration adds `experiments.task`, defaulting every existing row to `anomaly`. A dataset
has no task; what it has is annotation, and whether it is ready for a task is a readiness check, not a
column.

**A method declares the tasks it supports** in `Capabilities.tasks`, defaulting to `[anomaly]` so
that no existing plugin changes. Creating an experiment refuses a method that does not declare the
task, and the method picker filters by it. `AnomalyModel` gains a task-neutral name with the old one
kept as an alias; the plugin boundary is otherwise unchanged — still one module and one entry.

**A prediction still has an image-level `score`**, so ranking, the threshold slider and the
disagreement view keep working for every task (for detection it is the top confidence). What a task
adds is optional: a class-index `label_map` for segmentation, and `instances` — box, class key,
confidence, optional mask — for detection and instance segmentation.

**Ground truth reaches training only through `TrainContext.targets`**, a `TargetProvider` that is
`None` for `anomaly`. An anomaly method therefore cannot see a defect mask by construction, not by
convention. The training-set policy is the task's: `anomaly` trains on the normals of the train
subset, as now; a supervised task trains on the annotated samples of the train subset.

**Evaluation is chosen by task from a registry**, the way job kinds and methods already are. The
anomaly evaluator is today's `eval/runner.py`, moved without changing a number. Segmentation
accumulates a per-class confusion matrix in constant memory and reports IoU and Dice per class and
their means; detection reports COCO-style AP at IoU 0.50 and 0.50:0.95 per class. ADR-0028 applies
unchanged: a confidence is not comparable across runs, so a confidence cut is resolved per run by one
shared rule, and compare puts only threshold-free metrics side by side.

**Annotation schema v2 adds a `BoxShape` and an optional `instance_id` on every shape**; a v1
document reads as v2 unchanged. Completion keeps rendering the binary mask the anomaly task reads,
and also writes a class-index PNG and an instances file, with the revision pinning the class-to-index
table it used so a renamed or reordered class cannot silently relabel old truth.

## Consequences

- One app, one catalogue and one annotation store serve every task, and a public dataset with masks
  is usable for anomaly and segmentation without being imported twice.
- **Every result screen branches on the task.** The overview, benchmark and sample views need a
  task-specific body. The shared shell — run bar, subset, the one `SampleStage` with its vector
  layer — is what keeps this from being N copies of each screen.
- **Grouped samples are an open question outside `anomaly`.** ADR-0011 aggregates image scores into
  a sample verdict; what a sample-level detection result means for a multi-channel part is not
  decided here, and the first supervised tasks will be evaluated per image.
- The taxonomy becomes load-bearing. `AnnotationLabel` exists but has no management screen, and the
  editor's class hotkeys collide with the `0`/`1` view keys; both are prerequisites, not follow-ups.
- `Sample.label` stays the anomaly verdict. It is not a class, and a segmentation run ignores it.
- Two migrations are committed to: the `task` column, and whatever the class-index materialisation
  needs on annotation revisions.
