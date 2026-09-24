# ADR-0040: Few-shot segmentation is a task, and its references are a split

**Status:** Accepted (2026-09-24)

## Context

The workbench is asked to do a second thing as a peer of anomaly ranking: given one to ten
`(image, mask)` references of an object type it has never seen, segment that type in new images.
There is no fixed taxonomy and, ideally, no per-task training. The current methods for this task
(INSID3, RSRM, FSSDINO; see [papers.md](../papers.md)) match frozen DINOv3 patch features against
foreground and background prototypes built from the references. That is the same machinery
`dino_memory` uses to model normality.

ADR-0039 already makes a task a frozen property of the experiment that chooses its evaluator. Three
questions were still open:

- **What the training set of a few-shot task is.** One option was a new "support set" entity beside
  splits. Another was to reuse the planned supervised `semantic_segmentation` task, which trains on
  every annotated sample of the train subset.
- **How many classes one run segments.**
- **What an image without the object means.**

## Decision

- **`few_shot_segmentation` is its own `Task`, separate from supervised `semantic_segmentation`.**
  Its fit consumes a handful of chosen references rather than a training corpus. Its measurements
  are about references (the shot count, and sensitivity to which references were drawn), and they
  mean nothing for a supervised run.
- **One experiment segments one class.** The class is an `AnnotationLabel` key frozen on the
  experiment, and the output is foreground against background. Several classes are several runs.
  Merging them into one label map is a later, separate choice.
- **The references are a split.**
  - Two split strategies are added: `manual`, whose `train` subset is an explicit list of sample
    ids, and `few_shot`, which draws `shots` samples containing the class under a seed.
  - `train` holds the references and `test` holds the queries.
  - Splits are already immutable and per sample, which is exactly the property a frozen reference
    set needs. So there is no new entity and no `Subset` migration.
- **Absence is truth, not a gap.**
  - A completed annotation revision with a region of the class is a positive.
  - A completed revision without one is a confirmed absence.
  - An image with no completed revision is unlabelled. It is excluded from metrics, and the number
    excluded is shown.
- **ADR-0039 applies unchanged.**
  - The masks reach `fit` only through `TrainContext.targets`.
  - A prediction's image-level `score` is presence confidence, and its map is the foreground
    probability.
  - The evaluator is chosen by task. It reports foreground IoU and Dice, boundary F1, the
    false-positive rate on absent images, recall on present images, and presence ROC-AUC.
  - Under ADR-0028, a threshold is resolved per run and never carried between runs as a value.
- **The first methods are training-free feature matching.** There is a torch-free colour floor, a
  prototype baseline, and our debiased multi-prototype method. SAM-family models are references
  and annotation assistants, not the runtime path.

## Consequences

- One catalogue, one annotation store and one viewer serve both tasks. A benchmark with pixel masks
  (VisA) is a few-shot benchmark with no second import.
- **Support sensitivity is measurable by construction.** The same `few_shot` split with three
  seeds gives three reference draws.
- An interactive "try references, look, fix" loop does not fit a frozen experiment. It lives in a
  studio that previews through the resident worker (ADR-0026), and it only freezes into an
  experiment when asked. That is a second path to the same run, and it has to be kept from growing
  into a second results screen.
- **Binary per class defers multi-class segmentation.** If a dataset has five classes, five runs
  are needed today.
- A dataset whose images are mostly unannotated yields small test sets. The excluded count is shown
  rather than hidden, but the metrics will be noisy until absences are confirmed.
