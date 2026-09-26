# ADR-0041: Truth is task-scoped — the sample owns its anomaly label, and classes live in annotations

**Status:** Accepted (2026-09-26). Supersedes ADR-0005, and restates what it keeps from it.

## Context

ADR-0005 made the `Sample` the unit of identity, labelling and splitting, and gave it *the* label:
`normal`, `defect` or `unlabeled`. That was written when anomaly ranking was the only task. Three
more tasks now read truth from completed annotations — few-shot segmentation (ADR-0040), semantic
segmentation and object detection (ADR-0039) — and their truth is a class, not a verdict. A board
with a short circuit is not "defective" to a detector; it shows an instance of `short`. An FSS-1000
image of a bucket is neither normal nor defective at all.

With one label for every task, a dataset of classes has to be bent into anomaly shape. A few-shot
panel of twenty classes becomes twenty datasets over the same 200 images, each calling its target
class `defect` and the other nineteen classes `normal`; a detection dataset carries `defect` on
every sample because its adapter has no other word for "this directory has objects in it".

The alternatives considered:

- **Keep one label and encode class truth in it per dataset.** It is what the panel did. It costs a
  copy of the dataset per class, and it cannot say two classes at once.
- **A per-sample class column or tag list beside the label.** It duplicates what a completed
  annotation already records, per image and per pixel, and would have to be kept in step with it.
- **A nullable label, `NULL` meaning "not applicable".** It adds a fourth state that means what
  `unlabeled` already means — no anomaly verdict — and every reader would have to treat the two
  alike.

## Decision

**Truth is scoped by the task that reads it.**

- **The sample's label is anomaly truth.** `normal` and `defect` are verdicts; `unlabeled` is the
  absence of one, not a third verdict. An adapter sets a verdict only where its source asserts one —
  a directory of good parts, a label column — never as a stand-in for "has annotations".
- **Class truth lives in completed annotations** (ADR-0032). A completed revision answers for every
  class the dataset had when it was completed: present where it holds a region or box of the class,
  absent where it holds none. A source that ships class truth — boxes, or a mask per image of a
  named class — enters it as each image's first completed revision and sets no label.
- **The label stays non-null.** A dataset whose truth is its classes leaves every sample
  `unlabeled`, which already reads as "no anomaly verdict" everywhere.
- **One bridge, and only one.** The default class `defect` is the anomaly class, so anomaly truth
  answers for it: an imported ground-truth mask is its region, and a sample labelled `normal` is its
  absence. For every other class, only a completed annotation answers.
- **A multi-class source is one dataset.** A task that reads one class (ADR-0040) names it on the
  experiment; its negatives are the images whose completed annotation does not show that class.
  No dataset is copied per class.
- **What truth a dataset holds is derived, never stored**: `labels` when a sample has a verdict,
  `classes` when a completed annotation shows a class, both, or neither. The screens follow it: a
  dataset of classes is counted, filtered and covered by its classes, and is offered anomaly
  detection only once a sample carries a verdict.

What ADR-0005 decided is kept:

- **`Sample` is the unit of identity, labelling and splitting**, identified by
  `(dataset_id, group_key, external_id)`; `group_key` namespaces ids that collide across capture
  groups. **SplitAssignment is sample-level**, so every image of a part lands in the same subset.
- **`Channel` is a per-dataset data row, never a schema dimension.** An `Image` belongs to one
  sample and has an optional channel; nothing in the schema, the queries or the model interface
  encodes a channel count.
- **Pixel ground truth is keyed to `Image`**, so it attaches to one view of a part.
- **An `Experiment` freezes its configuration at creation.** Re-running with different settings is
  a new experiment; there is no `Run` entity.

## Consequences

- One source is one catalogue entry. FSS-1000's panel is one dataset of twenty classes, and a
  few-shot run on any of them needs no re-import. A detection dataset no longer claims hundreds of
  defects it never asserted.
- Anomaly and class tasks can share a dataset without either lying about the other: a labelled
  anomaly dataset can gain classes by annotation, and a class dataset can gain verdicts by labelling.
- **The bridge is an overlap, taken knowingly.** On an anomaly dataset the `defect` class and the
  `defect` verdict describe the same thing through two stores, and a revision completed without a
  `defect` region on a sample labelled `defect` says two different things. The completed revision
  wins for class reads; the label wins for anomaly reads.
- **A verdict on a class dataset says nothing about classes.** Labelling an FSS-1000 image `normal`
  opts the dataset into anomaly detection and does not make it absent of anything.
- Deriving the truth kind costs a query per dataset on every catalogue read. Only completed
  revisions with a class table count as class truth; an imported mask is anomaly truth and does not.
- Kept from ADR-0005: sample-level splits cost data; there are no per-image verdicts, so a part
  whose defect shows in one view is still a defective part (the evaluation layer aggregates, handbook evaluation.md); re-running
  multiplies experiments; `group_key` is import-derived and load-bearing; and nothing records
  whether two channels' scores share a scale.
