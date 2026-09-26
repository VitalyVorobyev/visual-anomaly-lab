# ADR-0005: Sample owns label and split; channel is data, not schema

**Status:** Superseded by ADR-0041 (2026-09-26)

## Context

In the showcase dataset one physical part is photographed under several illuminations almost
simultaneously; the files share a filename stem and live in sibling folders. Most parts have three
views, some have two, folder names vary, and numeric IDs repeat across capture groups, so the same
number in two groups names two different parts. Other datasets have one image per sample and no
channels at all. The model must represent all of these without a channel count anywhere in it.

The central modelling risk is leakage: if the views of one part were assigned to splits
independently, a model could train on one view of a part and be tested on another view of the same
object, yielding optimistic and meaningless numbers.

The alternatives were an image as the unit of labelling and splitting (simple, and leaky), a fixed
set of channel columns (simple, and wrong for the next dataset), and a nested `Run` under an
experiment for retries.

## Decision

**`Sample` is the unit of identity, labelling and splitting. `Channel` is a per-dataset data row,
never a schema dimension.**

- **Sample** is one logical physical part, identified by `(dataset_id, group_key, external_id)`;
  `group_key` namespaces the IDs that collide across capture groups. It owns the label
  (`normal`, `defect` or `unlabeled`).
- **Image** belongs to exactly one sample and has an **optional** channel. A sample may hold one
  image or many; nothing in the schema, the queries or the model interface encodes a count.
- **Channel** rows are a per-dataset dictionary populated at import (see ADR-0006). Two channels,
  three, or none are all ordinary cases.
- **SplitAssignment is sample-level.** Every image of a part lands in the same subset by
  construction; there is no image-level split.
- **Mask** is keyed to Image, so pixel ground truth attaches to one view of a part.
- **An Experiment freezes its configuration at creation** and is immutable thereafter. Re-running
  with different settings is a new experiment; there is no `Run` entity.

The canonical entities are `Dataset`, `Channel`, `Sample`, `Image`, `Mask`, `Split`,
`SplitAssignment`, `Experiment`, `Job`, `ImageResult`, `SampleResult`, `MetricSet`.

## Consequences

Leakage across views of one part is structurally impossible rather than a rule to remember. A
two-view part or a single-image dataset imports with no special case. An experiment is a
self-contained record whose configuration cannot drift after results are attached to it.

- **Sample-level splits cost data.** On a small dataset, splitting by part gives far fewer
  independent items than splitting by image would. Correct, and statistically expensive.
- **No per-image labels.** A part whose defect shows in one view only is still labelled defect as a
  whole, so a per-view model is trained against a label its input may not support. The problem is
  pushed into aggregation (see the handbook's [evaluation](../architecture/evaluation.md) page)
  rather than solved.
- **"Re-run is a new experiment" multiplies rows.** Tuning a parameter ten times leaves ten
  experiments and ten artifact directories, with no built-in notion of "the same thing, retried".
- **`group_key` is import-derived and load-bearing.** An adapter that derives it differently
  between imports silently changes sample identity.
- **Channel comparability is not modelled.** Because channels are rows, nothing records whether two
  channels' scores share a scale; evaluation has to handle that explicitly.
