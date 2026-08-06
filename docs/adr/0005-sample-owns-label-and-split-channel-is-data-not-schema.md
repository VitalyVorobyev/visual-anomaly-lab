# ADR-0005: Sample owns label and split; channel is data, not schema

**Status:** Accepted (2026-08-06)

## Context

In the reference dataset a single physical part is photographed under three illuminations
(bright-field, dark-field, dome) almost simultaneously; the three files share a filename stem and
live in sibling folders. But one group in `unsorted/` has only **two** channels, and the
folder names vary across the corpus (`Bright`/`BrightField`/`Brightfield`, and so on). The brief is
explicit: support grouped multi-view samples, but "do not hard-code exactly three illuminations"
and do not assume future datasets use this acquisition setup.

Two further facts constrain the schema. Numeric image IDs are unique only *within* a capture group,
so `set1/12` and `set2/12` are different parts. And there are **no pixel masks** anywhere in the
dataset, so nothing pixel-level can be evaluated today (see ADR-0011).

The central modelling risk is leakage: if the three images of one part were assigned to splits
independently, a model could train on the bright-field view of a part and be tested on its dark-field
view — the same physical object, yielding optimistic and meaningless numbers.

## Decision

**`Sample` is the unit of identity, labelling, and splitting. `Channel` is a per-dataset data row,
never a schema dimension.**

- **Sample** = one logical physical part. Identified by `(dataset_id, group_key, external_id)`;
  `group_key` namespaces the numeric IDs that collide across capture groups. It owns
  `label ∈ {normal, defect, unlabeled}`.
- **Image** belongs to exactly one Sample and has an **optional** `channel_id`. A sample may hold
  one image or many; nothing in the schema, queries, or model interfaces encodes a channel count.
- **Channel** rows form a **per-dataset dictionary**, populated at import from canonicalized folder
  names (see ADR-0006). Two channels, three, or none are all ordinary cases.
- **SplitAssignment is sample-level**: `(split_id, sample_id) -> subset ∈ {train, val, test}`. Every
  image of a part therefore lands in the same subset, by construction. There is no image-level split.
- **Mask** exists as a table keyed to Image, deliberately unpopulated. It reserves the shape the
  brief calls for without inviting pixel-level metrics we cannot compute.
- **Experiment freezes its configuration at creation** — dataset, split, model key, model config,
  preprocessing config — and is immutable thereafter. **Re-running is a new Experiment.** There is
  no `Run` entity nested under an experiment.

The complete model is: `Dataset`, `Channel`, `Sample`, `Image`, `Mask`, `Split`, `SplitAssignment`,
`Experiment`, `Job`, `ImageResult`, `SampleResult`, `MetricSet`.

## Consequences

Train/test leakage across views of one part is structurally impossible rather than a rule someone has
to remember. The two-channel group and any future single-image dataset import without special
cases. An experiment is a self-contained, reproducible record: its config cannot drift after results
are attached to it, so a comparison view can trust what it displays.

Negative consequences, accepted honestly:

- **Sample-level splits cost data.** With 189 labeled parts, splitting by sample gives far fewer
  independent training items than splitting by image would. This is correct but statistically
  expensive on a small dataset.
- **No per-image labels.** A part whose defect is genuinely visible in only one channel is still
  labelled defect as a whole; a per-channel model is then trained against a label its input may not
  support. This pushes the problem into aggregation (see ADR-0011) rather than solving it.
- **"Re-run = new experiment" multiplies rows.** Tuning a parameter ten times leaves ten experiments
  and ten artifact directories, with no built-in notion of "these are the same thing, retried".
  Comparison views and cleanup have to cope with clutter.
- **`group_key` is import-derived and load-bearing.** If an adapter derives it differently between
  imports, sample identity silently changes and results no longer line up across datasets.
- **The unused Mask table is speculative.** It will look like dead schema until a masked dataset
  arrives, and it may not match that dataset's actual needs when it does.
- **Channel comparability is not modelled.** Because channels are just rows, nothing records whether
  two channels' scores are on the same scale — an assumption ADR-0011 has to make explicitly.
