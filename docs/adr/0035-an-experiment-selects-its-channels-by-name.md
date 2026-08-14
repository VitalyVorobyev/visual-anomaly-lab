# ADR-0035 — An experiment selects its channels, by name

**Status:** Accepted (2026-08-14)

## Context

ADR-0005 made `Channel` a per-dataset data row and `Sample` the unit of identity, labelling and splitting, so
one physical part photographed under three illuminations is one sample owning three images. Nothing since has
let a *run* say which of those images it wants.

That absence had a workaround, and the workaround was the problem. The only way to ask "how well does
bright-field alone do?" was to import the bright-field images as a dataset of their own — which is exactly
what had happened to the reference corpus, where three catalogue entries turned out to be one dataset whose
channel folders had been reorganised to make three imports possible. Three datasets over the same 189 physical
parts defeats the design in three ways at once:

- **Leakage returns.** Sample-level `SplitAssignment` exists so a part's views cannot straddle subsets. Three
  datasets means three independent splits over the same parts, so one part's bright view can train while its
  dark view is tested.
- **Nothing aggregates.** `eval/aggregate.py` reduces a sample's per-channel scores to one verdict. With one
  channel per dataset there is nothing to reduce, and ADR-0011's whole protocol is inert.
- **The comparison is not a comparison.** Two numbers from two imports differ by the channel *and* by
  whatever else differed between the imports, with nothing recording which.

## Decision

**`Experiment.channels` is a frozen JSON array of channel names. Empty means every channel.**

- **Names, not ids.** `upsert_channel` matches on `(dataset_id, name)`, so ids do in fact survive a re-import
  — the argument is legibility, not stability. An experiment is a frozen scientific record that has to stay
  readable in a job log, a stored manifest and an audit script months later, and `["bright"]` says what `[17]`
  does not. Every other frozen column already stores meaning this way: `model_type` is a registry key,
  `preprocessing_config` stores resolved dimensions, and `ImageRecord.channel` — the plugin boundary itself —
  is already a name.
- **Empty means all**, matching the "an empty control means unset" contract the option forms already use, and
  making migration 013 a plain `ADD COLUMN` with no backfill and nothing discarded.
- **Applied in `list_images_for_split` and nowhere else.** That function's docstring already promises the
  "which images" question has exactly one answer in the codebase; training, inference and the on-demand
  diagnostic path narrow identically because none of them does its own filtering.
- **An image with no channel is excluded by a non-empty selection.** It belongs to no named channel, and
  "unassigned" is not a synonym for "all of them".
- **An unknown name is a 422 at creation**, naming what the dataset actually has. The selection is stored in
  `Channel.position` order rather than the order the client sent, so two identical requests produce identical
  frozen records.
- **No plugin changes.** A method still receives a flat sequence of records and still returns one prediction
  per input; it simply receives fewer of them.

Rejected: putting the selection on the `Split`. A split is about *which samples* and exists to prevent
leakage; which views a model is shown is about what the model sees, which is the experiment's business, in the
same way preprocessing is. Putting it on the split would also make "bright only" and "all three" incomparable
by construction, which is the one thing this decision exists to enable.

## Consequences

The interesting comparison becomes a first-class experiment variable: two runs on one split, one region build
and one set of labels, differing only in the channel. Merging a per-channel dataset family back into the
single multi-channel dataset ADR-0005 describes therefore loses nothing, and the reference corpus's 113
previously-unimported unlabeled samples came back with it.

Negative consequences, accepted honestly:

- **A selection can silently shrink a run.** Choosing channels that no sample in the split carries produces a
  legal, empty run. The unknown-name check catches typos, not a valid name that happens to match nothing in
  this split.
- **The frozen name is only as stable as the channel dictionary.** An operator who renames a channel by
  re-importing under a different canonicalization leaves old experiments naming a channel that no longer
  exists. The record stays readable, which is the point, but it stops being re-runnable.
- **Comparison across differing selections needs care.** Two runs that read different channels are legitimately
  comparable — that is the feature — but a viewer that indexes a sample's images positionally will line up the
  wrong ones. Selecting by name is now required wherever per-channel results are displayed side by side.
- **It multiplies experiments.** The channel is one more axis on a catalogue that ADR-0005 already noted has
  no notion of "these are the same thing, retried".
