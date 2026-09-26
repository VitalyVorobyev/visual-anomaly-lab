# ADR-0035: An experiment selects its channels, by name

**Status:** Accepted (2026-08-14)

## Context

ADR-0041 made `Channel` a per-dataset data row and `Sample` the unit of identity, labelling and
splitting, so one physical part photographed under three illuminations is one sample owning three
images. Without a way for a *run* to say which of those images it wants, the only way to ask "how
well does one illumination do alone?" is to import that channel as a dataset of its own. Several
datasets over the same physical parts defeat the design:

- **Leakage returns.** Sample-level `SplitAssignment` keeps a part's views from straddling subsets;
  separate datasets have independent splits, so one view of a part can train while another is tested.
- **Nothing aggregates.** With one channel per dataset, ADR-0011's sample-level reduction is inert.
- **The comparison is confounded.** Two numbers from two imports differ by the channel and by
  whatever else differed between the imports, with nothing recording which.

The live alternative was to put the selection on the `Split`.

## Decision

**`Experiment.channels` is a frozen list of channel names. Empty means every channel.**

- **Names, not ids.** An experiment is a frozen scientific record that must stay readable in a job
  log, a manifest and an audit script; `["bright"]` says what `[17]` does not. Other frozen columns
  already store meaning this way, and the plugin boundary's `ImageRecord.channel` is a name.
- **Empty means all**, matching the "an empty control means unset" contract of the option forms.
- **Applied in `list_images_for_split` and nowhere else**, so training, inference and on-demand
  diagnostics narrow identically.
- **An image with no channel is excluded by a non-empty selection**; unassigned is not a synonym for
  all of them.
- **An unknown name is refused at creation**, naming what the dataset has, and the selection is
  stored in channel order so identical requests produce identical records.
- **No plugin changes.** A method receives a flat sequence of records and returns one prediction
  per input; it simply receives fewer.

Rejected: the selection on the `Split`. A split decides *which samples* and exists to prevent
leakage; which views a model sees is the experiment's business, as preprocessing is. It would also
make "one channel" and "all channels" incomparable by construction, which is the comparison this
decision exists to enable.

## Consequences

- The channel is a first-class experiment variable: two runs on one split, one region build and one
  set of labels, differing only in what they read. A per-channel family of datasets can be merged
  back into one multi-channel dataset without loss.
- **A selection can silently shrink a run.** A valid name that no sample in the split carries yields
  a legal, empty run; the unknown-name check catches typos only.
- **The frozen name is only as stable as the channel dictionary.** Re-importing under a different
  canonicalization leaves old experiments readable but not re-runnable.
- **Displays must align images by channel name**, never by position, wherever runs with different
  selections are shown side by side.
- **It multiplies experiments** on a catalogue that has no notion of "the same thing, retried".
