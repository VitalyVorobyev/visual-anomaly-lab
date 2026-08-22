# ADR-0006: Import via pluggable adapters and a reviewable manifest

**Status:** Accepted (2026-08-06)

## Context

The reference dataset is not tidy. Channel folders appear as `Bright`, `BrightField`, and
`Brightfield`; `Dark`/`DarkField`/`Darkfield`; `Dome`/`DomeIllumination`. Labels are encoded as
folder names in `set1`/`set2` but absent in `unsorted/`. Most groups have three channels, one has
two, and another uses machine-generated timestamped filenames that do not group by stem at all.
Numeric IDs repeat across groups.

Any import routine that hard-codes this layout will break on the next dataset — and the brief
forbids assuming the acquisition setup generalizes. At the same time, a fully automatic importer
that silently guesses would quietly produce a corrupt dataset: a mis-grouped part is a labelling
error that propagates into every experiment run afterwards.

## Decision

**Import is a two-phase operation — scan, then commit — mediated by an editable adapter and a
reviewable manifest.**

1. **Scan.** A named adapter walks a dataset root and proposes a structure without touching the
   database. The first adapter, `channel_folders`, recognizes label folders (`defect` /
   `no-defect`), canonicalizes channel folder names through an **editable mapping** (fuzzy match to
   a canonical set, with unknown names surfaced rather than dropped), and groups images by filename
   stem **within a capture group**. Adapters are registered by name; adding a layout means adding
   an adapter, not editing the importer.
2. **Manifest.** The scan emits a JSON manifest: proposed datasets, channels, samples, images,
   labels, plus a **warnings** list. Warnings are informative, not fatal — a sample with two
   channels is a *warning*, never an error (ADR-0005), and files that could not be grouped (the
   timestamped ones) are surfaced individually so the operator can decide.
3. **Review.** The manifest is presented in the UI. The operator inspects the proposal, corrects
   channel canonicalization, resolves ungrouped files, and adjusts labels before anything is
   written.
4. **Commit.** The accepted manifest creates the database rows and is **persisted verbatim** to
   `data/manifests/` so the import is reproducible and auditable after the fact.

**Images are never copied** (ADR-0022). At import, a **sha256** of each file is recorded; a separate
verify operation re-hashes the referenced files later to detect moved, replaced, or corrupted
source images.

## Consequences

The messiness of real acquisition folders is handled by data (an editable mapping, a reviewable
manifest) rather than by code branches. A new dataset with different conventions costs one adapter.
Because the committed manifest is stored, "how did this dataset come to look like this?" is
answerable months later, and a re-import can be replayed. Hashing gives us an integrity check that
distinguishes "the file changed" from "the model changed" when results stop reproducing.

Negative consequences, accepted honestly:

- **Import is no longer one click.** Every dataset requires a human review pass. For a fully regular
  dataset this is pure ceremony, and it will be tempting to click through without reading — at which
  point the review step provides false assurance rather than safety.
- **The manifest is a third representation.** Folder layout, manifest JSON, and database rows can
  all disagree after the fact. The manifest records what was *proposed and accepted*, not what the
  database currently holds; edits made after import are invisible to it.
- **Fuzzy channel matching can be confidently wrong.** Two genuinely different channels with similar
  names could be merged. The editable mapping is the escape hatch, but only if someone notices.
- **Referencing in place makes the DB fragile to filesystem moves.** Renaming a folder breaks every
  image path; the sha256 verify detects the damage but does not repair it, and no re-link tool is
  planned initially.
- **Hashing 3.2 GB costs time.** A full import reads every byte of the dataset — minutes, not
  seconds — and repeats that cost on each verify.
- **Adapter registry adds indirection** for what is, today, exactly one adapter.
