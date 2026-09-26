# ADR-0006: Import via pluggable adapters and a reviewable manifest

**Status:** Accepted (2026-08-06)

## Context

Real acquisition folders are not tidy. In the showcase dataset one channel appears under several
spellings, labels are folder names in some groups and absent in others, most parts have three
views and some two, one group uses timestamped filenames that do not group by stem, and numeric IDs
repeat across groups. Public benchmarks bring their own layouts: class folders, CSV tables,
official split files.

An importer that hard-codes one layout breaks on the next dataset. An importer that silently
guesses produces a corrupt dataset instead: a mis-grouped part is a labelling error that propagates
into every experiment run afterwards.

## Decision

**Import is two-phase — scan, then commit — mediated by named adapters and a reviewable manifest.**

1. **Scan.** A named adapter walks a dataset root and proposes a structure without touching the
   database. Adapters are registered by name, and each declares its options as a pydantic model
   whose JSON Schema drives the import form. A new layout is a new adapter, not an edit to the
   importer or the UI.
2. **Manifest.** The scan emits a manifest: proposed channels, samples, images and labels, plus
   **warnings**. Warnings are informative, not fatal — a two-view part is a warning, never an error
   (see ADR-0041) — and files that could not be grouped are surfaced individually.
3. **Review.** The operator inspects the proposal, corrects channel canonicalization, resolves
   ungrouped files and adjusts labels before anything is written. Channel canonicalization is an
   **editable mapping**, not code.
4. **Commit.** The accepted manifest creates the rows and is **persisted verbatim**, so the import
   is reproducible and auditable.

**Images are never copied** (see ADR-0022). A **sha256** of each file is recorded at import, and a
separate verify operation re-hashes later to detect moved, replaced or corrupted sources.

## Alternatives considered

- **One built-in layout.** Simple, and it breaks on the second dataset; every new convention becomes
  a branch in the importer.
- **Automatic inference with no review.** One click, and a wrong grouping is discovered only when an
  experiment's numbers stop making sense, if at all.

## Consequences

Messy acquisition folders are handled by data — a mapping and a manifest — rather than code
branches, and a new convention costs one adapter. "How did this dataset come to look like this?" is
answerable later from the stored manifest. The hash separates "the file changed" from "the model
changed" when results stop reproducing.

- **Import is not one click.** Every dataset gets a review pass. For a regular dataset that is
  ceremony, and clicking through without reading turns it into false assurance.
- **The manifest is a third representation.** Folder layout, manifest and database can all
  disagree; the manifest records what was proposed and accepted, not what the database holds now.
- **Fuzzy channel matching can be confidently wrong.** Two different channels with similar names
  can be merged; the editable mapping helps only if someone notices.
- **Reference in place is fragile to filesystem moves.** Renaming a folder breaks every path; verify
  detects the damage and does not repair it.
- **Hashing reads every byte** at import and again at each verify, which on a large dataset is
  minutes, not seconds.
