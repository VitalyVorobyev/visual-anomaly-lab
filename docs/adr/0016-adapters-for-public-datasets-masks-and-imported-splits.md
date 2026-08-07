# ADR-0016: Adapters for public datasets, masks in the catalog, imported splits

**Status:** Accepted (2026-08-07)

**Extends:** ADR-0006 (import via pluggable adapters), ADR-0011 (splits are seeded and
sample-level).

## Context

ADR-0015 put public benchmarks on the critical path, and neither of the two on hand can be
read by `channel_folders`. GKN's `Nick` and `Scratch` directories match no label vocabulary,
so it imports 197 defective parts as unlabelled. VisA's structure is described by a table
rather than by its directory names, and that table carries the **official train/test split**
the published numbers were computed on.

Two further gaps surfaced while measuring rather than while planning:

- **The mask table was empty by design.** Schema v1 defined `mask` and the comment beside it
  said so outright: "deliberately unpopulated: the reference dataset has no masks". VisA has
  100 per class.
- **The schema-driven options form did not exist.** ADR-0006 and `system-design.md` §5 both
  state that an adapter's options model "drives the UI form". The import screen hardcoded
  exactly one option — `exclude` — and relied on the adapter's Python defaults for
  everything else. That was survivable while one adapter existed whose defaults happened to
  fit the one dataset. It stops being survivable the moment an adapter has a *required*
  option, because nothing in the UI can supply it.

Adding a vocabulary wide enough to cover `Nick` and `Scratch` was considered and rejected:
guessing more aggressively makes the guesses worse, and ADR-0006 already accepts that fuzzy
matching can be confidently wrong.

## Decision

**Two adapters, masks as first-class catalog rows, and a split strategy that adopts a
published partition instead of drawing one.**

- **`folder_classes`** — the simple contract: name the directories holding defect-free
  images and the ones holding defective images. One image becomes one sample with **no
  channel**, which is also the first thing to exercise the single-view path ADR-0005 has
  allowed since day one. Directory patterns cover the subtree beneath them, because "point
  at the folder" has to mean the folder and what is under it. A file matching no named
  directory imports unlabelled and is reported; a file matching two is read as defective,
  which is the safe reading — a defect in the training set teaches the model that defects
  are normal.
- **`csv_table`** — every column name is an option, as are the values meaning normal,
  defective, and each subset. `filter_column`/`filter_value` turn one table covering a
  benchmark family into one dataset per class, which is the one-class protocol those
  benchmarks are scored under.
- **Masks enter the catalog.** `ManifestImage` gains `mask_path`; commit writes `mask` rows
  keyed by `(image_id, kind)`; `verify` walks them. A mask the manifest no longer mentions
  is **left alone**, for the same reason a missing image is reported rather than deleted.
- **`SplitStrategy.IMPORTED`.** A split can be materialized from the subsets recorded in the
  manifest the dataset was committed from, with `manifest_id` stored in the split's params.
  No seed, no fractions, no stratification.
- **The options form is generated from each adapter's JSON Schema**, as originally
  specified. Control type follows the schema node's type; a field whose default is *empty*
  is shown, and a field with a working default is folded behind a disclosure.
- **`ChannelFoldersOptions.channels` now defaults to empty.** A shipped vocabulary made one
  acquisition setup's illumination names part of the application, which is what ADR-0005
  forbids. The showcase test supplies its own.

## Consequences

Adding support for a public dataset costs configuration rather than code, and the two
adapters between them cover GKN, VisA, MVTec-AD, BTAD and most single-view public sets. The
single-view path and the mask table both stop being untested claims. An imported split makes
a comparison against a published figure meaningful, because it is computed on the same
partition.

Negative consequences, accepted honestly:

- **`verify` cannot detect mask drift.** Schema v1 has no `mask.sha256` and the schema is
  frozen (ADR-0004), so a mask re-exported in place is invisible. The report counts masks
  separately from images so it never implies a check it did not make; lifting this is a
  migration.
- **An imported split trusts the source completely.** Nothing verifies that a published
  train subset contains no defects, or that the partition is sane. If the table is wrong,
  the split is wrong in exactly the same way — which is arguably the point, but it means a
  bad benchmark file produces a bad experiment silently.
- **Sample identity in `folder_classes` is the filename stem**, so two files in one
  directory differing only in extension merge into one sample. Warned about, not prevented.
- **`{dir}/../../ground_truth/{class}` is a template language.** The mask options grew
  placeholders and `..` traversal to reach MVTec-shaped trees. It is a small language with
  no validation: a bad placeholder yields zero masks and a warning, not an error naming the
  mistake.
- **The generated form cannot express everything the model accepts.** A `dict` field gets a
  JSON textarea, and an optional field with a non-empty default has no way to be set back to
  null through the UI. Both are reachable through the API.
- **A default now exists in one place and is shown in another.** The form sends nothing for
  an untouched control, so the backend's default stays authoritative — but the placeholder
  showing it is a copy that is only correct because it came from the same schema in the same
  response.
