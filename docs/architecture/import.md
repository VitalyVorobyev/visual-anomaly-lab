# Dataset import

Import is **two-stage: adapter → reviewable manifest → commit** (ADR-0006). Directory layouts in the wild are
irregular, and a one-shot importer that guesses silently produces a corrupt catalog that is discovered only
much later.

```mermaid
sequenceDiagram
    participant UI
    participant API as FastAPI
    participant AD as Import adapter
    participant FS as Source images
    participant DB as SQLite

    UI->>API: POST /api/import/scan {root_path, adapter, options}
    API->>AD: run adapter (as a job)
    AD->>FS: walk tree, group, hash
    AD-->>API: manifest {samples, channel_mapping, warnings}
    API-->>UI: manifest (nothing written to DB yet)
    UI->>UI: review — fix channel mapping, labels, drop entries
    UI->>API: POST /api/import/commit {edited manifest}
    API->>DB: insert Dataset, Channels, Samples, Images
    API->>FS: save manifest to data/manifests/
    API-->>UI: dataset_id
```

## Stage 1 — `POST /api/import/scan`

Runs a **pluggable adapter** against a root path. The first adapter, `channel_folders`, encodes the reference
layout:

- **Label detection** from folder names, tolerating the variants that occur in practice (`defect`, `Defect`,
  `no-defect`, `no_defect`, `normal`, `ok`); anything unmatched yields `unlabeled` rather than a guess.
- **Channel canonicalization** by fuzzy matching folder names to canonical keys — `Bright` / `BrightField` /
  `Brightfield` → `bright`, `Dark` / `DarkField` / `Darkfield` → `dark`, `Dome` / `DomeIllumination` →
  `dome`. The proposed mapping is **part of the manifest and editable in the UI**: the fuzzy matcher is a
  convenience, not an authority, and a new dataset with unfamiliar channel names must be importable without a
  code change.
- **Grouping by filename stem** within a group folder: the same stem across channel folders is the same
  physical part. Group + stem form `(group_key, external_id)`, which is what makes numeric IDs that repeat
  across groups safe. The **group key keeps the label component** — the same stem exists under both a
  defect and a no-defect folder, so dropping it would collide two different parts onto one identity.

**Matching is by component, not by position.** The adapter does not know whether label folders sit above
channel folders or below them: each path component is tested against the label vocabulary, then the channel
vocabulary, and whatever is left becomes the group key. Prefix matching applies to **tokens** rather than
whole components, because normalization strips separators — a directory named `"<Channel> <Group>"`
normalizes to a string that *begins with* the channel name, and matching it whole would swallow the group
name and silently merge that group into its parent. That case is not hypothetical; it is how one real
capture group is laid out (ADR-0013).

The adapter emits a **manifest JSON**: proposed samples `{group_key, label, images: [{path, channel}]}`, the
channel mapping, and a `warnings` list. Warnings are deliberately non-fatal:

- a sample with a channel count different from its siblings (e.g. the two-channel group) is a **warning,
  not an error** — variable channel counts are legitimate data, and the importer must never enforce a fixed
  count;
- images that matched no channel in a dataset that *has* channels are **surfaced for review**, together
  with the directory names that were not recognized, so the operator can add a mapping rather than
  discover a mis-import later. (The reference tree's machine-generated timestamped filenames were assumed
  not to group; measurement shows they group perfectly — see ADR-0013. The path exists for datasets that
  genuinely do not.)
- unreadable files, zero-byte files and duplicate hashes are reported with their paths.

## Stage 2 — `POST /api/import/commit`

Takes the (possibly edited) manifest, creates or updates `Dataset`, `Channel`, `Sample` and `Image` rows in
one transaction, and **saves the committed manifest to `data/manifests/`**. It is **synchronous, not a
job**: the walk and the hash already happened during the scan, so this is a few hundred inserts and
measures in milliseconds. It is also idempotent, never downgrades a hand-made label, and reports rather
than deletes a recorded file the manifest no longer mentions (ADR-0013). The stored manifest is the
reproducibility record: it states exactly which files became which samples under which channel mapping, and
re-importing the same tree can be diffed against it.

## Invariants

- **Images are never copied.** Only absolute paths are stored (ADR-0001). The source tree stays read-only.
- **`sha256` is recorded** for every image at scan time.
- **`POST /api/import/verify`** re-checks existence and hashes as a job, reporting missing, modified or
  unreadable files. It detects drift and never repairs it. This is what keeps a reference-in-place catalog
  trustworthy over months.

## Proving the abstraction

Two further adapters ship, and between them they cover the public benchmarks (ADR-0016). Both produce **one
image per sample with `channel_id = NULL`**, which is what finally demonstrated that the domain model handles
single-view datasets and that nothing downstream assumes grouping — a claim the design made from the start and
nothing exercised until then.

- **`folder_classes`** — the simple contract: name the directories holding defect-free images
  (`normal_dirs`) and defective ones (`defect_dirs`), as globs relative to the root, each covering the subtree
  beneath it. Optional `mask_dir` / `mask_pattern` templates locate ground truth, including in a sibling
  directory. The matched directory's name is recorded on the sample, so a per-defect-type breakdown needs no
  schema that enumerates defect types. Nothing is guessed: a file in a directory no option names imports
  unlabelled **and is reported**.
- **`csv_table`** — reads a table the dataset ships. Every column name is an option, as are the values meaning
  normal, defective, and each subset. `filter_column` / `filter_value` turn one table covering a benchmark
  family into one dataset per class, which is the one-class protocol those benchmarks are scored under. Set
  `channel_column` and rows sharing a sample identity become one multi-channel sample — the same adapter,
  no special case, because channel count is data.

`csv_table` also carries the source's **published partition** through into the manifest, which is what
`SplitStrategy.IMPORTED` materializes ([the evaluation layer](evaluation.md)). Note that an official one-class protocol generally has train and
test and **no `val` subset at all**, so an empty validation set is ordinary rather than a broken split.

## The options form

An adapter's options model is a pydantic model, and its **JSON Schema drives the import form**: control type
follows the schema node's type, descriptions become help text, and defaults become placeholders rather than
pre-filled values — so an untouched control sends nothing and the backend's default stays the only definition
of it. A field whose default is *empty* is shown; a field that already has a working answer is folded behind a
disclosure, which is what keeps "where are the good images" from being the tenth question on the screen.

This was specified from the beginning and **built late**: until then the import screen hardcoded a single
option and relied on Python defaults for the rest, which was survivable only while one adapter existed whose
defaults fitted the one dataset on hand. `csv_table` has a required option, and nothing in the UI could supply
it. ADR-0016 records the gap.

---

