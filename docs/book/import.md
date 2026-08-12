# Import and registration

Import is deliberately two-phase: **scan proposes; commit persists**. Source trees are read-only.

## Public packs

The catalogue recognizes complete local VisA and GKN packs under `/datasets/` and offers one registration
action per dataset/class. Registration still uses ordinary adapters; the pack catalogue only supplies known,
credited options. This avoids a parallel “special benchmark” data model.

VisA uses `csv_table`, preserves official one-class assignments and mask paths, and registers each object
class as a dataset. GKN uses `folder_classes`, with `Good` normal and `Nick`/`Scratch` defective. These are
provider mappings, not global label conventions.

## General scan

The frontend requests an adapter catalogue. Each adapter exposes a Pydantic options model; its JSON Schema
generates the form. A scan job walks the selected root, probes decodable images, and writes a reviewable
manifest containing dataset metadata, channels, samples, images, labels, masks, imported subset hints, and
warnings.

Review at least:

- total samples versus total images;
- channel names and images per sample;
- counts of normal, defect, and unknown labels;
- duplicate identities and missing views;
- unreadable files or masks;
- imported subset counts.

Commit validates the manifest again, creates catalogue rows in one transaction, records adapter provenance,
and schedules media preparation. It indexes source paths rather than copying bytes.

## Rescan, verify, and delete

The committed manifest is immutable provenance. `verify` checks that referenced files still exist and match
the recorded facts. A changed source tree is not silently absorbed; rescan creates a new proposal and commit
applies explicit reconciliation semantics described in the [import handbook](../architecture/import.md).

Deleting a dataset removes catalogue metadata and derived workspace artifacts after dependency confirmation.
It never deletes the external source tree. Experiments tied to the dataset must be removed or explicitly
cascaded through the UI.

## Failure semantics

- Cancellation stops scanning and publishes no dataset.
- An unreadable optional file becomes a manifest warning; a structurally invalid manifest cannot commit.
- Paths are server-local absolute paths. Browser uploads are not the import model.
- One corrupt image must be named. It must not become an unexplained lower sample count.

To support a new layout, see [Add an import adapter](add-adapter.md).
