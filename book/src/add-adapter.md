# Add an import adapter

An adapter interprets a source layout into the common manifest. It does not write the database, copy images,
or know the frontend.

## Contract

Implement `ImportAdapter` under `backend/src/anomaly_lab/datasets/adapters/`:

- stable `name` and one-line `summary`;
- a Pydantic options model with descriptions;
- `scan(root, options, dataset_name, progress) -> Manifest`;
- deterministic sample identity, labels, channels, masks, and imported subsets;
- progress/cancellation through the callback.

Register it in `datasets/adapters/__init__.py`. Its schema-generated import form appears automatically.

## Design sample identity first

The adapter's most consequential decision is which files form one sample. Prefer explicit table IDs. For
filenames, define parsing and collision behavior. Never infer channel count into schema or assume missing
channels make a sample invalid unless the source protocol says so.

## Reuse support helpers

Use the shared extension filter, image probe, warning collection, and manifest builder. A corrupt file,
duplicate identity, missing mask, or inconsistent dimensions should produce the same warning shape across
adapters. Keep only layout-specific parsing in the new module.

## Tests

Generate tiny synthetic PNGs and CSVs. Cover:

- stable order independent of filesystem enumeration;
- one and multiple channels;
- normal, defect, and unknown labels;
- imported train/test values;
- masks and absent masks;
- duplicates and unreadable files;
- cancellation;
- JSON Schema reaching the generic UI control mapping;
- scan → manifest serialization → commit → verify.

Never use a real dataset image as a fixture. Public-pack detection is separate: add a pack entry only when a
stable, credited layout benefits users, and implement it as known options passed to the ordinary adapter.
