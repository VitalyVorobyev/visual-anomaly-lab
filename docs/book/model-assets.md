# Model assets and offline replay

Some methods are code-only. Others need pretrained teachers, backbones, or prompt-segmentation weights.
Visual Anomaly Lab treats those files as **versioned model assets**, not as incidental downloads hidden
inside a training function.

## The catalogue

Every known asset has a stable key and records:

- its purpose and the method that consumes it;
- licence and upstream provenance;
- expected byte size and SHA-256 digest;
- the filename and app-owned destination;
- whether it is installed, missing, corrupt, or supplied from an external path.

Opening the method catalogue is cheap: importing a plugin does not fetch weights or eagerly import its deep
runtime. The experiment form can therefore explain a missing prerequisite before a long run starts.

## Install or reference

There are two ownership modes.

**Install** downloads to a partial file through a cancellable job, verifies size and SHA-256, then renames it
atomically. A failed or cancelled download never appears as an installed asset. The app may delete an asset
it installed.

**External source** verifies a file already on disk and records its absolute path without copying it. The app
does not own that file and will only remove the reference, never the source. This is useful for a shared model
cache or an air-gapped transfer.

## Reproducibility contract

An experiment stores the effective asset identity that shaped its fitted model. Replaying it requires the
same bytes, not merely a file with the same display name. Asset checks happen before expensive work begins,
and an offline run must not fall back to a network download.

Portable deployment is a separate boundary. An ONNX bundle contains the tensors needed by its exported
graph and checksums every file, so the Rust target does not need the workbench's training asset catalogue.
See [Portable ONNX deployment](deployment.md).

## Failure semantics

- A missing asset blocks the dependent method with an actionable prerequisite.
- A hash or size mismatch is corruption or an unapproved version; it is never accepted as “close enough.”
- A cancelled download leaves only a removable partial file.
- Removing an app-owned asset does not delete experiments or already exported bundles, but retraining may
  require reinstalling it.
- Removing an external reference never deletes the referenced file.

For contributor details, start with the current [method architecture](../architecture/methods.md) and the
[model-asset API](../architecture/README.md#component-responsibilities).
