# Roadmap

Where the workbench stands, and what is still open. Task-level detail is in
[backlog.md](backlog.md); how any of it works is in the [handbook](architecture/README.md); why it
is shaped that way is in the [decision records](adr/README.md).

Sizing is honest, not aspirational: one developer plus Claude Code on an Apple Silicon Mac, evenings
and weekends.

## What it does today

A dataset can be taken from a directory tree to a comparison between methods without leaving the
application.

- **Import** a directory tree through a pluggable adapter, or register a local copy of a public
  benchmark (VisA, GKN) in one atomic action. Source images are referenced in place and never
  copied; the import is idempotent and leaves a reviewable manifest.
- **Browse and label** the result as a catalogue that groups — a collection is a dataset's stored
  override or the reference pack it came from — with a virtualised grid, channel filters and an
  image-first sample viewer. A dataset names the channel it is read in, and every screen that has
  room for one photograph of a part opens on it.
- **Annotate** at pixel level: polygon and brush with editable contour tracing, undo/redo, autosave
  with conflict detection, and a keyboard queue. Truth is versioned and lives in the source frame.
  One annotation covers every channel of a part, while revisions stay per image.
- **Prepare** an invertible region profile — object detection, crop and resample pinned as an
  immutable revision an experiment can reference, so a run's spatial input is reproducible.
- **Split** a dataset at sample level, or adopt the split a benchmark published.
- **Train and score** through one plugin interface. Seven methods ship: `pixel_reference`
  (numpy + Pillow, the floor), `efficientad_custom`, `patchcore_anomalib`,
  `dinomaly_anomalib`, `dinomaly_custom`, `glass_anomalib` and `dino_memory` — a frozen
  DINOv2/DINOv3 patch memory that is a coreset bank, a per-position bank or a per-position
  Gaussian depending on one `scoring` field. It cleared the paired VisA gate and beat its
  PatchCore control on all three floor metrics ([measurements.md](measurements.md)).
  `dinomaly_custom` is the in-house Dinomaly, pinned bit-exact against the wrapper it is
  measured against, with the encoder and the decoder depth as fields the wrapper cannot
  offer. Jobs run as subprocesses with live progress, cancellation and a replayable log.
- **Select channels per experiment** by name, with per-channel score normalization before
  aggregation, so "how well does bright-field alone do?" is one run rather than a second dataset.
- **Read** image- and pixel-level metrics, browse every scored sample, filter to the model's
  mistakes, and ask a fitted method about any image on demand.
- **Compare** N runs of one split side by side, find the samples they disagree on, and open one with
  every method's map in its own pane. Nothing is compared in score units.
- **Export** a supported fitted method as a checksummed ONNX deployment bundle that passes
  Python-versus-portable parity and runs through a small Rust reference consumer.

## Open

- **The visual pass.** Key screens reviewed at 1440×900 and 1024×768 in both themes: hierarchy,
  density, contrast, focus, and loading/empty/error/disabled states. The last criterion of the
  deployment-and-onboarding milestone that is not met.
- **The large-catalogue experiment workflow** — id query, multi-select methods, date range, cursor
  pagination, sortable headers, compatible selection handed to Compare.
- **`dinomaly_custom`'s VisA parity gate.** The in-house implementation ships and is pinned
  bit-exact against `dinomaly_anomalib` on one training step, on the map and on the score, but
  it has not been measured on public data. Until it is, both are listed and the wrapper is the
  reference. On parity the wrapper retires the way `efficientad_anomalib` did, with an ADR-0008
  changelog entry recording it. `dinomaly_custom` also does not export ONNX yet, which the
  wrapper does.
- **Method evaluation that is still open**: AnomalyVFM as a zero-shot reference (its resource gate
  passed; plugin integration and the public quality gate remain). SuperADD is no longer on this
  list — the three things its evaluation was waiting on are what `dino_memory` now provides in-house
  (ADR-0037).
- The measurement and follow-up work each method left behind, in [backlog.md](backlog.md).

## Deliberately not built

`classical_circular` — a circle-fit, polar-transform baseline that would exploit the showcase
dataset's geometry. The universal goal is served by `pixel_reference` instead, which is the same
statistical core with no geometry assumption. It stays optional, and may never be built; its design
is sketched in [methods.md](architecture/methods.md).
