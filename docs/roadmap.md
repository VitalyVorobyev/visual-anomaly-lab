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
- **Train and score** through one plugin interface. Seven anomaly methods ship: `pixel_reference`
  (numpy + Pillow, the floor), `efficientad_custom`, `patchcore_anomalib`,
  `dinomaly_custom`, `glass_anomalib`, `dino_memory` and `subspace_ad`. `dino_memory` is a frozen
  DINOv2/DINOv3 patch memory that is a coreset bank, a per-position bank or a per-position
  Gaussian depending on one `scoring` field. It cleared the paired VisA gate and beat its
  PatchCore control on all three floor metrics ([measurements.md](measurements.md)).
  `subspace_ad` keeps what the normal patches *span* rather than the patches themselves — a PCA
  over the same frozen encoders, with no training step at all — and is the first method whose
  defaults were chosen by a measured sweep rather than picked (ADR-0038).
  `dinomaly_custom` is the in-house Dinomaly, with the encoder and the decoder depth as
  fields the anomalib wrapper it was measured against could not offer; that wrapper reached
  VisA parity and retired ([measurements.md](measurements.md)). Jobs run as subprocesses with
  live progress, cancellation and a replayable log.
- **Select channels per experiment** by name, with per-channel score normalization before
  aggregation, so "how well does bright-field alone do?" is one run rather than a second dataset.
- **Read** image- and pixel-level metrics, browse every scored sample, filter to the model's
  mistakes, and ask a fitted method about any image on demand.
- **Compare** N runs of one split side by side, find the samples they disagree on, and open one with
  every method's map in its own pane. Nothing is compared in score units.
- **Export** a supported fitted method as a checksummed ONNX deployment bundle that passes
  Python-versus-portable parity and runs through a small Rust reference consumer.

## Open

- **The visual pass, its states half.** Every screen has been reviewed at rest at 1440×900 and
  1024×768 in both themes with the `lab-visual-pass` skill (2026-09-24): hierarchy, density and
  contrast hold, no screen nests a scroller, and the one structural finding left is the five
  buttons inside links, waiting on lab-ui's `ButtonLink`. What has not been looked at is the
  transient half — loading, error and disabled states, and keyboard focus — which the screenshot
  script does not drive yet.
- **The large-catalogue experiment workflow** — id query, multi-select methods, date range, cursor
  pagination, sortable headers.
- **`dinomaly_custom` does not export ONNX yet**, which the retired anomalib wrapper did — that
  follow-up is in [backlog.md](backlog.md).
- **Method evaluation that is still open**: AnomalyVFM as a zero-shot reference (its resource gate
  passed; plugin integration and the public quality gate remain). SuperADD is no longer on this
  list — the three things its evaluation was waiting on are what `dino_memory` now provides in-house
  (ADR-0037).
- The measurement and follow-up work each method left behind, in [backlog.md](backlog.md).
- **Few-shot segmentation as a peer task** (ADR-0040). One to ten references of a class define it,
  and frozen DINOv3 feature matching segments it in new images, with absence as a first-class
  answer. Built so far: the task value, the target class on the experiment, the `manual` and `few_shot`
  reference splits, per-class coverage, class-index masks with a pinned class table, references reaching
  `fit` as targets, predicted masks, an evaluator for masks and presence, and `color_prototype`, the
  torch-free floor that runs the slice end to end, and the shared frozen-DINO blocks the next methods
  build on (one encoding path, INSID3's positional debiasing, guided refinement), and `fss_dino`, a
  reproduction of the FSSDINO baseline on them, and `proto_seg`, ours, whose debiasing, prototype bank,
  adaptation and refinement are each a field. The dataset workspace is grouped as Data · Truth · Runs,
  its band says per task what is still needed, the create screen asks for the task first, and the
  Splits tab draws references for a class. A segmentation run's results are read in its own terms —
  overlap, per-sample outcomes (hit, low IoU, miss, false presence), and a labelled headline — and Compare
  reads runs of one class across reference draws. Still open:
  - the public gate that decides between the methods;
  - the reference studio's live preview and its accept / fix / mark-absent loop (choosing references
    by eye and freezing them into a run is built);

  The order is in [backlog.md](backlog.md).
- **Supervised segmentation and detection** (ADR-0039) follow it, and reuse its seams.

## Deliberately not built

`classical_circular` — a circle-fit, polar-transform baseline that would exploit the showcase
dataset's geometry. The universal goal is served by `pixel_reference` instead, which is the same
statistical core with no geometry assumption. It stays optional, and may never be built; its design
is sketched in [methods.md](architecture/methods.md).
