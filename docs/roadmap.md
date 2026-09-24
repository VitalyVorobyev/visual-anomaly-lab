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
- **Annotate** at pixel level: polygon, box and brush with editable contour tracing, instance ids,
  class keys, undo/redo, autosave with conflict detection, and a keyboard queue. Completion records
  the object instances beside the masks. Truth is versioned and lives in the source frame.
  One annotation covers every channel of a part, while revisions stay per image.
- **Prepare** an invertible region profile — object detection, crop and resample pinned as an
  immutable revision an experiment can reference, so a run's spatial input is reproducible. On a
  grouped dataset the channels of one part can share one union crop, so they stay registered.
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

- **The visual pass's last two findings, both upstream in lab-ui.** Every screen has been reviewed at rest
  in both viewports and themes, and in its transient states — pending, error, a Tab walk and disabled
  controls (`lab-visual-pass --states`). Failed reads now show their error promptly, and every screen
  says what went wrong. What is left: five buttons nest inside links, waiting on a `ButtonLink`, and two
  disabled tabs explain themselves only in a tooltip, waiting on `Tabs` (see [backlog.md](backlog.md)).
- **`dinomaly_custom` does not export ONNX yet**, which the retired anomalib wrapper did — that
  follow-up is in [backlog.md](backlog.md).
- **Method evaluation that is still open**: AnomalyVFM as a zero-shot reference (its resource gate
  passed; plugin integration and the public quality gate remain). SuperADD is no longer on this
  list — the three things its evaluation was waiting on are what `dino_memory` now provides in-house
  (ADR-0037).
- The measurement and follow-up work each method left behind, in [backlog.md](backlog.md).
- **Few-shot segmentation as a peer task** (ADR-0040). One to ten references of a class define it, and a
  run segments that class in every other sample, with absence as a first-class answer. What stands:
  - **Truth:** class-index masks with a pinned class table, per-class coverage, and references as a split
    (`manual`, or a seeded `few_shot` draw).
  - **Methods:** `color_prototype` (the torch-free floor), `fss_dino` (a reproduction of FSSDINO) and
    `proto_seg` (ours), on one frozen-DINO encoding path with INSID3's positional debiasing and guided
    refinement.
  - **Reading:** an evaluator for masks and presence, result screens in the task's own terms, and Compare
    across reference draws.
  - **Workflow:** the dataset workspace as Data · Truth · Runs with per-task readiness, a task-first create
    screen, and the reference studio — choose references by eye, preview any image live, accept / fix /
    mark absent, and freeze into a run.

  Still open: the public gate that decides between the methods.
- **Supervised segmentation and detection** (ADR-0039) follow it, and reuse its seams.

## Deliberately not built

`classical_circular` — a circle-fit, polar-transform baseline that would exploit the showcase
dataset's geometry. The universal goal is served by `pixel_reference` instead, which is the same
statistical core with no geometry assumption. It stays optional, and may never be built; its design
is sketched in [methods.md](architecture/methods.md).
