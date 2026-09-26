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
  benchmark (VisA, GKN, FSS-1000, PKU-Market-PCB) in one atomic action. Source images are referenced in place and never
  copied; the import is idempotent and leaves a reviewable manifest.
- **Browse and label** the result as a catalogue that groups — a collection is a dataset's stored
  override or the reference pack it came from — with a virtualised grid, channel filters and an
  image-first sample viewer. A dataset names the channel it is read in, and every screen that has
  room for one photograph of a part opens on it. Truth is task-scoped (ADR-0041): a sample's
  normal/defect label is anomaly truth and a class lives in annotations, so a dataset of classes —
  FSS-1000's panel, PKU-Market-PCB — is one dataset counted, filtered and covered by its classes, with
  no verdicts it never asserted, and offered anomaly detection only once a sample carries one.
- **Annotate** at pixel level: polygon, box and brush with editable contour tracing, instance ids,
  class keys, undo/redo, autosave with conflict detection, and a keyboard queue. Completion records
  the object instances beside the masks. Truth is versioned and lives in the source frame.
  One annotation covers every channel of a part, while revisions stay per image.
- **Prepare** an invertible region profile — object detection, crop and resample pinned as an
  immutable revision an experiment can reference, so a run's spatial input is reproducible. On a
  grouped dataset the channels of one part can share one union crop, so they stay registered.
  MobileSAM can reject masks that wrap the frame border and unite the rest; that rule is opt-in, since
  on held-out public classes it localised the part but kept 0.92 of defect pixels, below the
  predeclared 0.98 ([measurements.md](measurements.md)).
- **Split** a dataset at sample level, or adopt the split a benchmark published — from a preset
  card per task with its dry-run composition and one press to create it, or by hand under **Custom
  split** with a live preview. A split no experiment ran on can be deleted.
- **Train and score** through one plugin interface. Eight anomaly methods ship: `pixel_reference`
  (numpy + Pillow, the floor), `efficientad_custom`, `patchcore_anomalib`,
  `dinomaly_custom`, `glass_anomalib`, `dino_memory`, `subspace_ad` and `anomalyvfm_anomalib`. `dino_memory` is a frozen
  DINOv2/DINOv3 patch memory that is a coreset bank, a per-position bank or a per-position
  Gaussian depending on one `scoring` field. It cleared the paired VisA gate and beat its
  PatchCore control on all three floor metrics ([measurements.md](measurements.md)).
  `subspace_ad` keeps what the normal patches *span* rather than the patches themselves — a PCA
  over the same frozen encoders, with no training step at all — and is the first method whose
  defaults were chosen by a measured sweep rather than picked (ADR-0038).
  `anomalyvfm_anomalib` is the zero-shot reference: AnomalyVFM's published checkpoint, verified by
  digest and built offline, scores a run with no train job and reads no normal image. It cleared its
  public gate's floors on VisA at 768 px ([measurements.md](measurements.md)).
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
- The measurement and follow-up work each method left behind, in [backlog.md](backlog.md).
- **Few-shot segmentation as a peer task** (ADR-0040). One to ten references of a class define it, and a
  run segments that class in every other sample, with absence as a first-class answer. What stands:
  - **Truth:** class-index masks with a pinned class table, per-class coverage, and references as a split
    (`manual`, or a seeded `few_shot` draw).
  - **Methods:** `color_prototype` (the torch-free floor), `fss_dino` (a reproduction of FSSDINO) and
    `proto_seg` (ours), on one frozen-DINO encoding path with INSID3's positional debiasing and guided
    refinement.
  - **Reading:** an evaluator for masks and presence, with the probability map's pixel AP read
    threshold-free beside the cut's IoU, result screens in the task's own terms, and Compare
    across reference draws.
  - **Workflow:** the dataset workspace as Data · Truth · Runs with per-task readiness, a task-first create
    screen, and the reference studio — choose references by eye, preview any image live, accept / fix /
    mark absent, and freeze into a run.

  The public VisA gate made `proto_seg` the default, and showed that no method yet draws a usable mask of
  a small defect at the fixed `>= 0.5` cut ([measurements.md](measurements.md)). Every few-shot method
  can scale its probability on its own references (`calibration` `leave_one_out`); its gate leg kept
  the unscaled default, because the scaled cut stops flagging absent images but finds too few of the
  present ones. On FSS-1000's object classes `proto_seg` ranks pixels and images almost perfectly (pixel AP
  0.81, presence ROC-AUC 1.000) while `fss_dino`, which assumes the class is present, ranks with the floor;
  at the fixed cut `proto_seg` still marks nearly every absent image ([measurements.md](measurements.md)).
  Still open: a mask that honours the run's own presence score.
- **Supervised segmentation** (ADR-0039) runs end to end on the few-shot task's seams: a run pins its
  dataset's classes at creation, fits on the annotated images of its train subset through
  `label_targets`, writes an 8-bit label map per image, and is read by a per-class confusion matrix
  (mean IoU, per-class IoU and accuracy, pixel accuracy, frequency-weighted IoU). `color_classifier` is
  the torch-free floor, `dino_linear_seg` the first deep method (a softmax head on frozen DINO patch
  features, trained on a bounded sample of annotated pixels), and `class_stratified` draws a split for
  them from annotated samples, stratified by the classes each shows. Its results read in its own terms: IoU per class across subsets, the
  confusion matrix drawn, a per-sample verdict (`false_class` beside the few-shot outcomes), the label
  maps over the image on the sample page and on every gallery tile — prediction solid, truth dashed, one
  palette colour per pinned class — and the
  task in the dataset's readiness band. `dino_linear_seg` is supported: sampling each class in its own
  share and adding one constant per class, fitted for IoU on held-out folds of the training images
  (`logit_bias` `held_out_iou`), it beats the floor by the predeclared margin on both VisA classes, where
  the head's own argmax labelled a few percent of every image defect ([measurements.md](measurements.md)).
  Its mask of a small VisA defect is usable, not good.
- **Detection** (ADR-0039) runs end to end on the same seams, torch-free: a run pins its classes, fits
  through `box_targets` on the boxes of its annotated training images — a revision's instances, or an
  imported mask's connected components — writes boxes per image, and is read by COCO's protocol
  (AP@[.5:.95] as the headline, AP50, AP75, recall, AP per class). `color_detector` is the floor, the
  colour classifier's components boxed; `dino_linear_det` is the first deep detector, `dino_linear_seg`'s
  head fitted on painted box interiors and decoded by the floor's components (`dl` extra). It is offered segmentation's splits, `class_stratified` and
  `manual`, and named in the dataset's readiness band. Its results read in its own terms: AP per class
  across subsets, a per-sample verdict and drawn boxes — truth dashed, predictions solid, toned by match,
  false positive or miss — at a confidence cut each subset resolves by one printed rule (the F1-optimal
  confidence at IoU 0.5), and Compare on the AP family alone. `dino_linear_det` is experimental: on the
  public detection gate neither it nor the floor boxes a VisA defect — AP@[.5:.95] below 0.01 on both
  classes, where many of the boxes VisA's masks give are specks of a few pixels. On PKU-Market-PCB, whose
  truth is drawn as boxes, it finds defects the floor does not (AP50 0.105 against 0.0001) but places the
  box loosely (AP@[.5:.95] 0.027), so a box-regression head is next ([measurements.md](measurements.md)).

## Deliberately not built

`classical_circular` — a circle-fit, polar-transform baseline that would exploit the showcase
dataset's geometry. The universal goal is served by `pixel_reference` instead, which is the same
statistical core with no geometry assumption. It stays optional, and may never be built; its design
is sketched in [methods.md](architecture/methods.md).
