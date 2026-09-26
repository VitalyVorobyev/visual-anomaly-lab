# Backlog

Open work only. Anything shipped is described in the [handbook](architecture/README.md), not here.
[roadmap.md](roadmap.md) says where the workbench stands; this says what to do next.

**Sizes.** `S` ≈ half a day. `M` ≈ one focused day. `L` = multi-day, and should be split before it
is started rather than after. Sizes include reading the generated code properly — a task is not done
until its output has been reviewed.

## Interface

- [ ] **A disabled tab says why on screen, upstream in lab-ui** (S): the experiment page's
      Architecture and Inspector tabs, when a run recorded nothing for them, explain themselves only
      in a `title` tooltip — the one reason `lab-visual-pass --states` still reports. `Tabs` would
      show the reason on focus and hover through its own `Tooltip`; release, then drop the `title`s.
      The editor's disabled Discard has the same shape.
- [ ] **A sortable column header, upstream in lab-ui** (S): the experiment catalogue orders by
      header buttons, but `Table` cannot set `aria-sort` on the `<th>`, so a screen reader hears a
      button rather than a sorted column. Add a `sort` prop to `Column`, release, and move the
      catalogue's `SortHeader` onto it.
- [ ] **A segmented control that fits five choices, upstream in lab-ui** (S): Explore's mode picker
      has five segments, one more than the sample viewer's 288-px rail holds on a line, so it wraps
      and Text sits on a row of its own. `SegmentedControl` could share its width across the
      segments (or tighten their padding) when they would not fit; release, then drop the
      `flex-wrap` in `routes/sample/ExploreSection.tsx`.
- [ ] **A thumbnail that shows its truth at thumbnail size** (S): the guided run's strips lay the
      source-sized outline from `GET /api/images/{id}/mask` over each thumbnail, and at 112 px a VisA
      defect is a speck and a PKU-Market-PCB box is a hairline nobody can find. Serve a thumbnail
      cropped to the truth's region (padded, one per class) or an outline drawn at the thumbnail's
      size, and use it in the Goal and Split strips.
- [ ] **Few-shot presets from coverage, as readiness counts it** (S): on VisA the guided run and the
      create form offer few-shot segmentation, because the imported defect masks give `defect`
      coverage, but the split presets read `class_counts`, which counts class-table revisions only, so
      no 1-shot or 5-shot preset is offered and the Split step can only point at Splits. One reading of
      "a class a sample shows" should feed both.

## Few-shot segmentation

The second task (ADR-0040), in dependency order. Each item is one PR.

- [ ] **Gate the mask on presence** (S): on FSS-1000 `proto_seg` ranks presence perfectly (ROC-AUC
      1.000), yet its `>= 0.5` mask marks foreground on 99.7 % of absent images
      ([measurements.md](measurements.md)). Write an empty mask where the run's presence score falls
      below a cut resolved on the references by one printed rule, predeclare a rerun of the FSS-1000
      and VisA protocols, and compare with the recorded verdicts.

Later, each behind a measured gate:
- INSID3 upstream and FSS-SAM3 as quality references.
- SAM-assisted pseudo-labelling at scale.
- A learned boundary refiner, or a dense CRF (which needs a maintained package chosen first).
- ONNX export with a mask output contract.

## Supervised tasks

Planned in ADR-0039. Supervised segmentation's slice runs — pinned classes, label targets, label
maps, the confusion-matrix evaluator, the `color_classifier` floor, the `dino_linear_seg` deep head,
the `class_stratified` split, its result screens and its public gates (`measurements.md`). Detection
runs end to end torch-free — box truth, box targets, stored boxes, COCO's AP, the `color_detector`
floor, the `class_stratified` split and its result screens — and the `dino_linear_det` deep detector
runs on the frozen-DINO path and has had its public gates on VisA and PKU-Market-PCB. What remains:

- [ ] **A box-regression head on the frozen DINO features** (M): on PKU-Market-PCB `dino_linear_det`
      finds defects (AP50 0.105, against the floor's 0.0001) but boxes connected regions of a 14-px
      grid, so its AP75 is 0.005 ([measurements.md](measurements.md)). Add a head that regresses box
      edges from the patch features, as a field of the detector, and predeclare a rerun of the PCB
      protocol against the recorded verdict.

## Methods

- [ ] **ONNX export for `dino_memory`'s single-image modes** (M): `per_image` fusion has a real
      single-input graph — encoder, distance kernel, upsample, blur — and `portable_formats` is empty
      today because `feature_concat` does not, and a format that is true for one configuration of a
      method is worse than an absent one. Either the capability becomes configuration-aware at the
      export seam, or the export refuses `feature_concat` by name the way `pixel_reference` refuses a
      multi-reference bank.
- [ ] **Give PatchCore the tuning EfficientAD got, then re-compare** (M): the first head-to-head is
      defaults against a tuned run, so the 0.047 sample ROC-AUC gap says nothing about the methods.
      `backbone`, `layer_set`, `coreset_ratio` and `max_candidate_vectors` are all fields, so each is
      an ablation the comparison screen can already show.
      At 768 × 768 the shipped `max_candidate_vectors` keeps about 2 % of each image's patches, and the
      AnomalyVFM gate's control scored 0.63 image ROC-AUC there ([measurements.md](measurements.md)): a
      frame-aware cap belongs in the same ablation.
- [ ] **Run both on the official one-class split** (`official-1cls`, split 2) (S): every number so
      far is on a generated split, and nothing is compared to a published figure until the official
      protocol runs.
- [ ] **Measure the custom EfficientAD hypotheses in order** (S each, mostly unattended compute):
      the step-budget curve first — one run continues into three points — then `calibration_holdout`,
      then `score_reduction`.
- [ ] **Run the teacher protocol sweep** (M, postponed): three runs at 30 000 steps is about three
      and a half hours, and the sweep's design changed underneath it — with new runs no longer using
      the anomalib teacher, the like-for-like leg is a deliberate exercise rather than routine.
      `protocol.py` is written and takes the budget as an argument.
- [ ] **Make the custom autoencoder resolution-agnostic** (M): replace the hard-coded `//64 - 1`
      upsample ladder and the 8×8 bottleneck so the 256 px floor goes away. The guard refusing
      smaller inputs is honest, but it refuses a configuration the architecture could support.
- [ ] **Batch inference for the deep methods** (M): one image per forward pass today. PatchCore's
      backbone forward is 7 ms of a ~22 ms image — worth it only once inference is the bottleneck in
      a comparison.
- [ ] **Sweep the three axes the campaign held fixed** (S): per-layer L2 normalization before
      pooling, `concat` instead of `mean` aggregation, and `final_norm=False`. Each is already a
      flag in `research/subspace_ad`, none is a plugin field, and all three change what the PCA sees
      rather than how it is read — so unlike τ and ρ they cost a forward pass each. Note before
      starting that `concat` is infeasible on a wide window at ViT-L: `upper_half` is thirteen
      blocks of 1 024, and a 13 312-dimensional covariance is 1.4 GB.
- [ ] **Measure `rotation_fill=masked`** (S): the campaign held it at `zeros` throughout, which is
      what a literal reading of the paper does, so the option that excludes a rotation's invented
      corners is shipped unmeasured. It is a one-axis rerun of one phase, and the honest expectation
      is that it matters most where the part does not fill the frame.
- [ ] **Decide what a `subspace_ad` bundle can promise** (S): the graph failed the real-pixel
      export-parity gate at `atol = rtol = 1e-4` with the ranking unchanged
      ([measurements.md](measurements.md)). Either predeclare a gate whose tolerance is sized to a
      24-block encoder feeding a residual — and say why that bound still separates rounding from a
      different operation — or measure the ViT-B and ViT-S configurations, whose encoders are half as
      deep, under the existing bound and let the plugin declare ONNX only for those. A fit over several
      channels stays refused; one graph per channel would need a bundle contract with a channel input.

## Evaluation

- [ ] **Give `TrainContext` labelled validation data**, so a method can report validation AUROC per
      epoch (M) — ADR-0007, handbook `evaluation.md`. The training chart wanted it and could not have it: `val` is
      filtered to normals and carries no labels, so there is one class and no AUROC. A
      plugin-interface decision, not a chart.

## Deployment

- [ ] **Test a dedicated-hardware handoff** (M): copy only the bundle and the runner, run offline on
      a second target, record latency, memory, provider and parity.

## Upstream

- [ ] **Report anomalib's non-reproducible coreset** (S): `SparseRandomProjection` is constructed
      with no `random_state`, so `KCenterGreedy` selects a different bank on every run at a fixed
      `torch.manual_seed`. Worked around here by pinning both streams; the library's own users have
      no way to know.
- [ ] **Revisit our training loop against anomalib's Lightning path** (S) if their module stops
      reaching into `trainer.datamodule`. Ours exists only because that coupling would cost the
      preprocessing bridge.

## Optional, later

- [ ] **`classical_circular`** — circle detection (Hough seed → radial-ray subpixel edges → robust
      fit), polar transform with FFT angular-correlation orientation, per-channel robust reference,
      and an inverse-polar z-map. The only plugin permitted to assume the showcase dataset's
      geometry, and the one place its design is written down is
      [methods.md](architecture/methods.md).
