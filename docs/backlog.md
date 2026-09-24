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
- [ ] **A link that looks like a button, upstream in lab-ui** (S): five screens nest a `<Button>`
      inside a `<Link>` (catalogue header, dataset band, sample viewer, experiment catalogue), which
      the control-inside-a-link rule forbids. lab-ui has `react-router` as a peer already; add a
      `ButtonLink` there, release, and replace all five.

## Few-shot segmentation

The second task (ADR-0040), in dependency order. Each item is one PR.

- [ ] **Calibrate the foreground probability** (M): absent images score foreground almost everywhere
      at 0.5. Fit the probability's scale on the references themselves (leave-one-out over the
      support set), so the cut means the same thing on every class; re-run the gate's protocol and
      compare with the recorded verdict.
- [ ] **A cross-domain few-shot gate** (M): VisA defects are small and subtle, the hardest target for
      a method built for objects. Choose a public few-shot segmentation dataset with object classes,
      predeclare the protocol in `measurements.md`, and run the three methods on it.

Later, each behind a measured gate:
- INSID3 upstream and FSS-SAM3 as quality references.
- SAM-assisted pseudo-labelling at scale.
- A learned boundary refiner, or a dense CRF (which needs a maintained package chosen first).
- ONNX export with a mask output contract.

## Supervised tasks

Planned in ADR-0039. Supervised segmentation's slice runs — pinned classes, label targets, label
maps, the confusion-matrix evaluator, the `color_classifier` floor and the `class_stratified` split. What remains, in dependency
order, one PR each:

- [ ] **Segmentation result screens** (M): through `taskViews.tsx` — a per-class IoU table across
      subsets, the stored confusion matrix drawn, the label map over the image on `SampleStage`
      (truth dashed, prediction solid), a per-sample verdict for the gallery in place of the neutral
      `scored`, and the task in the dataset readiness band.
- [ ] **A deep segmentation method on the frozen-DINO path** (M): a linear head on the shared patch
      features, fitted through `label_targets`, writing label maps; `dl`-gated tests, seed
      reproducibility in both directions, and one module plus one registry entry.
- [ ] **Public segmentation gate** (M): predeclared in `measurements.md` before it runs — VisA's
      masks as a `defect`-only segmentation benchmark on a supervised split, the deep head against
      `color_classifier`, with mean IoU as the decision and the others reported.
- [ ] **Detection** (L, split before starting): COCO-style AP, and predictions drawn against truth
      on the vector layer.

## Spatial input

- [ ] **Revisit automatic mask selection without test leakage** (M): MobileSAM's largest credible
      mask can be a background segment whose mask covers 63 % but whose bounding box is the full
      frame. Design the boundary/objectness rule on training normals, freeze it, then validate on
      different public classes.
- [ ] **Measure compact source-map persistence** (M): projected float32 maps consume about 1.23 GB
      for a 200-image VisA test set. Compare compressed source maps against prepared-frame map plus
      pinned-transform projection, preserving constant-memory evaluation and exact overlay semantics.

## Methods

- [ ] **ONNX export for `dinomaly_custom`** (S): the retired anomalib wrapper exported and this does
      not, which is the one capability asymmetry between them. The graph is not the hard part —
      encoder, bottleneck, decoder, cosine distance, upsample, blur, top-one-percent mean, all of it
      already traceable — the generic Python-versus-portable parity gate is what has to be written
      and run. `portable_formats` stays empty until it passes, because the export offer is made from
      the registry before any configuration is read.
- [ ] **Sweep `dinomaly_custom`'s encoder** (S): the reason the field exists. The default is the
      encoder anomalib pins, so a run against `dinov2_vit_b14` or `dinov3_vit_s16` on identical
      pixels answers whether Dinomaly's published result is about the method or about DINOv2 — a
      question the retired wrapper could not ask. Pair it with a `decoder_depth` ablation: depth 8 is the
      published recipe and nothing here has measured whether it is the right one for this data.
- [ ] **Evaluate AnomalyVFM as the zero-shot reference** (M): the resource gate passed — the pinned
      1.421 GB, 355.36M-parameter asset runs at 591 ms/image at 768 px on MPS with 2.07 GiB driver
      memory. App-managed offline loading, plugin integration and the public quality gate remain.
- [ ] **Sweep `dino_memory`'s layer selection for the DINOv3 backbone** (S): at the shared 448 px
      gate size DINOv3 ViT-S/16 cleared the floors but trailed DINOv2 ViT-S/14-reg4 on every
      metric with `last_two` ([measurements.md](measurements.md)). Before concluding the
      backbone is weaker for this task, sweep `layers` (and consider `mid_late`) on the same
      pixels — the deficit may belong to the recipe, not the encoder.
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
- [ ] **Run the public promotion gate for `subspace_ad`** (M): it ships `experimental` because it
      has not run the one thing every other promoted method ran — a paired control on shared
      immutable pixels, VisA `candle` and `pcb1` at 448 × 448 against PatchCore
      ([measurements.md](measurements.md)). The sweep behind its defaults is far more evidence than
      any gate produces, but it is evidence of a different kind: it was collected outside the
      application, by its own harness. `scripts/dino-memory-public-gate.py` is the template.
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
- [ ] **ONNX export for `subspace_ad`** (M): unlike `dino_memory`, every configuration of this
      method has a single-input static graph — encoder, centre, project onto a fixed basis,
      residual, sort for the tail mean, upsample, blur — with no data-dependent control flow and no
      per-channel branch once a channel's basis is chosen. `portable_formats` is empty today because
      nothing has been measured for parity, not because the graph is hard.

## Evaluation

- [ ] **Give `TrainContext` labelled validation data**, so a method can report validation AUROC per
      epoch (M) — ADR-0007, ADR-0011. The training chart wanted it and could not have it: `val` is
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
