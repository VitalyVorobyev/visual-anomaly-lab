# Backlog

Open work only. Anything shipped is described in the [handbook](architecture/README.md), not here.
[roadmap.md](roadmap.md) says where the workbench stands; this says what to do next.

**Sizes.** `S` ≈ half a day. `M` ≈ one focused day. `L` = multi-day, and should be split before it
is started rather than after. Sizes include reading the generated code properly — a task is not done
until its output has been reviewed.

## Interface

- [ ] **Visual QA in light and dark at 1440×900 and 1024×768** (M): hierarchy, density, contrast,
      focus, loading/empty/error/disabled states, and no same-axis nested scroll.
- [ ] **Finish the large-catalogue experiment workflow** (M): id query, multi-select methods, date
      range, cursor pagination, sortable column headers, and compatible selection handed to Compare.

## Spatial input

- [ ] **Revisit automatic mask selection without test leakage** (M): MobileSAM's largest credible
      mask can be a background segment whose mask covers 63 % but whose bounding box is the full
      frame. Design the boundary/objectness rule on training normals, freeze it, then validate on
      different public classes.
- [ ] **Measure compact source-map persistence** (M): projected float32 maps consume about 1.23 GB
      for a 200-image VisA test set. Compare compressed source maps against prepared-frame map plus
      pinned-transform projection, preserving constant-memory evaluation and exact overlay semantics.

## Methods

- [ ] **Evaluate AnomalyVFM as the zero-shot reference** (M): the resource gate passed — the pinned
      1.421 GB, 355.36M-parameter asset runs at 591 ms/image at 768 px on MPS with 2.07 GiB driver
      memory. App-managed offline loading, plugin integration and the public quality gate remain.
- [ ] **Sweep `dino_memory`'s layer selection for the DINOv3 backbone** (S): at the shared 448 px
      gate size DINOv3 ViT-S/16 cleared the floors but trailed DINOv2 ViT-S/14-reg4 on every
      metric with `last_two` ([measurements.md](measurements.md)). Before concluding the
      backbone is weaker for this task, sweep `layers` (and consider `mid_late`) on the same
      pixels — the deficit may belong to the recipe, not the encoder.
- [ ] **Let `diagnose` pass a whole sample group** (S): `experiments/diagnose.py` scores a single
      record, so asking a `feature_concat` `dino_memory` model about one image of a multi-channel
      sample lands in the channel refusal instead of producing a diagnostic. The refusal is readable
      and names the sample; the fix is to resolve the sample's group the way `infer` does.
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
