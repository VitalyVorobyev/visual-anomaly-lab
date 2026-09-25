# Measurements

The numbers that still decide something: what was measured, under what protocol, and what it settled.
Method design is in [methods.md](architecture/methods.md); this page is the evidence it rests on.

**Conventions.** Every gate is predeclared — floors fixed before the run, seeds recorded, one child process
per leg so peak memory is comparable. Host: Apple Silicon, macOS 26.5.2, Python 3.12.12, anomalib 2.6.0,
torch 2.13.0, timm 1.0.28; public data only. Unless stated, a "paired public gate" is VisA `candle` and
`pcb1`, seed 20260812, against a PatchCore control on the same immutable prepared pixels, with floors of
0.80 image ROC-AUC, 0.85 pixel ROC-AUC and 0.60 AU-PRO on the two-class mean.

## EfficientAD against the anomalib baseline

The comparison ADR-0029 keeps honest. Protocol: VisA's **published** one-class test set, byte-identical,
with the validation holdout carved from `train` only (810 normal train, 90 val, 100 + 100 test). The
baseline leg is anomalib's EfficientAD wrapper, which is not in the registry, so it cannot be re-run in-app.

**Teacher choice.** Two public EfficientAD teachers disagree on the same data (ADR-0031):

| Sample ROC-AUC by aggregation | `anomalib` teacher | `nelson1425` teacher |
|---|---:|---:|
| `max` (the paper's) | 0.751 | **0.886** |
| top-64 mean | 0.758 | 0.888 |
| p99 | 0.765 | 0.798 |
| plain mean | 0.756 | 0.761 |

The same effect across metrics, VisA `candle`, 4 000 steps, three seeds each (ranges do not overlap):

| Teacher | Sample ROC-AUC | Pixel ROC-AUC | AU-PRO |
|---|---:|---:|---:|
| `anomalib` | 0.769 | 0.891 | 0.539 |
| `nelson1425` | 0.889 | 0.978 | 0.914 |

Verdict: the teacher is an experiment variable, larger in effect than 26 000 additional training steps.

**Step budget**, read at four points of one continued trajectory:

| Steps | Sample ROC-AUC | Pixel ROC-AUC | AU-PRO | ms/image |
|---|---:|---:|---:|---:|
| 4 000 | 0.886 | 0.981 | 0.916 | 26.5 |
| 8 000 | 0.916 | 0.989 | 0.936 | 26.6 |
| 16 000 | 0.943 | 0.992 | **0.944** | 25.5 |
| **30 000** | **0.955** | **0.994** | 0.943 | 26.4 |

Verdict: localisation converges before ranking — AU-PRO plateaus by 16 000 steps while sample ROC-AUC still
climbs at 30 000.

## Dinomaly — promoted

Transformer reconstruction over a frozen DINOv2 encoder: 37.42M parameters, 15.36M trainable; resumable
payload 245.9 MB, encoder fingerprinted rather than copied. At 392 × 392 on MPS: 122.4 ms per training step,
42.3 ms inference, 0.82 GiB peak RSS; training only 1.07× faster than CPU, so CPU is a credible fallback.

Paired public gate (anomalib wrapper leg, 392 × 392):

| Mean over both classes | Dinomaly | PatchCore | Floor |
|---|---:|---:|---:|
| Image ROC-AUC | **0.9634** | 0.6896 | 0.80 |
| Pixel ROC-AUC | **0.9953** | 0.9626 | 0.85 |
| AU-PRO | **0.9514** | 0.8006 | 0.60 |

Verdict: all three floors cleared. ≈ 600 s to fit against PatchCore's ≈ 17 s — bounded, Mac-feasible,
resumable, and a different failure mode from a memory bank.

## `dinomaly_custom` parity

Protocol: the wrapper's exactly — 392 × 392, DINOv2 ViT-S/14-reg4, paired public gate. One training step
matches anomalib at atol = 0 (loss, every trainable gradient, post-step weights, eval map, image score), so
the gate measures only the divergence training order and RNG streams accumulate over 5 000 steps.

| Mean over both classes | `dinomaly_custom` | Wrapper's recorded run | PatchCore | Floor |
|---|---:|---:|---:|---:|
| Image ROC-AUC | **0.9636** | 0.9634 | 0.6896 | 0.80 |
| Pixel ROC-AUC | **0.9951** | 0.9953 | 0.9626 | 0.85 |
| AU-PRO | **0.9511** | 0.9514 | 0.8006 | 0.60 |

Verdict: parity to the third decimal; ≈ 570 s per training leg (wrapper ≈ 610 s), 50 ms inference, 0.98 GB
peak RSS. `dinomaly_custom` is the carried implementation.

## Export parity on real pixels — `dinomaly_custom` exports

Whether a method's ONNX bundle computes what the method computes on the activations a *trained* network
sees on real parts — the export job's own check uses a dataset-free ramp. A method lists a portable format
only after passing this. Predeclared before it ran; `scripts/export-parity-gate.py`, method-agnostic.

**Protocol.** VisA `candle` and `pcb1`, official 1cls split, the 200-image test subset of each, identity
profile at the candidate's gate size, the candidate's recorded gate configuration and seed 20260812 at its
full step budget. The application's own train and infer jobs fit and score on the preferred device; the
ordinary export job writes and fixture-checks the bundle. Every test image's prepared tensor then goes to
both the method's `portable_reference` (torch, CPU) and the bundle's graph (ONNX Runtime 1.28.0, CPU),
with the score read through the manifest's contract as a host would (`deployment/parity.py`, the rule the
export job applies).

**Tolerance, fixed before the run: `atol = rtol = 1e-4`** on every map (`allclose`) and `atol = 1e-4` on
every score. It is the bound PatchCore's bundle already declares for a deep frozen backbone: both sides run
the same float32 operations through different kernels and summation orders, across a twelve-block encoder
and a decoder stack, so a disagreement of order 10⁻⁶–10⁻⁵ is rounding. The bound sits an order above that,
and two orders below the gap between a normal and a defective part's score; a larger disagreement means the
graph computes a different operation.

**Decision rule.** Pass only if, on both classes: the export job publishes a bundle; every test image is
within tolerance on map and score; and image ROC-AUC over the portable scores is within **0.001** of that
over the Python scores. Reported, not gated: the portable outputs against the workbench's stored results,
which also carry the accelerator's own rounding.

**`dinomaly_custom`** at its gate configuration (392 × 392, DINOv2 ViT-S/14-reg4, 5 000 steps on MPS; a
134.7 MB opset-18 graph with the score as a named tensor), 200 test images per class:

| | `candle` | `pcb1` |
|---|---:|---:|
| Images within tolerance | 200 / 200 | 200 / 200 |
| Worst map error | 2.3 × 10⁻⁶ | 1.5 × 10⁻⁶ |
| Worst score error | 4.5 × 10⁻⁷ | 3.5 × 10⁻⁷ |
| Image ROC-AUC, Python / portable | 0.9624 / 0.9624 | 0.9648 / 0.9648 |
| Export job's fixture, map / score error | 1.5 × 10⁻⁶ / 3.0 × 10⁻⁷ | 1.8 × 10⁻⁶ / 0 |
| Portable against stored MPS results, map / score (reported) | 1.7 × 10⁻⁶ / 3.6 × 10⁻⁷ | 1.2 × 10⁻⁶ / 3.3 × 10⁻⁷ |

Verdict: passed on both classes, with the worst disagreement some forty times inside the tolerance and the
ranking identical. `dinomaly_custom` lists ONNX.

## `dinomaly_custom`'s encoder and decoder depth — the defaults stay; the result is the method's

Whether Dinomaly's recorded result is about the method or about DINOv2, and whether the published
decoder depth of 8 suits this data. Predeclared before the sweep ran; `scripts/dinomaly-encoder-sweep.py`,
8 cells in 1.9 h on MPS (torch 2.13.0, timm 1.0.28, anomalib 2.6.0, numpy 2.5.1, Pillow 12.3.0).
Each arm is a full fit, so no arm can share a forward pass with another: this is the gate's harness,
the application's own train and infer jobs, not a campaign's (ADR-0038).

**Protocol.** VisA `candle` and `pcb1`, official 1cls split, one identity prepared-input build per class
at **448 × 448**, the one size both patch sizes divide, so every arm sees identical pixels. That is not
the recorded gate's 392, which /16 does not divide, so the default arm is re-run here rather than read
from the parity table. Every field not named below is the shipped default (5 000 steps, batch 1,
learning rate 2 × 10⁻³, decoder depth 8); seed 20260812. One child process per cell.

| Arm | Encoder | Decoder depth |
|---|---|---:|
| default | `dinov2_vit_s14_reg4` | 8 |
| `dinov2_vit_b14` | DINOv2 ViT-B/14 | 8 |
| `dinov3_vit_s16` | DINOv3 ViT-S/16 (licence-gated) | 8 |
| `depth_4` | `dinov2_vit_s14_reg4` | 4 |

**Reported.** Per class and arm: image ROC-AUC, pixel ROC-AUC, AU-PRO, fit time, ms/image over the
200-image test subset (the infer job's wall time, maps included), peak RSS of the child.

**Decision rule, fixed before the run.** A variant replaces the default only if its image ROC-AUC beats
the default arm's by **at least 0.01 on both classes**, and its two-class mean pixel ROC-AUC and mean
AU-PRO each lose **no more than 0.01**. Several passing → the largest summed image gain. A licence-gated
encoder is never the default, whatever it scores — an untouched run must not need an account; if one
passes, it is recorded as the better choice for those with access. The rule applies to `depth_4`
unchanged, so a cheaper decoder that only ties does not move the default; its cost is reported. If
nothing passes, the defaults stay and the answer to the question is what the table records. One seed
per cell, so the 0.01 margin on *each* class is the guard against reading seed noise as a result.

**Budget.** One smoke cell per arm on `candle` at 200 steps, run to time the arms before the sweep:
0.13 s per training step for the default, 0.35 s for ViT-B/14, 0.12 s for DINOv3 ViT-S/16, 0.09 s at
depth 4 — about 2.1 h for the eight cells. Its metrics are not part of the sweep.

**Result.** One seed per cell.

| Class | Arm | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | Fit s | ms/image | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| `candle` | default | 0.9639 | 0.9953 | 0.9584 | 644 | 85.6 | 0.95 |
| | `dinov2_vit_b14` | 0.9708 | 0.9956 | 0.9703 | 1655 | 160.4 | 1.84 |
| | `dinov3_vit_s16` | 0.9628 | 0.9940 | 0.9596 | 563 | 74.9 | 0.91 |
| | `depth_4` | 0.9652 | 0.9952 | 0.9606 | 442 | 77.7 | 0.78 |
| `pcb1` | default | 0.9656 | 0.9968 | 0.9643 | 638 | 85.6 | 0.93 |
| | `dinov2_vit_b14` | 0.9696 | 0.9976 | 0.9689 | 1640 | 161.7 | 2.03 |
| | `dinov3_vit_s16` | 0.9626 | 0.9967 | 0.9600 | 565 | 76.3 | 0.91 |
| | `depth_4` | 0.9704 | 0.9969 | 0.9649 | 444 | 79.6 | 0.80 |

**Verdict, by the rule.** No variant passes, so **the defaults stay**: `dinov2_vit_s14_reg4` and decoder
depth 8. Image ROC-AUC gains over the default are +0.007 / +0.004 for ViT-B/14 (`candle` / `pcb1`),
−0.001 / −0.003 for DINOv3 ViT-S/16 and +0.001 / +0.005 for depth 4 — none reaches the 0.01 margin on
either class, and no arm loses more than 0.002 on the mean pixel metrics.

**What the table answers.** Dinomaly's result here is the method's, not DINOv2's: a different family at
the same size, DINOv3 ViT-S/16, lands within 0.003 of the default on every metric. A larger encoder buys
under 0.01 for 2.6× the fit time, 1.9× the latency and twice the memory. Depth 4 ties at 69 % of the fit
time and 91 % of the latency; by the rule a tie does not move the default, so it stays a field for a run
that wants the cheaper fit.

## GLASS — available, not recommended

Frozen ImageNet backbone, trainable projection and discriminator: 28.80M parameters, 3.94M trainable,
49.3 MB resumable payload. Batch 8 buys no throughput on this host, so the plugin runs at batch 1,
288 × 288.

| Mean over both classes | GLASS | PatchCore | Floor |
|---|---:|---:|---:|
| Image ROC-AUC | 0.7938 | 0.9013 | 0.8000 |
| Pixel ROC-AUC | 0.8986 | 0.9910 | 0.8500 |
| AU-PRO | 0.6052 | 0.8762 | 0.6000 |

Verdict: missed the image floor by 0.0062 and lost to PatchCore on all three, so it ships experimental. The
gate is deliberately **not** the upstream protocol (39 200 exposures, category-specific `svd`, best
checkpoint chosen on test evidence): it uses 5 000 exposures, one generic synthesis distribution and the
final checkpoint. A rerun may raise the budget but must not tune distribution, stopping point or checkpoint
per category.

## AnomalyVFM — integrated; public gate predeclared, not yet run

Resource gate (`scripts/anomalyvfm-smoke-test.py`): a 1.421 GB, 355.36M-parameter adapted RADIO
checkpoint, 591 ms/image at 768 px on MPS, 2.07 GiB driver memory. Verdict: Mac-credible when the
weights are read once per job; 768 px kept for quality.

It set two invariants, and `anomalyvfm_anomalib` honours both: anomalib downloads inside its model
constructor (`local_files_only=False`), so the plugin resolves and verifies the pinned checkpoint and
answers the constructor's download call with that file; and anomalib reports export as unsupported,
so `portable_formats` is empty. Whether it is credible on quality is the public gate's question, which
has not run; until it does the method ships experimental.

**Public gate — predeclared.** `scripts/anomalyvfm-public-gate.py`. The paired public gate at the frame
the resource gate kept: VisA `candle` and `pcb1`, official 1cls split, one identity prepared-input build
per class at **768 × 768**, scored by `anomalyvfm_anomalib` (defaults) and by the standard PatchCore
control, which is re-run on the same build rather than read from another size. AnomalyVFM's train job
only verifies the checkpoint; it reads no image. Seed 20260812, which only the control consumes. One
child process per leg.

**Reported.** Per class and as the two-class mean: image ROC-AUC, pixel ROC-AUC, AU-PRO; for each leg,
train seconds, ms/image over the 200-image test subset (the infer job's wall time, maps and the
checkpoint's load included) and peak RSS of the child.

**Decision rule, fixed before the run.** AnomalyVFM is promoted from experimental to the supported
zero-shot reference if its two-class means clear the standard floors (0.80 image ROC-AUC, 0.85 pixel
ROC-AUC, 0.60 AU-PRO). Beating PatchCore is **not** required: the control is fitted on each class's
normals and AnomalyVFM sees none, so the difference is recorded as what those normals bought on these
classes, not as a verdict. Missing any floor keeps it experimental. The network has no random stream, so
one run per class is the whole measurement; a different frame is a separate, predeclared question,
never a rerun of this gate.

**Budget.** AnomalyVFM's scoring is about four minutes of the run at the measured 591 ms/image (400 test
images); the two 768-pixel builds and the PatchCore control at 768 have not been timed.

## Region profiles — identity stays the default

Rule, fixed before the run: at least +0.01 mean pixel ROC-AUC *and* +0.01 mean AU-PRO, losing no more than
0.02 on either, with zero build failures.

| Class | Profile | Mean crop | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | Missed defect pixels |
|---|---|---:|---:|---:|---:|---:|
| `candle` | identity | 100.0 % | 0.867 | 0.963 | 0.890 | 0 / 405 291 |
| `candle` | threshold | 14.1 % | 0.620 | 0.511 | 0.192 | 347 202 / 405 291 |
| `pcb1` | identity | 100.0 % | 0.717 | 0.982 | 0.713 | 0 / 1 398 914 |
| `pcb1` | threshold | 31.4 % | **0.834** | 0.833 | 0.456 | 351 910 / 1 398 914 |

Verdict: mean deltas **−0.300 pixel ROC-AUC**, **−0.478 AU-PRO** — rejected. The `pcb1` image gain (+0.117)
is what the gate exists to catch: without inverse projection and uncovered-pixel accounting it reads as
evidence *for* localisation while the crop omitted a quarter of the defect pixels. `foreground_threshold`
stays available as an explicit, previewable choice. Source-frame float maps cost about 1.23 GB per 200-image
run regardless of crop — settled by [anomaly-map storage](#anomaly-map-storage), not a reason to change the verdict.

## MobileSAM mask selection — the border rule stays opt-in

`mobile_sam` keeps the largest mask inside its area window. **Design evidence** — MobileSAM's candidates on
12 evenly spaced *training normals* of each of six VisA design classes (`candle`, `capsules`, `cashew`,
`chewinggum`, `fryum`, `pcb1`; 72 images, `scripts/mask-selection-gate.py --design`): the default chose a
background segment on 72 / 72, covering at least 66 % of the frame's one-pixel border with a box of at least
99.6 % of the frame. Every candidate on those images covered either at most 3 % of the border or at least
66 %; none fell between. On `candle` the largest mask away from the border held two of the four candles.

**Rule, frozen from that evidence alone:** drop any mask covering more than **0.33** of the border (the gap's
midpoint) and return the box around **every** surviving mask (`max_border_fraction = 0.33`,
`selection = union`); nothing survives → an explicit extraction failure. Area window, grid and quality
thresholds unchanged.

**Validation, on classes the design never saw:** VisA `macaroni1`, `macaroni2`, `pcb2`, `pcb3`, `pcb4`,
`pipe_fryum` and MVTec-AD's ten object classes decide; MVTec-AD's five textures are reported only, since a
texture has no object to find. At most 24 evenly spaced anomalous test images per class, one MobileSAM pass
per image, both rules applied to the same candidates. A selection is **correct** when its unpadded box keeps
at least 99 % of the image's ground-truth defect pixels and covers at most 90 % of the frame; a failure is
incorrect and keeps nothing. The rule becomes the `mobile_sam` default when, pooled over the deciding
images, (1) its correct rate beats the current default's by at least 0.20, (2) it keeps at least 0.98 of all
defect pixels, and (3) it fails on at most 5 % of images. The largest surviving mask alone is reported
beside it and decides nothing.

| 16 deciding classes, 384 images | Correct | Defect pixels kept | Failures | Mean box |
|---|---:|---:|---:|---:|
| current default (largest in window) | 0.151 | 0.975 | 0 % | 96.0 % |
| **border 0.33, union** | **0.781** | **0.920** | 0 % | 50.2 % |
| border 0.33, largest (reported only) | 0.734 | 0.797 | 0 % | 37.4 % |

Verdict: checks (1) and (3) pass, **(2) fails** — the rule stays available and **is not the default**. It
does what it was designed to do: the default's mean box was at least 97 % of the frame on 13 of the 16
classes; the rule's was 17–48 % on every VisA class, and it kept at least 0.99 of the defect pixels on
`macaroni1`, `macaroni2`, `pipe_fryum`, `capsule`, `hazelnut`, `pill` and `toothbrush`.
The lost pixels are concentrated: `metal_nut` keeps 0.52, `pcb3` 0.86, `pcb2` 0.90, `screw` 0.92, where
defects lie outside every surviving mask's box. Two classes show the rule's limits in the other
direction: on `zipper` and `transistor` a surviving mask still spans the frame, so it keeps everything
and localises nothing. Uniting rather than picking one mask was right — the largest surviving mask alone
keeps 0.80. Textures: both rules fail on 74–78 % of images, which is the correct answer for a picture
with no object. Retention is measured on the unpadded box; a profile's `padding_fraction` would recover
some of the loss, and the protocol did not credit it. 504 images, one CPU MobileSAM pass each (1.01 s
mean; MPS rejects the prompt grid's float64), 519 s in all.

## DINO patch memory — promoted

Frozen DINO patch features are the model (ADR-0037); nothing is trained. The 15.4 MB checkpoint is a
5 000-vector coreset over 50 000 bounded candidates; the encoder travels as a fingerprint. Paired public
gate at 448 × 448 (divisible by both patch sizes, so every backbone saw identical pixels), `global_knn`,
`last_two` layers, k = 1. The recorded leg is **DINOv2 ViT-S/14-reg4** (ungated weights):

| Mean over both classes | DINO memory | PatchCore | Floor |
|---|---:|---:|---:|
| Image ROC-AUC | **0.9000** | 0.8565 | 0.80 |
| Pixel ROC-AUC | **0.9921** | 0.9889 | 0.85 |
| AU-PRO | **0.9315** | 0.9246 | 0.60 |

Verdict: all floors cleared and the control beaten on all three means. 13–21 s to fit, 42 ms/image, ≈ 1.0 GB
peak RSS on MPS (PatchCore: 48 ms, 1.75 GB). Per class: `pcb1` 0.891 vs 0.775 image ROC-AUC; `candle` 0.909
vs 0.938, with a pixel-level win.

- **DINOv3 ViT-S/16**, same recipe, licence-gated (`HF_TOKEN`), not the recorded gate: 0.8147 / 0.9850 /
  0.8588 — clears the floors, trails DINOv2 on every metric, weakest on `candle` (0.725 image). At 448 px
  /14 gives a 32 × 32 grid, /16 a 28 × 28 one. The ungated default is also the measured best.
- **Per-position ablation** (`local_knn`, r = 1, `pcb1`): image 0.847 / pixel **0.9969** / AU-PRO **0.9510**
  against `global_knn`'s 0.891 / 0.9952 / 0.9252. Restricting the bank by position sharpens *where* and
  costs a little *whether*. VisA is not a registered benchmark; this is the one public anchor for the mode.

## SubspaceAD — defaults from a sweep, gate open

**Not a gate**: a parameter search run outside the application by its own harness (ADR-0038). The promotion
gate is in [backlog.md](backlog.md); the method ships `experimental` until it runs.

**Protocol.** VisA's twelve categories under the official one-class split for three tuning phases, MVTec-AD's
fifteen held out as a fourth. The category is the unit of evidence: seeds fold inside a category, categories
average, comparisons are paired within category, ± is the standard error over categories, ★ marks a
difference larger than twice its own. 215,874 rows, about twelve hours. Few-shot protocol is the paper's:
k ∈ {1, 2, 4} normals, each augmented with 30 random rotations up to 345°, corners black, `transistor` not
rotated. Three axes are free to sweep — τ is a `cumsum` over one projection, ρ one sort, k nested draws —
so only encoder and input size cost a forward pass.

| Axis | Swept | Verdict | Margin |
|---|---|---|---|
| Layer window | 9 windows, at depth 12 and 24 | `upper_half` | +0.12 image AUROC over the paper's window at depth 12 |
| Window *encoding* | relative band vs fixed count | **a fraction of depth** | fixed count loses 6.5 / 1.5 / 0.0 points at ViT-S / ViT-B / ViT-L on VisA, and 3.6–4.7 ★ at depth 12 on MVTec |
| Backbone | DINOv2 and DINOv3 at S, B, L | `dinov2_vit_l14` | +0.0249 ★ over ViT-B on VisA — and **nothing at all on MVTec** |
| Prepared size | 448, 672 | 672 | a tie for detection at ViT-S/B; +0.0079 ★ at ViT-L |
| τ (`variance`) | 0.95, 0.97, 0.99, 1.0 | 0.99 | 1.0 collapses the score to float noise (0.589) |
| ρ (`tail_fraction`) | 0.001 – 0.02 | 0.002 | interior, and neither benchmark's preference is decisive |

**Leading VisA arm by phase** (mean over twelve categories, each averaged over seeds):

| | Phase 3 | Phase 2 | Phase 1 |
|---|---:|---:|---:|
| Encoder | `dinov2_vit_l14` | `dinov3_vit_b16` | `dinov2_vit_s14_reg4` |
| Window | `upper_half` · 12–24 of 24 | `final7` · 6–12 | `last_four` · 9–12 |
| τ · ρ | 0.99 · 0.002 | 0.99 · 0.005 | 0.99 · 0.01 |
| Image AUROC | **0.9472 ±0.0135** | 0.9275 ±0.0215 | 0.8939 ±0.0325 |
| Pixel AUROC | 0.9869 | 0.9841 | 0.9832 |
| AU-PRO | **0.9683** | 0.9555 | 0.9448 |
| Worst category | `macaroni2` · 0.847 | `macaroni2` · 0.730 | `macaroni2` · 0.587 |

The standard error more than halved, so the spread shrank rather than easy categories lifting the mean.
Phase 1 is a near-tie: `dinov3_vit_s16` with `final_band` reaches 0.8947 ±0.0307 on detection and loses
AU-PRO by 1.2 points; DINOv2 carries forward for that and for its ungated weights.

**Phase 4 — MVTec-AD held out**, shipped configuration unchanged:

| | MVTec-AD, 15 categories | VisA, 12 categories |
|---|---:|---:|
| Image AUROC | **0.9731 ±0.0105** | 0.9472 ±0.0135 |
| Average precision | 0.9823 | 0.9482 |
| Pixel AUROC | 0.9797 | 0.9869 |
| AU-PRO | 0.9549 | 0.9683 |
| Worst category | `screw` · 0.871 | `macaroni2` · 0.847 |

`grid`, `leather` and `metal_nut` are 1.0000, `bottle` 0.9992; the low ones are `screw` (0.871), `cable`
(0.912), `transistor` (0.928, fitted on 9,216 patches against 285,696 elsewhere, being unrotated) and `pill`
(0.952).

Phase 4 verdicts — no default changed:

- **Window encoding: confirmed, more strongly.** At depth 24 fixed count loses by **−0.0098 ★ (3–9)** on
  MVTec (a tie on VisA, −0.0016); at depth 12 by **−0.0472 ★ (2–12)** on DINOv3 ViT-B and **−0.0364 ★
  (3–11)** on DINOv2 ViT-B.
- **ρ: interior optimum.** 0.001 is worse (0.9691 against 0.9731). VisA prefers 0.002 by +0.0051 ±0.0037,
  MVTec 0.005 by +0.0023 ±0.0013 — neither decisive. 0.002 ships because VisA falls 2.5 points between 0.002
  and 0.02 while MVTec spans 0.6 points over the whole range.
- **`upper_half`: never worse.** Against `final7` it wins on VisA (+0.0098 ★, 10–2) and ties on MVTec
  (+0.0015, 5–6); it beats both readings of the paper's window on MVTec (+0.0133 ★ and +0.0230 ★, 12–1 and
  13–1).
- **Backbone: did not transfer.** ViT-L beat DINOv2 ViT-B by +0.0249 ★ (11–1) on VisA and tied on MVTec
  (**−0.0002 ±0.0014**, 5–7) at roughly twice the compute. ViT-L stays default as never worse; on
  MVTec-like data ViT-B gives the same answer for half the cost.
- **Few-shot curve flattens immediately.** k = 1 → 2 buys +0.0156; 2 → 4 buys +0.0023.

**Plugin reproduces the campaign.** Shared arithmetic (`models/subspace.py`, `models/score_map.py`),
different plumbing (prepared PNGs through `load_array`, `evenly_spaced` fit images, rotations seeded from one
integer), so not bit-exact. VisA `candle`, k = 4:

| Arm | Plugin | Campaign, per seed | Patches | Rank |
|---|---:|---|---:|---:|
| ViT-S/14-reg4, 448 px, `upper_half`, τ 0.99, ρ 0.01 | 0.9386 | 0.9369 / 0.9246 / 0.9451 / 0.9436 / 0.9356 | 126,976 both | 91 vs 91–92 |
| ViT-L/14, 672 px, `upper_half`, τ 0.99, ρ 0.002 — **the shipped defaults** | 0.9462 | 0.9504 / 0.9428 / 0.9445 | 285,696 both | 300 vs 299–303 |

Identical patch counts, ranks inside the campaign's range, image AUROC within its seed spread. At the
shipped defaults a four-image fit is 120 s and inference 649 ms/image on MPS (upper bounds; measured under
load).

**Open.** No paired control on shared prepared pixels. Held fixed throughout: per-layer L2 before pooling,
`concat` instead of `mean`, `final_norm=False`; `rotation_fill=masked` never measured. All four are in
[backlog.md](backlog.md).

## Few-shot segmentation — `proto_seg` is the default; no method yet draws a usable mask

The first few-shot gate (ADR-0040), predeclared below before it ran. `scripts/few-shot-public-gate.py`,
72 runs in 1.8 h on MPS (torch 2.13.0, timm 1.0.28, numpy 2.5.1, Pillow 12.3.0).

**Protocol.** VisA `candle` and `pcb1`, identity prepared input at 448 × 448, which both DINO patch sizes
divide. The target class is `defect`, from VisA's pixel masks. For each class, `few_shot` splits draw
k ∈ {1, 2, 5, 10} references among the 100 defect samples under seeds {0, 1, 2}. Every other sample is a
query: the rest of the defects, and the 1 000 normals as confirmed absences. Three methods run at their
shipped defaults on the same pixels: `color_prototype`, `fss_dino` (DINOv2 ViT-B/14) and `proto_seg`
(DINOv2 ViT-B/14). That is 72 runs, one child process each.

**Reported.**
- Per method and shot count: foreground IoU, boundary F1, presence ROC-AUC, the false-positive rate on
  absent images, and ms per image.
- Each is a mean over the two classes and three seeds, with the spread across seeds as the measure of
  support sensitivity.

**Decision rule, fixed before the run.** The primary number is foreground IoU at 5 shots.
- `proto_seg` becomes the default few-shot method (the reference studio's first choice) if both hold:
  - it beats `fss_dino` by at least 0.02 on the primary;
  - its presence ROC-AUC at 5 shots is no more than 0.02 below `fss_dino`'s.
  Otherwise `fss_dino` is the default, and `proto_seg` stays experimental.
- The DINO methods are credible at all only if `fss_dino` beats `color_prototype` on the primary.

**Result.** Means over both classes and three seeds; the spread is across seeds.

| Method | Shots | Foreground IoU | ± seeds | Boundary F1 | Presence ROC-AUC | Absent FPR | ms/image |
|---|---|---|---|---|---|---|---|
| `color_prototype` | 1 | 0.0024 | 0.0010 | 0.0020 | 0.510 | 0.934 | 28 |
| | 2 | 0.0029 | 0.0002 | 0.0025 | 0.480 | 1.000 | 29 |
| | 5 | 0.0026 | 0.0002 | 0.0021 | 0.301 | 1.000 | 28 |
| | 10 | 0.0025 | 0.0001 | 0.0022 | 0.318 | 1.000 | 28 |
| `fss_dino` | 1 | 0.0086 | 0.0021 | 0.0030 | 0.584 | 1.000 | 85 |
| | 2 | 0.0055 | 0.0025 | 0.0026 | 0.695 | 1.000 | 88 |
| | 5 | 0.0037 | 0.0010 | 0.0015 | 0.724 | 1.000 | 90 |
| | 10 | 0.0035 | 0.0004 | 0.0010 | 0.721 | 1.000 | 91 |
| `proto_seg` | 1 | 0.0252 | 0.0128 | 0.0087 | 0.700 | 0.943 | 99 |
| | 2 | 0.0111 | 0.0057 | 0.0051 | 0.738 | 0.981 | 102 |
| | 5 | 0.0357 | 0.0253 | 0.0121 | 0.820 | 0.926 | 102 |
| | 10 | 0.0216 | 0.0194 | 0.0075 | 0.849 | 0.955 | 103 |

By class at 5 shots, presence ROC-AUC is 0.920 (`proto_seg`) and 0.912 (`fss_dino`) on `candle`, and
0.721 and 0.536 on `pcb1`; `color_prototype` ranks below chance on both (0.342, 0.260). `proto_seg`'s
IoU lead is `candle`'s alone (0.067 against 0.004 on `pcb1`).

**Verdict, by the rule.** `proto_seg` beats `fss_dino` by 0.032 on the primary and leads it on presence
ROC-AUC, so **`proto_seg` is the default few-shot method** and the reference studio's first choice. The
credibility check passes by its letter only: `fss_dino` leads `color_prototype` by 0.001 IoU, which is
inside the noise. The evidence that the DINO methods beat the floor is presence ranking, where both do
and the floor ranks worse than chance.

**What the gate says beyond its rule.** No method draws a usable mask of a VisA defect: every IoU is
below 0.04, and the false-positive rate on absent images is near 1. The foreground probability is not
calibrated to the evaluator's fixed `>= 0.5` rule — on a small, subtle defect class most of an image
clears it — so the primary measures the cut as much as the segmentation. The methods do rank *which*
images hold the class. Mask ranking apart from the cut is `pixel_average_precision`, which the
evaluator reports and this verdict predates; a calibrated foreground probability is the calibration leg
below. VisA
defects are also a hard target for a method built for objects, which is why a cross-domain few-shot
dataset is the next gate.

### Calibration leg — `none` stays the default

Predeclared before it ran. `scripts/few-shot-public-gate.py --leg calibration`, 72 runs in 1.9 h on MPS
(torch 2.13.0, timm 1.0.28, numpy 2.5.1, Pillow 12.3.0), asks whether a foreground
probability fitted on the references makes the fixed `>= 0.5` cut mean something
([methods](architecture/methods.md#calibrating-the-foreground-probability)).

**Protocol.** The first leg's, unchanged: VisA `candle` and `pcb1`, identity prepared input at 448 × 448,
target class `defect`, `few_shot` splits under seeds {0, 1, 2} — so the same reference draws — and every
other sample a query. Each method runs twice on each split: at its shipped defaults, which is `calibration`
`none`, and with `calibration` `leave_one_out`, every other field at its default. The fitted model is the
same in both — bank, prototypes or colour models — so the pair differs only in the scale. Shot counts are
k ∈ {5, 10}: the primary, and the most references a scale can be fitted on. At k = 1 the two variants are
the same run, since one reference cannot be left out. 2 classes × 2 shot counts × 3 seeds × 3 methods × 2
variants is 72 runs, one child process each; at the 102 s a run measured below, about 2 h. The `none` runs
repeat the first leg's cells and are the check that nothing else moved.

**Reported.** Per method, variant and shot count, the first leg's columns, pixel average precision and
present-image recall (the share of images showing the class whose mask touches it), each a mean over the
two classes and three seeds; and the fitted scales.

**Decision rule, fixed before the run.** Per method, at 5 shots:
- `leave_one_out` becomes the method's default if all three hold:
  - the false-positive rate on absent images falls by at least 0.20 against `none`;
  - foreground IoU falls by no more than 0.005, a tolerance well inside the first leg's seed spread;
  - present-image recall keeps at least half of its `none` value.
  Otherwise `none` stays its default and `leave_one_out` ships as an option.
- Two checks of construction, not part of the rule. The scale is strictly increasing and is applied to the
  presence score after the score is computed, so presence ROC-AUC must be identical between the variants.
  It is applied to the whole map, so pixel average precision may move only by the evaluator's fixed
  histogram bins; a change above 0.01 would be a defect in the implementation, not a result.
- The leg does not re-decide the default few-shot method.

**What preceded the rule.** One smoke cell — `candle`, 2 shots, seed 0, all three methods in both variants
— ran before the rule was fixed, to prove the script. In it every calibrated map stayed below 0.5
everywhere: absent-image FPR fell from 1.0 to 0 and foreground IoU to 0. `fss_dino` and `color_prototype`
were already below the IoU tolerance, so an IoU condition alone would have adopted a scale that draws
nothing; the recall condition was added for that reason, and the smoke cell is not part of the leg. The
same cell moved `color_prototype`'s presence ROC-AUC by 0.0007, because the logit's clip merged nearly
saturated scores into ties; the clip is now float64's own, so the construction check above holds.

**Result.** Means over both classes and three seeds; the spread is the IoU's across seeds. The `none` rows
reproduce the first leg's 5- and 10-shot cells to the reported digit.

| Method | Shots | Calibration | Foreground IoU | ± seeds | Boundary F1 | Pixel AP | Presence ROC-AUC | Absent FPR | Present recall |
|---|---|---|---|---|---|---|---|---|---|
| `color_prototype` | 5 | `none` | 0.0026 | 0.0002 | 0.0021 | 0.011 | 0.301 | 1.000 | 0.960 |
| | | `leave_one_out` | 0.0001 | 0.0001 | 0.0004 | 0.010 | 0.301 | 0.211 | 0.068 |
| | 10 | `none` | 0.0025 | 0.0001 | 0.0022 | 0.015 | 0.318 | 1.000 | 0.974 |
| | | `leave_one_out` | 0.0000 | 0.0000 | 0.0002 | 0.014 | 0.318 | 0.212 | 0.022 |
| `fss_dino` | 5 | `none` | 0.0037 | 0.0010 | 0.0015 | 0.104 | 0.724 | 1.000 | 0.979 |
| | | `leave_one_out` | 0.0870 | 0.0300 | 0.0737 | 0.104 | 0.724 | 0.120 | 0.372 |
| | 10 | `none` | 0.0035 | 0.0004 | 0.0010 | 0.098 | 0.721 | 1.000 | 0.998 |
| | | `leave_one_out` | 0.0642 | 0.0165 | 0.0758 | 0.098 | 0.721 | 0.047 | 0.332 |
| `proto_seg` | 5 | `none` | 0.0357 | 0.0253 | 0.0121 | 0.153 | 0.820 | 0.926 | 0.911 |
| | | `leave_one_out` | 0.0570 | 0.0344 | 0.0617 | 0.153 | 0.820 | 0.001 | 0.188 |
| | 10 | `none` | 0.0216 | 0.0194 | 0.0075 | 0.154 | 0.849 | 0.955 | 0.965 |
| | | `leave_one_out` | 0.0518 | 0.0178 | 0.0976 | 0.154 | 0.849 | 0.001 | 0.185 |

By class at 5 shots, `leave_one_out` against `none`:

| Method | Class | Foreground IoU | Absent FPR | Present recall |
|---|---|---|---|---|
| `fss_dino` | `candle` | 0.0035 → 0.174 | 1.000 → 0.239 | 1.000 → 0.726 |
| | `pcb1` | 0.0040 → 0.0002 | 1.000 → 0.001 | 0.958 → 0.018 |
| `proto_seg` | `candle` | 0.067 → 0.114 | 0.851 → 0.002 | 0.954 → 0.358 |
| | `pcb1` | 0.0041 → 0.0002 | 1.000 → 0.000 | 0.867 → 0.018 |

The fitted scales move the cut up the unscaled map: 0.5 falls at an unscaled probability of 0.82–1.00 for
the DINO methods and above 0.999 for `color_prototype`, with slopes 0.4–4.0 and 0.14–0.77.

Both construction checks hold: presence ROC-AUC is identical between the variants in every cell, and pixel
average precision moves by at most 0.0004.

**Verdict, by the rule.** Absent-image FPR falls by 0.79 (`color_prototype`), 0.88 (`fss_dino`) and 0.92
(`proto_seg`), and foreground IoU falls by no more than the tolerance for any method — it rises by 0.083
for `fss_dino` and 0.021 for `proto_seg`. But present-image recall keeps 7 %, 38 % and 21 % of its `none`
value, under the half the rule asks for, so **`none` stays every method's default** and `leave_one_out`
ships as an option.

**What the leg says beyond its rule.** A scale fitted on the references means what it says: a pixel at 0.5
is as likely the class as not, and on a defect that covers a fraction of a percent of the frame almost no
pixel is. The cut then draws the class only where a method is confident, which is what the rise in IoU
and boundary F1 on `candle` is — and draws nothing on `pcb1`, where no method is confident, so the mean
recall falls with it. Calibration answers the question the backlog asked (the cut now means the same thing
on every class) and shows that on VisA the honest answer at 0.5 is mostly "absent". The recall condition
is what held it back, and it was added to reject a scale that draws nothing; on `candle` the scale draws
something useful, on `pcb1` it does not.

## Few-shot segmentation on FSS-1000 — predeclared, not yet run

The cross-domain gate the VisA verdict calls for: object classes rather than defects. Predeclared here
before any run; `scripts/few-shot-public-gate.py --benchmark fss1000`. 540 runs, one child process each;
at an estimated 30 s a run (200 queries, against the 1 100 of a VisA run), roughly 4–5 h on MPS.

**Data.** [FSS-1000](https://github.com/HKUSTCV/FSS-1000) (Li et al., CVPR 2020): 1 000 classes of ten
224 × 224 images, each with a foreground mask. The upstream archive is `fewshot_data.zip`, SHA-256
`4e49282322891eae4511153f1b63f6274042702ecd2b6da4bff949e106ac20e0`; no licence is published with it
([README](../README.md#public-reference-data)).

**The panel is a rule, not a choice by eye.** The 240 classes the authors hold out for testing
(`fss_test_set.txt` in the upstream repository), sorted by name, sampled by `evenly_spaced(240, 20)`:
`abe's_flyingfish`, `banana_boat`, `bucket`, `chalk_brush`, `clam`, `diver`, `electronic_stove`,
`flying_snakes`, `hair_razor`, `jet_aircraft`, `little_blue_heron`, `moist_proof_pad`, `oriole`,
`poached_egg`, `rally_car`, `sealion`, `spinach`, `tiltrotor`, `wandering_albatross`, `wooden_spoon`.
The script re-derives the list from `fss_test_set.txt` when it sits beside the data, and refuses to run on
a panel that differs. Two known upstream defects stay out: a stray `.jpeg` beside a paired `.jpg` in
`banana_boat` and `wandering_albatross` is not imported, and `peregine_falcon`'s one broken mask is not in
the panel.

**Protocol.**
- Each panel class is its own dataset, registered as the reference pack registers it
  ([import](architecture/import.md#reference-packs)). Its ten images, with their masks, are the target
  class `defect`; the other nineteen classes' 190 images are confirmed absences.
- `few_shot` splits draw k ∈ {1, 2, 5} references among the ten under seeds {0, 1, 2}, and every other
  image is a query: 10 − k that show the class and 190 that do not. Ten shots would leave no image of the
  class to segment, so k = 10 is not run.
- Identity prepared input at 448 × 448, as on VisA, so the DINO encoders see a 32 × 32 patch grid rather
  than the 16 × 16 of the native size.
- Three methods at their shipped defaults on the same pixels: `color_prototype`, `fss_dino` (DINOv2
  ViT-B/14) and `proto_seg` (DINOv2 ViT-B/14), all with `calibration` at `none`.
- 20 classes × 3 shot counts × 3 seeds × 3 methods is 540 runs.

**Reported.**
- Per method and shot count: pixel average precision, foreground IoU, boundary F1, presence ROC-AUC,
  the false-positive rate on absent images, present-image recall and ms per image. Each is a mean over
  the twenty classes and three seeds, with the spread across seeds of pixel AP and of IoU.
- Per class at 5 shots: pixel AP for each method, so a lead that is one class's alone is visible.
- The share of query pixels that show the class, which is pixel AP's chance level: about 0.011 at 1 shot
  and 0.006 at 5, from a mean foreground of 24 % of the frame (11–41 % by class).

**Why the primary is pixel AP and not IoU.** On VisA the fixed `>= 0.5` cut decided the IoU as much as
the segmentation did. Here, at 5 shots, five present queries stand against 190 absent ones, and the pooled
IoU counts every pixel drawn on an absent image, so it would again measure the cut. Pixel AP ranks the
stored foreground probability against the truth over the same queries, present and absent, without a cut.

**Decision rule, fixed before any run.** The primary number is pixel average precision at 5 shots.
- `proto_seg` stays the default few-shot method unless both hold:
  - `fss_dino` beats it by at least 0.02 on the primary;
  - `fss_dino`'s presence ROC-AUC at 5 shots is no more than 0.02 below `proto_seg`'s.
  Then `fss_dino` becomes the default and `proto_seg` is experimental. This is the VisA rule, pointed the
  other way.
- The DINO methods carry across domains only if each beats `color_prototype` by at least 0.02 on the
  primary.
- The gate does not re-decide calibration.

**Not comparable with published FSS-1000 numbers.** Published figures are a mean IoU over present queries
alone. Here:
- every pooled metric includes absent queries;
- a mask is foreground wherever it is non-zero, like every imported mask, where the usual protocol cuts at
  128. FSS-1000's masks are anti-aliased, and over the whole dataset the soft edge adds 3.8 % to the
  foreground; for a thin object it adds more than half again (up to 58 % on a `flying_snakes` image);
- the panel is 20 of the 240 test classes, and a class's references and queries are drawn from its own ten
  images.

**What preceded the rule.** One smoke cell — `bucket`, 1 shot, seed 0, `color_prototype` only, torch-free
— ran to prove the harness. It is not part of the gate.

**Result.** Not yet run.

## Supervised segmentation — neither pixel sampling alone promoted `dino_linear_seg`; neither method drew a usable defect mask

The first supervised segmentation gate (ADR-0039), in two legs, each predeclared before it ran. The legs
share the protocol and the decision rule; they differ only in how `dino_linear_seg` samples its training
pixels: `raster` in the first, `per_class` in the second. Neither leg promoted it, and `raster` measured
higher on the primary. Its defaults and maturity are now decided by the logit-bias gate below, which
promoted it under `per_class` sampling.

### Leg 1 — `raster` pixel sampling

`scripts/semantic-public-gate.py`, 12 runs in 12.5 min on MPS (torch 2.13.0, timm 1.0.28, numpy 2.5.1,
Pillow 12.3.0).

**Protocol.** VisA `candle` and `pcb1`, identity prepared input at 448 × 448, which both DINO patch sizes
divide. The dataset is read as a semantic segmentation benchmark of one class, `defect`: VisA's pixel
masks are the imported ground truth a supervised run reads through its label targets, and each normal
sample answers with an empty map, exactly as the few-shot gate reads them. For each class, a
`class_stratified` split at its shipped defaults (70 % train, stratified by class signature) is drawn
under seeds {0, 1, 2} over all samples — 1 000 or 1 004 normals and 100 defects, so about 70 defects and
700 normals train and 30 and 300 test. Two methods run at their shipped defaults on the same pixels, with
the method seed equal to the split seed where the method has one: `color_classifier` (the floor; it
draws nothing at random) and `dino_linear_seg` (DINOv2 ViT-B/14, last block). That is 12 runs, one child
process each, scored on the test subset.

**Reported.**
- Per method and class, as a mean over seeds with the spread across seeds: mean IoU, the background's IoU,
  pixel accuracy, mean class accuracy, frequency-weighted IoU, ms per image and seconds to fit.
- Per-sample outcomes of the test subset (hit, low IoU, miss for defect samples; correct absence or false
  presence for normal ones), pooled over seeds.

**Decision rule, fixed before the run.** The primary number is test mean IoU as the evaluator defines it —
over the annotation classes, background excluded, so here the IoU of `defect` — averaged over the three
seeds, per class.
- `dino_linear_seg` leaves experimental (`supported`) if it beats `color_classifier` by at least 0.05 on
  the primary **on both classes**. Otherwise it stays experimental.
- The margin is absolute, not relative, because the floor may sit near zero, where any ratio is large. It
  is 0.05 rather than the few-shot gate's 0.02 because one class and 30 test defects make a seed's draw
  move IoU by more; and it must hold on both classes because the few-shot gate's lead came from one class
  alone.
- `color_classifier` is the floor and stays experimental whatever the result.

**Result.** Test subset, means over three seeds; the spread is across seeds. Outcomes are samples pooled
over the three seeds (90 defect and 900 or 903 normal samples per class).

| Class | Method | Mean IoU | ± seeds | Background IoU | Pixel accuracy | Mean class accuracy | ms/image | Fit s |
|---|---|---|---|---|---|---|---|---|
| `candle` | `color_classifier` | 0.0007 | 0.0002 | 0.647 | 0.647 | 0.942 | 32 | 8 |
| | `dino_linear_seg` | 0.0829 | 0.0208 | 0.998 | 0.998 | 0.834 | 83 | 67 |
| `pcb1` | `color_classifier` | 0.0038 | 0.0007 | 0.853 | 0.854 | 0.805 | 33 | 8 |
| | `dino_linear_seg` | 0.0388 | 0.0097 | 0.984 | 0.984 | 0.876 | 88 | 59 |

| Class | Method | Defects: hit / low IoU / miss | Normals: correct absence / false presence |
|---|---|---|---|
| `candle` | `color_classifier` | 0 / 88 / 2 | 0 / 900 |
| | `dino_linear_seg` | 2 / 84 / 4 | 80 / 820 |
| `pcb1` | `color_classifier` | 0 / 90 / 0 | 0 / 903 |
| | `dino_linear_seg` | 6 / 78 / 6 | 0 / 903 |

Peak RSS: 0.17 GB (`color_classifier`), 1.9–2.0 GB (`dino_linear_seg`).

**Verdict, by the rule.** `dino_linear_seg` leads `color_classifier` by 0.082 mean IoU on `candle` and by
0.035 on `pcb1`. The margin of 0.05 holds on one class, not both, so **`dino_linear_seg` stays
experimental**.

**What the gate says beyond its rule.** Neither method draws a usable mask of a VisA defect. The floor
labels a third of `candle`'s pixels and a seventh of `pcb1`'s as defect; the deep head is far more
precise but still marks some defect on nearly every normal image, and reaches an IoU of 0.5 on 2 of 90
defect samples on `candle` and 6 of 90 on `pcb1`. The cause in the head's case is visible in its own training log: `plan_pixels`
samples labelled pixels evenly in raster order, so a class covering a fraction of a percent of each frame
gets almost nothing — 28–33 of 130 900 sampled pixels on `candle` and 93–109 on `pcb1` were defect — and
`inverse_frequency` weighting then weighs each of those few pixels 1 200–4 500 times a background one, which buys recall
(mean class accuracy 0.83–0.88) at the price of false presence everywhere. The floor, which samples each
class separately, saw 30 000–100 000 defect pixels and still could not separate defect by colour. The gate
therefore measures the sampling rule as much as the head.

### Leg 2 — `per_class` pixel sampling

Predeclared before it ran. `dino_linear_seg` split each training image's pixel budget equally among the
classes present in it, evenly spaced within each class (`pixel_sampling` `per_class`, then its shipped
default; [methods](architecture/methods.md#dino_linear_seg)).
`scripts/semantic-public-gate.py`, 12 runs in 13.1 min on MPS, same packages as leg 1. Under the plan's defaults about 770 training
images share 131 072 pixels, 170 each, so a defect image gives up to 85 defect pixels where raster
sampling gave it none or one.

**Protocol and decision rule: leg 1's, unchanged.** The same script, classes, prepared size, split
strategy and seeds, so the same splits; both methods rerun at their shipped defaults, whose only change from leg 1 was
`pixel_sampling` — `class_balancing` stays `inverse_frequency`, and every other field keeps leg 1's value.
`color_classifier` draws nothing at random and is rerun rather than reused, as a check that nothing else
moved. The primary is test mean IoU averaged over the three seeds, and `dino_linear_seg` leaves
experimental only if it beats `color_classifier` by at least 0.05 on it **on both classes**.

This is the one further leg the first leg's cause justified. Whatever it shows, the gate is not rerun
under another sampling or weighting rule to reach the margin.

**Result.** Test subset, means over three seeds, as in leg 1. `color_classifier` reproduced leg 1's quality
metrics to the reported digit, so the splits and pixels were the same.

| Class | Method | Mean IoU | ± seeds | Background IoU | Pixel accuracy | Mean class accuracy | ms/image | Fit s |
|---|---|---|---|---|---|---|---|---|
| `candle` | `color_classifier` | 0.0007 | 0.0002 | 0.647 | 0.647 | 0.942 | 33 | 8 |
| | `dino_linear_seg` | 0.0079 | 0.0016 | 0.971 | 0.971 | 0.937 | 89 | 70 |
| `pcb1` | `color_classifier` | 0.0038 | 0.0007 | 0.853 | 0.854 | 0.805 | 34 | 9 |
| | `dino_linear_seg` | 0.0112 | 0.0022 | 0.944 | 0.944 | 0.900 | 91 | 62 |

| Class | Method | Defects: hit / low IoU / miss | Normals: correct absence / false presence |
|---|---|---|---|
| `candle` | `color_classifier` | 0 / 88 / 2 | 0 / 900 |
| | `dino_linear_seg` | 0 / 90 / 0 | 0 / 900 |
| `pcb1` | `color_classifier` | 0 / 90 / 0 | 0 / 903 |
| | `dino_linear_seg` | 0 / 90 / 0 | 0 / 903 |

The head trained on 4 985–5 950 defect pixels per run, against 28–109 in leg 1. Peak RSS as in leg 1.

**Verdict, by the rule.** `dino_linear_seg` leads `color_classifier` by 0.007 mean IoU on `candle` and by
0.007 on `pcb1`, short of 0.05 on both, so **`dino_linear_seg` stays experimental**. Its lead is smaller
than in leg 1 on both classes, so `raster` stayed the default until the logit-bias gate below.

**What the leg says beyond its rule.** Sampling the defect fairly bought recall and cost precision. Mean
class accuracy rose to 0.90–0.94, but the head now labels 3–6 % of every image defect — the background IoU
fell from 0.98–1.00 to 0.94–0.97 — so every normal test image shows false presence and no defect sample
reaches an IoU of 0.5. Under `inverse_frequency` both legs train the head as if defect and background
were equally common; leg 1's few defect pixels happened to keep that boundary tight, and leg 2's
thousands do not. The sample was not what held the head back: its argmax is not calibrated to a class
that covers a fraction of a percent of the frame. Correcting that is a new question with its own gate,
not a third leg of this one.

## Supervised segmentation, logit bias — `dino_linear_seg` is supported, under `per_class` and `held_out_iou`

The second supervised segmentation gate, predeclared before it ran. `scripts/semantic-public-gate.py --gate
bias`, 18 runs in 22.5 min on MPS (torch 2.13.0, timm 1.0.28, numpy 2.5.1, Pillow 12.3.0). The first gate's two legs showed that
`dino_linear_seg`'s argmax is not calibrated to a small class: under `inverse_frequency` the head answers
as if defect and background were equally common. This gate asks whether a constant per class, added to
the logits at prediction and fitted for IoU on held-out folds of the training images (`logit_bias`
`held_out_iou`; [methods](architecture/methods.md#dino_linear_seg)), makes the head draw a usable mask.

**Protocol.** The first gate's, unchanged: VisA `candle` and `pcb1`, identity prepared input at 448 × 448,
one class `defect` from VisA's pixel masks, `class_stratified` splits at their shipped defaults under seeds
{0, 1, 2} — so the same splits — and the method seed equal to the split seed. Three runs per split, one
child process each, scored on the test subset:
- `color_classifier` at its shipped defaults, the floor;
- `dino_linear_seg` with `pixel_sampling` `per_class` and `logit_bias` `none` — the second leg's
  candidate, rerun as the pair's control;
- `dino_linear_seg` with `pixel_sampling` `per_class` and `logit_bias` `held_out_iou`, the candidate.

Every other field keeps its shipped default (`class_balancing` `inverse_frequency`). `per_class` because
the fitted constant is only as good as the held-out defect pixels it is fitted on, and raster sampling
gave the first leg 28–109 of them per run. 2 classes × 3 seeds × 3 runs is 18 runs, about 25 min.

**Reported.** The first gate's columns and per-sample outcomes, and the fitted constant and the held-out
IoU it reached, per run.

**Decision rule, fixed before the run.** The first gate's: the primary is test mean IoU (the IoU of
`defect`) averaged over the three seeds, per class.
- `dino_linear_seg` under `held_out_iou` leaves experimental (`supported`) if it beats `color_classifier`
  by at least 0.05 on the primary **on both classes**; `per_class` and `held_out_iou` then become its
  defaults.
- Otherwise it stays experimental. `per_class` and `held_out_iou` still become its defaults if the
  candidate's primary is above the shipped default's on both classes — `raster`, no bias: 0.0829 on
  `candle` and 0.0388 on `pcb1`, the first leg on the same splits. If not, the defaults are unchanged and
  `held_out_iou` ships as an option.
- Checks of construction, not part of the rule. `color_classifier` reproduces the first gate's metrics,
  and the `none` runs the second leg's, to the reported digit. Within a pair the head is fitted
  identically — the folds are fitted only to choose the constant — so the pair differs by the constant
  alone.
- The gate is not rerun under another sampling, weighting or bias to reach the margin.

**What preceded the rule.** One smoke cell — `candle`, seed 0 — ran the correction first considered,
`logit_bias` `training_prior`: the standard logit adjustment, which restores the training images' class
prior and so makes the argmax the Bayes answer for pixel accuracy. Its shift on `defect` was −8.30 (the
defect covers 0.025 % of the training pixels), and it drew no defect pixel on any test image: IoU 0.0000
against 0.0067 for the same head uncorrected and 0.0006 for the floor. The argmax of a calibrated
posterior is the wrong decision for the IoU of a class that rare, which is why the candidate fits its
constant for IoU instead; `training_prior` ships as an option. The cell is not part of the gate. A second
smoke, on VisA `macaroni1` seed 0 — outside the gate, so it looked at none of its cells — proved the
script and the held-out fit on real pixels before the rule was fixed: a fitted constant of −4.04 on
`defect` reached a held-out IoU of 0.144 on the training frames and a test IoU of 0.151, against 0.0019
uncorrected and 0.0004 for the floor.

**Result.** Test subset, means over three seeds; the spread is across seeds. Outcomes are samples pooled
over the three seeds (90 defect and 900 or 903 normal samples per class).

| Class | Method | Mean IoU | ± seeds | Background IoU | Pixel accuracy | Mean class accuracy | ms/image | Fit s |
|---|---|---|---|---|---|---|---|---|
| `candle` | `color_classifier` | 0.0007 | 0.0002 | 0.647 | 0.647 | 0.942 | 32 | 8 |
| | `dino_linear_seg`, `none` | 0.0079 | 0.0016 | 0.971 | 0.971 | 0.937 | 87 | 69 |
| | `dino_linear_seg`, `held_out_iou` | 0.2353 | 0.0402 | 0.9995 | 0.9995 | 0.633 | 87 | 62 |
| `pcb1` | `color_classifier` | 0.0038 | 0.0007 | 0.853 | 0.854 | 0.805 | 35 | 8 |
| | `dino_linear_seg`, `none` | 0.0112 | 0.0022 | 0.944 | 0.944 | 0.900 | 90 | 61 |
| | `dino_linear_seg`, `held_out_iou` | 0.1075 | 0.0302 | 0.9985 | 0.9985 | 0.322 | 91 | 62 |

| Class | Method | Defects: hit / low IoU / miss | Normals: correct absence / false presence |
|---|---|---|---|
| `candle` | `color_classifier` | 0 / 88 / 2 | 0 / 900 |
| | `dino_linear_seg`, `none` | 0 / 90 / 0 | 0 / 900 |
| | `dino_linear_seg`, `held_out_iou` | 6 / 70 / 14 | 647 / 253 |
| `pcb1` | `color_classifier` | 0 / 90 / 0 | 0 / 903 |
| | `dino_linear_seg`, `none` | 0 / 90 / 0 | 0 / 903 |
| | `dino_linear_seg`, `held_out_iou` | 2 / 76 / 12 | 658 / 245 |

The fitted constants on `defect` were −3.50, −3.31 and −3.57 on `candle` (held-out IoU on the training
frames 0.295, 0.292, 0.306) and −3.37, −3.09 and −2.76 on `pcb1` (0.074, 0.097, 0.089); the training
prior's shift would have been −6.9 to −8.5. Per seed, test IoU was 0.220, 0.196, 0.290 on `candle` and
0.099, 0.075, 0.148 on `pcb1`. Peak RSS 2.7 GB under `held_out_iou`, against 2.0 GB without it; the fold
heads add no measurable fit time, since encoding dominates.

All three construction checks hold: `color_classifier` reproduced the first gate's metrics and the `none`
runs the second leg's to the reported digit, and within every pair the saved head weights were
bit-identical, so each pair differs by the constant alone.

**Verdict, by the rule.** `dino_linear_seg` under `held_out_iou` leads `color_classifier` by 0.235 mean IoU
on `candle` and by 0.104 on `pcb1`, at least 0.05 on both, so **`dino_linear_seg` is supported**, and
`per_class` and `held_out_iou` are its defaults. It is also above the `raster` default on both classes
(0.235 against 0.083, 0.108 against 0.039).

**What the gate says beyond its rule.** The head was never the problem: the same weights that label 3–6 %
of every image defect reach an IoU of 0.24 and 0.11 once one constant per class is chosen for IoU. The
held-out estimate on the training frames predicted the test IoU closely on `candle` (0.30 against 0.24)
and conservatively on `pcb1` (0.09 against 0.11), so the constant generalises from the folds. The mask
is usable, not good: most defect samples are still below an IoU of 0.5, a sixth are missed, and about
a quarter of normal images show some false presence. The Bayes shift is the wrong decision for this
measure: it is roughly twice the fitted constant and draws nothing.

## Detection — `dino_linear_det` stays experimental; neither method boxes a VisA defect

The first public detection gate (ADR-0039), predeclared before it ran. `scripts/detection-public-gate.py`,
12 runs in 17.0 min on MPS (torch 2.13.0, timm 1.0.28, numpy 2.5.1, Pillow 12.3.0).

**Protocol.** VisA `candle` and `pcb1`, identity prepared input at 448 × 448, which both DINO patch sizes
divide. The dataset is read as a detection benchmark of one class, `defect`: every 8-connected component
of a VisA mask is one truth box, exactly as the application resolves an imported mask's boxes, and each
normal sample has none. For each class, a `class_stratified` split at its shipped defaults is drawn under
seeds {0, 1, 2} — the supervised segmentation gates' splits, so about 70 defects and 700 normals train
and 30 and 300 test. Two methods run at their shipped defaults on the same pixels, with the method seed
equal to the split seed where the method has one: `color_detector` (the floor; it draws nothing at
random) and `dino_linear_det` (DINOv2 ViT-B/14, last block, with the segmentation head's shipped
`per_class` sampling and `held_out_iou` constant). That is 12 runs, one child process each, through the
application's own train and infer jobs and detection evaluator, scored on the test subset.

**Reported.**
- Per method and class, as a mean over seeds with the spread of the primary across seeds: AP@[.5:.95],
  AP50, AP75, recall over the ten IoU thresholds and at IoU 0.5, the F1, precision and recall at each
  run's own printed confidence cut, ms per image, seconds to fit and peak RSS.
- Truth and predicted box counts, and per-sample outcomes of the test subset at the printed cut (hit,
  miss, mixed for defect samples; correct absence or false presence for normal ones), pooled over seeds.

**Decision rule, fixed before the run.** The primary number is test AP@[.5:.95] by COCO's protocol, of
the one class `defect`, averaged over the three seeds, per class.
- `dino_linear_det` leaves experimental (`supported`) if it beats `color_detector` by at least 0.05 on the
  primary **on both classes**. Otherwise it stays experimental.
- The margin and the both-classes condition are the supervised segmentation gate's, for its reasons: the
  floor may sit near zero, where any ratio is large, so the margin is absolute; about 30 test defects a
  seed make a draw move the number by more than a few hundredths; and a lead on one class alone is what
  the first segmentation gate and the few-shot gate showed. AP is on the same 0–1 scale as IoU, and
  demands more of a box than IoU does of a mask, so the margin is not loosened for it.
- Checks of construction, not part of the rule: within a class and seed, both methods are read against
  identical truth box counts and the same labelled test images.
- `color_detector` is the floor and stays experimental whatever the result. The gate is not rerun under
  other fields to reach the margin.

**What preceded the rule.** One smoke cell ran on VisA `macaroni1`, seed 0 — outside the gate, so it
looked at none of its cells — to prove the script and estimate its runtime: 2.8 min for both methods,
so about 15 min for the gate. Both methods sat near zero there: AP@[.5:.95] 0.0000 for the floor and
0.0006 for `dino_linear_det` (AP50 0.006), against 96 truth boxes on 30 defect samples. It did not change
the rule. It did show how the printed cut reads when a run matches almost nothing: the F1-optimal
confidence is then the highest one, so almost every detection is dropped and every normal sample reads
as a correct absence. The per-sample outcomes are reported, not decided on.

**Result.** Test subset, means over three seeds; the spread of AP@[.5:.95] is across seeds. Recall is
averaged over the ten IoU thresholds; F1 is at each run's own printed cut. Truth and predicted boxes, and
outcomes, are pooled over the three seeds (90 defect and 900 or 903 normal samples per class).

| Class | Method | AP@[.5:.95] | ± seeds | AP50 | AP75 | Recall | Recall at 0.5 | F1 at cut | ms/image | Fit s |
|---|---|---|---|---|---|---|---|---|---|---|
| `candle` | `color_detector` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 177 | 9 |
| | `dino_linear_det` | 0.0036 | 0.0020 | 0.0165 | 0.0008 | 0.009 | 0.029 | 0.048 | 86 | 71 |
| `pcb1` | `color_detector` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.000 | 0.000 | 124 | 13 |
| | `dino_linear_det` | 0.0001 | 0.0001 | 0.0001 | 0.0000 | 0.004 | 0.009 | 0.006 | 110 | 65 |

| Class | Method | Truth / predicted boxes | Defects: hit / miss / mixed / extra box | Normals: correct absence / false presence |
|---|---|---|---|---|
| `candle` | `color_detector` | 673 / 4 550 | 0 / 90 / 0 / 0 | 897 / 3 |
| | `dino_linear_det` | 673 / 896 | 10 / 37 / 40 / 3 | 855 / 45 |
| `pcb1` | `color_detector` | 241 / 23 360 | 0 / 87 / 3 / 0 | 903 / 0 |
| | `dino_linear_det` | 241 / 3 927 | 0 / 36 / 54 / 0 | 574 / 329 |

Per seed, `dino_linear_det`'s AP@[.5:.95] was 0.0064, 0.0019, 0.0025 on `candle` and 0.0001, 0.0000,
0.0000 on `pcb1`. Its fitted constants on `defect` were −3.21, −3.42, −3.46 on `candle` (held-out IoU of
the painted box interiors 0.319, 0.294, 0.303) and −4.28, −1.75, −1.35 on `pcb1` (0.085, 0.091, 0.086).
Peak RSS 0.16–0.19 GB (`color_detector`), 2.7 GB (`dino_linear_det`). Both construction checks hold: in
every class and seed the two methods were read against the same truth box count and the same 330 or 331
labelled test images.

**Verdict, by the rule.** `dino_linear_det` leads `color_detector` by 0.004 AP@[.5:.95] on `candle` and by
0.0001 on `pcb1`, short of 0.05 on both, so **`dino_linear_det` stays experimental**.

**What the gate says beyond its rule.** Neither method boxes a VisA defect. The floor matched no truth box
at IoU 0.5 in any run, while drawing about 5 boxes an image on `candle` and 24 on `pcb1`. The deep head
finds something on `candle` — 10 of 90 defect samples are hits at the cut — and almost nothing on `pcb1`,
where its extra boxes reach a third of the normal images. The truth is part of the reason. Over all 100
defect samples of each class, a VisA mask boxed by its components gives 7.1 boxes a defect image on
`candle` (median 1, at most 95) and 3.0 on `pcb1` (median 2, at most 25), and 78 % of `candle`'s boxes and
31 % of `pcb1`'s cover less than 0.01 % of the frame — about 20 pixels at 448 × 448, five times
`min_area`. Every such speck is a box a detector must find at IoU 0.5 or more, and a component detector
that draws one box around a cluster matches one of them at most. The same head that draws a usable
`candle` mask (IoU 0.24, the logit-bias gate above) scores 0.004 AP here, so this gate measures VisA's
masks read as boxes as much as it measures the features; it does not say whether frozen DINO features
are worth a box-regression head.

## Detection on box-drawn truth (PCB) — predeclared, not yet run

The gate the VisA verdict calls for: truth that annotators drew as boxes, so no box is a speck of a mask.
Predeclared here before any run; `scripts/detection-public-gate.py --benchmark pcb`. Six runs, one child
process each, estimated at 1–1.5 h on MPS (below).

**Data.** [PKU-Market-PCB](https://robotics.pkusz.edu.cn/resources/datasetENG/) (Huang and Wei,
arXiv:1901.08204): 693 images of printed circuit boards, each carrying one of six kinds of synthesised
defect — `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`, 115 or 116
images each — and a Pascal VOC file of its boxes. The archive read is `PCB_DATASET.zip`, 2 011 970 644
bytes, SHA-256 `7e398bb5828c7422e173e1ecd880adda5ce233f1fa4e2f79bcd21bebc9c13bc6`; no licence is published
with it ([README](../README.md#public-reference-data)). Its `rotation/` copies and `PCB_USED/` templates
are not part of the benchmark. Over the 693 files:
- 2 953 boxes, 482–503 a class; 1–6 an image (median 5); no image shows two classes, and one pair of
  boxes overlaps.
- Boxes are 25–284 px wide (median 65 × 64) on boards of 2240 × 2016 to 3056 × 2464 px. The median box
  covers 0.07 % of its image, and the largest 0.8 %.

**Why 1120 × 896, and what it costs.** The prepared frame is an identity profile, the whole board
contain-resized and edge-padded, which scales a board by 0.354–0.444. That puts the median box at about
25 × 25 prepared pixels — 1.8 DINOv2 patches a side — and the smallest at 9. At the VisA gates' 448 × 448
the median box would be 11 px, under one 14-px patch, and 2 657 of the 2 953 boxes would have a side
under a patch; at 1120 × 896 that is 232. The frame is divisible by 14 and 16 and near the boards' median
aspect, so it pads little. The cost: 80 × 64 = 5 120 patch tokens against 1 024 at 448², five times the
MLP work and about twenty-five times the attention, so a `dino_linear_det` image is estimated at
0.6–1 s where it was about 0.1 s. Tiling the board would keep the full resolution, but no region profile
tiles, and a centre crop would drop truth outside it. A smaller frame than 1120 × 896 is not tried if this
one fails on memory: the gate is then re-declared, not rerun.

**Protocol.**
- The dataset is registered as the reference pack registers it ([import](architecture/import.md#box-truth-a-pack-ships)):
  every VOC box, 1-based and inclusive, becomes a pixel-edge box of its class in each image's first
  revision. Classes pinned by a run are the taxonomy, `defect` then the six kinds; `defect` has no truth
  and stays out of every mean, as the evaluator leaves a class without truth out.
- `class_stratified` splits at their shipped defaults (train share 0.7) are drawn under seeds {0, 1, 2}.
  Each image shows one kind, so the six signatures are the six kinds, and each split trains on 485 images
  and tests on 208, about 81 and 35 of each kind.
- Two methods at their shipped defaults on the same pixels, the method seed equal to the split seed where
  the method has one: `color_detector` (the floor) and `dino_linear_det` (DINOv2 ViT-B/14, last block,
  `per_class` sampling, `held_out_iou`). 3 seeds × 2 methods is 6 runs, through the application's own
  train and infer jobs and detection evaluator, scored on the test subset.

**Reported.** Everything the VisA detection gate reports, and per class the AP@[.5:.95] and AP50 of each
method, as a mean over seeds.

**Two metrics, two questions.** `dino_linear_det` boxes each connected region of the head's
argmax over logits upsampled from a 14-px grid, so a box edge lands within about half a patch — 7 prepared
pixels, a quarter of the median box — of where the logits change. That caps its IoU on a 25-px box well
below 1 however good the features are, and a box-regression head is exactly the part that would lift it.
So AP50, which asks whether a detection finds and names the defect, answers the backlog's question —
are the frozen features worth a regression head — and AP@[.5:.95], the application's headline, which
also asks how tightly, decides the method's maturity, as on VisA.

**Decision rule, fixed before the run.** Each question passes by the same test on its own metric, test AP
averaged over the six classes and then over the three seeds: `dino_linear_det` beats `color_detector` by
at least 0.05, **and** its per-class AP (mean over seeds) is above the floor's on at least four of the six
classes.
- **On AP50:** if it passes, a box-regression head on the frozen DINO features is worth building, and
  goes on the backlog. If not, the features do not find PCB defects enough better than colour to justify
  one.
- **On AP@[.5:.95]:** if it passes, `dino_linear_det` leaves experimental (`supported`). If not, it stays
  experimental. The VisA gate's verdict does not count against it: that section itself found it measured
  VisA's masks read as boxes.
- The margin is the VisA detection gate's and the supervised segmentation gates', for their reasons:
  absolute because the floor may sit near zero, and 0.05 because a draw moves the number by a few
  hundredths. The four-of-six condition takes the place of VisA's both-classes one: one dataset here, and
  a lead that is one class's alone should not carry the mean.
- Checks of construction, not part of the rule: within a seed, both methods are read against identical
  truth box counts and the same 208 test images.
- `color_detector` is the floor and stays experimental whatever the result. The gate is not rerun under
  other fields or another frame to reach the margin.

**Not comparable with published PKU-Market-PCB numbers.** Published figures are commonly VOC mAP at IoU
0.5, often from detectors trained on 600 × 600 crops of the full-resolution boards (TDD-Net's augmented
set). Here the whole board is downscaled by 0.35–0.44, the split is this seeded one, and AP is COCO's.

**What preceded the rule.** One smoke cell — `color_detector`, seed 0, on a scratch copy of two boards of
each kind (12 images, six train and six test), torch-free — ran to prove the registration and the harness.
It is not part of the gate. Its inference took about 1.9 s an image at 1120 × 896; from that and the VisA
gate's timings, the six runs are estimated at 1–1.5 h.

**Result.** Not yet run.

## Anomaly-map storage

Whether a run's anomaly maps should be stored in a smaller form than a projected float32 `.npy` per
image. Predeclared before it ran; `scripts/map-storage-measure.py`.

**Protocol.** VisA `candle`, official split, the 200-image test subset, prepared at 256 × 256 under two
region profiles: `identity` (deciding) and `foreground_threshold` (reported, so a real crop and the NaN it
leaves outside are exercised). `pixel_reference` at its defaults scores each through the application's own
train and infer jobs — numpy only, so the measurement needs no torch. While the infer job runs,
`InferContext.write_map` is wrapped to keep a copy of each map in the prepared frame; before anything
else, projecting every copy through its pinned transform must reproduce the stored `.npy` bit for bit.
Each candidate is then written from those arrays and read back:

- **(a)** the source-frame map, float32 `.npy` — the format older runs hold;
- **(b)** the same array in `np.savez_compressed`, as float32 and as float16;
- **(c)** the prepared-frame map with its pinned `SpatialTransform` in one `.npz`, projected on read —
  stored plain and compressed.

**Reported, per format:** mean bytes on disk per map; write ms (encode and save — the projection every
format still pays at write time, for the run's display range and each map's peak, is reported once); read
ms, warm, decoding to the source-frame float32 array; overlay ms, decode plus `render_anomaly_map` at the
source size (the heatmap route's work) on 20 evenly spaced images; the evaluator's wall time and
`tracemalloc` peak over the whole run; how many decoded maps are bit-identical to (a); and whether the
evaluator's metrics are identical to (a)'s. The evaluator runs on each format by repointing
`image_result.map_path` at its files, and every read goes through the decoder a consumer would use:
`map_files.read_map`, which reads all five, or for the decisive run below, a `numpy.load` shim doing the
same.

**Why float16 is held to identity, not a tolerance.** Its 11-bit significand rounds a value by up to
2⁻¹¹ ≈ 4.9 × 10⁻⁴ of itself. The pixel curves bin scores into 65 536 bins over the run's own range, and
the overlay's threshold renderings compare against a user's cut; a value that close to a bin edge or a
cut moves across it, so no tolerance can be proven to leave every bin, curve point and region unchanged.

**Decision rule, fixed before the run.** A candidate is eligible only if, on the deciding leg:
1. every decoded map is bit-identical to (a) and the evaluator's metrics are identical — on the reported
   leg too;
2. the overlay's median is at most **1.5×** (a)'s;
3. the evaluator's wall time is at most **2×** (a)'s and its traced peak at most **1.5×**, so evaluation
   stays constant-memory;
4. its mean map is at most **half** of (a)'s bytes — a smaller saving does not pay for a second format.

The eligible candidate with the fewest bytes per map is adopted, with backward-compatible reading of
existing `.npy` maps. None eligible: storage stays as it is.

**Result.** numpy 2.5.1, Pillow 12.3.0; source maps 1284 × 1168. Re-projecting every prepared copy
reproduced its stored map bit for bit on both legs. Read, overlay and evaluator times are warm (page
cache), medians over three passes; the peak is `tracemalloc`'s over one evaluator pass.

| Leg | Format | Bytes / map | Write ms | Read ms | Overlay ms | Evaluate s | Peak MB | Maps identical |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `identity` | (a) `.npy` float32 | 5 998 976 | 1.19 | 0.31 | 69.9 | 2.28 | 60.4 | — |
| | (b) compressed float32 | 5 192 870 | 157.7 | 10.95 | 81.7 | 6.60 | 60.4 | 200 / 200 |
| | (b) compressed float16 | 1 758 148 | 57.9 | 4.69 | 76.2 | 4.05 | 62.3 | 0 / 200 |
| | (c) prepared + transform | 262 919 | 0.13 | 1.66 | 73.1 | 2.87 | 60.4 | 200 / 200 |
| | (c) compressed | **229 278** | 6.67 | 2.16 | 73.8 | 3.06 | 60.4 | 200 / 200 |
| `threshold` | (a) `.npy` float32 | 5 998 976 | 5.81 | 0.34 | 44.7 | 2.59 | 55.3 | — |
| | (b) compressed float32 | 752 383 | 23.9 | 2.68 | 47.2 | 3.52 | 55.3 | 200 / 200 |
| | (b) compressed float16 | 331 396 | 12.4 | 1.49 | 45.9 | 3.04 | 55.3 | 0 / 200 |
| | (c) prepared + transform | 262 918 | 0.25 | 0.47 | 44.8 | 2.63 | 55.3 | 200 / 200 |
| | (c) compressed | 229 521 | 6.84 | 0.95 | 45.3 | 2.82 | 55.3 | 200 / 200 |

Re-run with the shipped `read_map` as the reader, every ratio the rule reads stayed within 0.02 of these
and the verdict was the same. The projection every format pays at write time took 1.58 ms a map (median)
under `identity`. Every
bit-identical format gave metrics identical to (a)'s on both legs. Float16 did not: its maps moved by up to
0.062 (`identity`) and 0.125 (`threshold`), and `identity`'s pixel ROC-AUC went from 0.8892576 to
0.8892575 and its AU-PRO from 0.8074951 to 0.8074991 — small, and exactly the movement the rule excludes.
Compressing the source frame barely helps where a map covers the whole frame (0.87 of (a)'s bytes); it
helps under a crop only because the NaN outside it compresses.

**Verdict, by the rule.** Compressed float32 fails checks 3 and 4, float16 fails check 1. Both prepared-frame
forms pass every check; the compressed one is the smaller, at **3.8 % of (a)'s bytes** — 46 MB instead of
1.2 GB for the run — with the overlay at 1.06×, the evaluator at 1.34× and its peak unchanged, so
**prepared-frame maps with their pinned transform, `np.savez_compressed`, are adopted**, and an existing
`.npy` map still reads ([methods](architecture/methods.md#anomaly-maps)). One property moves with it: a stored map is now projected on every read, so the
source-frame array is Pillow's bilinear resize of the day rather than of the run, while the display range
and each map's peak were taken at write time; a change to Pillow's resampler would separate them, and the
bit-identity check above is the test that would show it.
