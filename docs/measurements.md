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

## AnomalyVFM — gated, not integrated

A 1.421 GB, 355.36M-parameter adapted RADIO checkpoint: 591 ms/image at 768 px on MPS, 2.07 GiB driver
memory. Verdict: Mac-credible when loaded on demand through the resident worker; 768 px kept for quality.

Invariants any integration must honour: anomalib downloads inside its model constructor
(`local_files_only=False`), so the wrapper must resolve, verify, then construct offline; and anomalib
reports export as unsupported, so the portable-export contract must show it unavailable.

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
run regardless of crop — a storage item in [backlog.md](backlog.md), not a reason to change the verdict.

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
images hold the class. Two open items follow (backlog, Few-shot segmentation): a threshold-free pixel
metric, so mask ranking is measured apart from the cut, and a calibrated foreground probability. VisA
defects are also a hard target for a method built for objects, which is why a cross-domain few-shot
dataset is the next gate.

