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
