# Measurements

The numbers that still decide something: what was measured, against what protocol, and what it
settled. A method's design and behaviour are in [methods.md](architecture/methods.md); this page is
the evidence those pages rest on.

**How to read it.** Every gate was predeclared — floors fixed before the run, seeds recorded, one
child process per leg so peak memory is comparable. Everything ran on Apple Silicon, macOS 26.5.2,
Python 3.12.12, anomalib 2.6.0, torch 2.13.0, timm 1.0.28, on public data only.

## EfficientAD: ours against the baseline

The comparison ADR-0029 exists to keep honest. The protocol is VisA's **published** one-class test
set, byte-identical, with the validation holdout carved from `train` only (810 normal train, 90 val,
100 + 100 test) — so nothing here is a generated split.

The baseline leg below was measured against `efficientad_anomalib`, the anomalib wrapper; that
method is now retired from the registry (ADR-0029 changelog), so these numbers are a historical
record rather than a comparison the workbench can re-run in-app.

**The teacher is not one file, and that assumption cost a result.** Two public EfficientAD teachers
disagree on the same data:

| Sample ROC-AUC by aggregation | `anomalib` teacher | `nelson1425` teacher |
|---|---:|---:|
| `max` (the paper's) | 0.751 | **0.886** |
| top-64 mean | 0.758 | 0.888 |
| p99 | 0.765 | 0.798 |
| plain mean | 0.756 | 0.761 |

A negative result measured before this was understood was measuring the teacher, not the method.

**The step-budget curve**, read at four points off one continued trajectory rather than four
independent runs:

| Steps | Sample ROC-AUC | Pixel ROC-AUC | AU-PRO | ms/image |
|---|---:|---:|---:|---:|
| 4 000 | 0.886 | 0.981 | 0.916 | 26.5 |
| 8 000 | 0.916 | 0.989 | 0.936 | 26.6 |
| 16 000 | 0.943 | 0.992 | **0.944** | 25.5 |
| **30 000** | **0.955** | **0.994** | 0.943 | 26.4 |

Localisation converges before ranking: AU-PRO plateaus by 16 000 steps while sample ROC-AUC is still
climbing at 30 000. A run stopped when the map looks right is stopped too early for the score.

## Dinomaly — promoted

Transformer reconstruction over a frozen DINOv2 encoder. 37.42M parameters, 15.36M trainable;
resumable payload 245.9 MB, with the frozen encoder fingerprinted rather than copied into it.

Resource gate at 392 × 392: 122.4 ms per training step and 42.3 ms inference on MPS, 0.82 GiB peak
RSS — only 1.07× faster than CPU for training, so CPU is a credible fallback.

Paired public gate against a PatchCore control on the same immutable prepared pixels, VisA
`candle` and `pcb1`, seed 20260812:

| Mean over both classes | Dinomaly | PatchCore | Floor required |
|---|---:|---:|---:|
| Image ROC-AUC | **0.9634** | 0.6896 | 0.80 |
| Pixel ROC-AUC | **0.9953** | 0.9626 | 0.85 |
| AU-PRO | **0.9514** | 0.8006 | 0.60 |

Cleared all three floors. Markedly slower to fit (≈ 600 s against PatchCore's ≈ 17 s) but bounded,
Mac-feasible and resumable, and it adds a genuinely different failure mode rather than another
memory-bank variant.

## GLASS — available, not recommended

Learned anomaly synthesis: a frozen ImageNet backbone feeding a trainable projection and
discriminator. 28.80M parameters, 3.94M trainable, 49.3 MB resumable payload. Batch 8 buys no
throughput on this host, so the plugin runs at batch 1, 288 × 288.

| Mean over both classes | GLASS | PatchCore | Floor required |
|---|---:|---:|---:|
| Image ROC-AUC | 0.7938 | 0.9013 | 0.8000 |
| Pixel ROC-AUC | 0.8986 | 0.9910 | 0.8500 |
| AU-PRO | 0.6052 | 0.8762 | 0.6000 |

It missed the image floor by 0.0062 while PatchCore beat it on all three on the same pixels, so it
is kept as an explicitly experimental comparison rather than promoted. This is deliberately **not**
the upstream protocol — anomalib's reference uses 39 200 image exposures, a category-specific `svd`
and best-checkpoint selection on test evidence; the gate here uses 5 000 exposures, one generic
synthesis distribution and the final checkpoint. A follow-up may raise the budget, but must not tune
the distribution, the stopping point or the checkpoint per category.

## AnomalyVFM — gated, not yet integrated

A 1.421 GB, 355.36M-parameter adapted RADIO checkpoint: 591 ms/image at 768 px on MPS with 2.07 GiB
driver memory. Mac-credible when loaded on demand through the single-resident-worker boundary, and
768 px is kept because the project prefers reference quality over latency.

Two invariants the gate exposed and any integration must honour: anomalib downloads inside its model
constructor with `local_files_only=False`, so the wrapper must resolve, verify and then construct
offline — a cached inference must never touch the network; and anomalib reports export as
unsupported, so the portable-export contract must show AnomalyVFM as unavailable rather than relay a
no-op export as success.

## Region profiles — identity stays the default

Localising the object before detection is an experiment variable, not an assumed improvement. The
rule was fixed before the result: at least +0.01 mean pixel ROC-AUC *and* +0.01 mean AU-PRO, losing
no more than 0.02 on either, with zero build failures.

| Class | Profile | Mean crop | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | Missed defect pixels |
|---|---|---:|---:|---:|---:|---:|
| `candle` | identity | 100.0 % | 0.867 | 0.963 | 0.890 | 0 / 405 291 |
| `candle` | threshold | 14.1 % | 0.620 | 0.511 | 0.192 | 347 202 / 405 291 |
| `pcb1` | identity | 100.0 % | 0.717 | 0.982 | 0.713 | 0 / 1 398 914 |
| `pcb1` | threshold | 31.4 % | **0.834** | 0.833 | 0.456 | 351 910 / 1 398 914 |

Mean deltas: **−0.300 pixel ROC-AUC**, **−0.478 AU-PRO**. The `pcb1` image ROC-AUC gain of +0.117 is
the trap this gate exists to catch — without inverse projection and uncovered-pixel accounting it
reads as evidence *for* localisation, while the crop silently omitted a quarter of all defect pixels.
Identity remains the default; `foreground_threshold` stays available as an explicit, previewable
choice.

Source-frame float maps cost about 1.23 GB per 200-image run regardless of crop size. That is a
storage problem ([backlog.md](backlog.md)), not a reason to change this verdict.

## DINO patch memory — promoted

A frozen DINO backbone whose patch features are the model (ADR-0037). Nothing is trained; the
15.4 MB checkpoint is a 5 000-vector coreset over 50 000 bounded candidates, and the encoder
travels as a fingerprint, not as weights. Paired public gate against a PatchCore control on the
same immutable prepared pixels, VisA `candle` and `pcb1` at 448 × 448 (the one size divisible by
both patch sizes, so every backbone row below saw identical pixels), seed 20260812, `global_knn`,
`last_two` layers, k = 1.

The recorded verdict is the **DINOv2 ViT-S/14-reg4** leg — its weights are ungated, so anyone can
reproduce it without a licence:

| Mean over both classes | DINO memory | PatchCore | Floor required |
|---|---:|---:|---:|
| Image ROC-AUC | **0.9000** | 0.8565 | 0.80 |
| Pixel ROC-AUC | **0.9921** | 0.9889 | 0.85 |
| AU-PRO | **0.9315** | 0.9246 | 0.60 |

Cleared all three floors and beat its control on all three means. 13–21 s to fit, 42 ms per image
and ≈ 1.0 GB peak RSS on MPS against PatchCore's 48 ms and 1.75 GB. Per class the split is honest:
`pcb1` is a rout (0.891 vs 0.775 image ROC-AUC), `candle` a narrow image-level loss (0.909 vs
0.938) with a pixel-level win.

**DINOv3 ViT-S/16, same recipe, licence-gated weights** (`HF_TOKEN` required, not the recorded
gate): means 0.8147 / 0.9850 / 0.8588 — clears the floors but trails DINOv2 on every metric, with
the deficit concentrated on `candle` (0.725 image ROC-AUC). At 448 px a /14 backbone sees a
32 × 32 grid where /16 sees 28 × 28; until a layer-selection sweep says otherwise, the ungated
default is also the measured best.

**Per-position ablation** (`local_knn`, r = 1, `pcb1` — the most registered class VisA offers):
image 0.847 / pixel **0.9969** / AU-PRO **0.9510** against `global_knn`'s 0.891 / 0.9952 / 0.9252
on the same pixels. Restricting the bank to positions sharpens *where* (both localisation metrics
rise) and costs a little *whether* (image ranking dips). VisA is not a registered benchmark; the
mode's real target is repeatably-fixtured data that cannot be published, and this row exists so
that claim has at least one public anchor.

## Dinomaly, ours — parity reached, wrapper retired

The in-house port was pinned to anomalib at bring-up — one training step matches at atol = 0 for
loss, every trainable gradient, the post-step weights, the eval map and the image score — so the
gate measures nothing but the divergence that training order and RNG streams accumulate over
5 000 steps. The wrapper's exact protocol: VisA `candle` and `pcb1` at 392 × 392, DINOv2
ViT-S/14-reg4 encoder, seed 20260812, paired PatchCore control on the same immutable pixels.

| Mean over both classes | Dinomaly (ours) | Wrapper's recorded run | PatchCore | Floor |
|---|---:|---:|---:|---:|
| Image ROC-AUC | **0.9636** | 0.9634 | 0.6896 | 0.80 |
| Pixel ROC-AUC | **0.9951** | 0.9953 | 0.9626 | 0.85 |
| AU-PRO | **0.9511** | 0.9514 | 0.8006 | 0.60 |

Agreement to the third decimal on every metric — the bring-up pins predicted exactly this. All
three floors cleared; ≈ 570 s per training leg against the wrapper's ≈ 610 s, 50 ms inference,
0.98 GB peak RSS. That is parity under the predeclared rule, so `dinomaly_anomalib` retires: the
implementation the workbench carries forward is the one whose encoder is configurable (any
`DinoBackbone` entry, decoder depth included) and whose every trainable line is in this
repository. The wrapper's numbers above stay as its recorded legacy row.

## SubspaceAD — defaults chosen by a sweep, gate still open

**This entry is not a gate.** Every other block on this page records a predeclared floor and a paired
control; this one records a parameter search, run outside the application by its own harness
(ADR-0038) because the alternative was unaffordable. The promotion gate `subspace_ad` has *not* run is
in [backlog.md](backlog.md), and the method ships `experimental` until it does.

**Protocol.** VisA's twelve categories under the official one-class split for three tuning phases, then
MVTec-AD's fifteen as a held-out fourth. A category is the unit of evidence: seeds fold inside a
category first, then categories average, comparisons are paired within category, ± is the standard
error over categories, and ★ marks a difference larger than twice its own. 215,874 rows over about
twelve hours of Apple Silicon. The few-shot protocol is the paper's — k ∈ {1, 2, 4} normals, each
augmented with 30 random rotations up to 345°, corners filled with black, and `transistor` excluded
from rotation as the paper excludes it.

**Why a sweep of that size was affordable**, which is the finding the budget rests on: three of the
axes are free. The basis is orthonormal, so a patch's score at rank *r* is `‖x−µ‖²` minus a prefix of
the squared coefficients — every τ is one `cumsum` off one projection. The image score is a prefix
mean of the sorted map, so every ρ is one sort. The k-shot draws are nested, so one pass snapshots
every k. Only the encoder and the input size cost a forward pass.

| Axis | Swept | Verdict | Margin |
|---|---|---|---|
| Layer window | 9 windows, at depth 12 and 24 | `upper_half` | +0.12 image AUROC over the paper's window at depth 12 |
| Window *encoding* | relative band vs fixed count | **a fraction of depth** | fixed count loses 6.5 / 1.5 / 0.0 points at ViT-S / ViT-B / ViT-L on VisA, and 3.6–4.7 ★ at depth 12 on MVTec |
| Backbone | DINOv2 and DINOv3 at S, B, L | `dinov2_vit_l14` | +0.0249 ★ over ViT-B on VisA — and **nothing at all on MVTec** |
| Prepared size | 448, 672 | 672 | a tie for detection at ViT-S/B; +0.0079 ★ at ViT-L |
| τ (`variance`) | 0.95, 0.97, 0.99, 1.0 | 0.99 | 1.0 collapses the score to float noise (0.589) |
| ρ (`tail_fraction`) | 0.001 – 0.02 | 0.002 | interior, and neither benchmark's preference is decisive |

**The leading VisA arm, phase by phase.** Every figure is a mean over twelve categories, each first
averaged over seeds.

| | Phase 3 | Phase 2 | Phase 1 |
|---|---:|---:|---:|
| Encoder | `dinov2_vit_l14` | `dinov3_vit_b16` | `dinov2_vit_s14_reg4` |
| Window | `upper_half` · 12–24 of 24 | `final7` · 6–12 | `last_four` · 9–12 |
| τ · ρ | 0.99 · 0.002 | 0.99 · 0.005 | 0.99 · 0.01 |
| Image AUROC | **0.9472 ±0.0135** | 0.9275 ±0.0215 | 0.8939 ±0.0325 |
| Pixel AUROC | 0.9869 | 0.9841 | 0.9832 |
| AU-PRO | **0.9683** | 0.9555 | 0.9448 |
| Worst category | `macaroni2` · 0.847 | `macaroni2` · 0.730 | `macaroni2` · 0.587 |

The worst category moved from 0.587 to 0.847 without a single change to the method — only to the
encoder, the window and the aggregation fraction — and the standard error over categories more than
halved, so the spread genuinely shrank rather than the mean being lifted by its easy categories.
Phase 1's leader is a near-tie the table hides: `dinov3_vit_s16` with `final_band` reaches 0.8947
±0.0307, eight ten-thousandths ahead on detection, and loses AU-PRO by 1.2 points. The DINOv2 arm is
carried forward for that reason and because its weights are ungated.

**Phase 4 held the defaults out against a benchmark they were not tuned on.** Fifteen MVTec-AD
categories, the shipped configuration unchanged:

| | MVTec-AD, 15 categories | VisA, 12 categories |
|---|---:|---:|
| Image AUROC | **0.9731 ±0.0105** | 0.9472 ±0.0135 |
| Average precision | 0.9823 | 0.9482 |
| Pixel AUROC | 0.9797 | 0.9869 |
| AU-PRO | 0.9549 | 0.9683 |
| Worst category | `screw` · 0.871 | `macaroni2` · 0.847 |

`grid`, `leather` and `metal_nut` are exactly 1.0000 and `bottle` is 0.9992; the four holding the mean
down are `screw` (0.871), `cable` (0.912), `transistor` (0.928) and `pill` (0.952). Note what
`transistor` is: the one category rotation is withheld from, because a rotated transistor is not a
normal transistor. It fits on 9,216 patches where every other category gets 285,696 — a thirty-first
as many, since the rotations *are* the augmentation — and still reaches 0.928.

Phase 4 confirmed three verdicts and overturned one:

- **The window's *encoding* got a stronger answer, not a weaker one.** At depth 24 the fixed-count
  reading loses to the relative band by **−0.0098 ★ (3–9)** on MVTec, where the same comparison on
  VisA was a tie (−0.0016, not decisive); at depth 12 it loses by **−0.0472 ★ (2–12)** on DINOv3 ViT-B
  and **−0.0364 ★ (3–11)** on DINOv2 ViT-B. Fifteen categories nobody tuned against decide what twelve
  tuned ones could not, which is the one direction this evidence could have gone that makes the
  finding more trustworthy rather than less.
- **The ρ bracket is closed and the optimum is interior.** Phase 3 ended with ρ pinned to the smallest
  value it had swept, which is the shape of a search that has not finished. Extending to 0.001 settles
  it: 0.001 is *worse* (0.9691 against 0.9731). The two benchmarks then disagree about where the
  interior optimum is, and **neither preference is decisive** — VisA prefers 0.002 by +0.0051 ±0.0037,
  MVTec prefers 0.005 by +0.0023 ±0.0013. ρ = 0.002 ships because the axis is steep where it matters
  and flat where it does not: VisA falls 2.5 points between 0.002 and 0.02 while MVTec spans 0.6
  points across the whole range, so picking 0.002 costs 0.0023 on MVTec and picking 0.005 costs 0.0051
  on VisA.
- **`upper_half` is never worse.** Against `final7` — the same window at depth 12, seven blocks against
  thirteen at depth 24 — it wins on VisA (+0.0098 ★, 10–2) and ties on MVTec (+0.0015, 5–6). It keeps
  beating both readings of the paper's own window decisively on MVTec (+0.0133 ★ and +0.0230 ★, 12–1
  and 13–1).
- **The backbone verdict is the one that did not transfer.** ViT-L beat DINOv2 ViT-B by +0.0249 ★
  (11–1) on VisA and by **−0.0002 ±0.0014 (5–7)** on MVTec — a dead tie, for roughly twice the compute.
  ViT-L stays the default because it is never worse and decisively better on one of the two, but the
  honest statement is narrower than Phase 3's: *on MVTec-like data ViT-B gives the same answer for
  half the cost*, and a user whose data resembles MVTec more than VisA should start there.
- **The few-shot curve flattens immediately.** k = 1 → 2 buys +0.0156; 2 → 4 buys +0.0023. The paper's
  k ≤ 4 is where the curve already is, not a limitation being worked around.

No default changed as a result of Phase 4, which is the outcome a held-out phase is run to be *allowed*
to report.

**The plugin reproduces the campaign.** The two share their arithmetic (`models/subspace.py`,
`models/score_map.py`) but not their plumbing: the plugin reads prepared PNG artifacts through
`load_array`, picks its fit images with `evenly_spaced`, and seeds its rotations from one integer,
where the campaign prepares from source, draws a seeded permutation and uses a per-category stream. So
this cannot be bit-exact. On VisA `candle`, k = 4:

| Arm | Plugin | Campaign, per seed | Patches | Rank |
|---|---:|---|---:|---:|
| ViT-S/14-reg4, 448 px, `upper_half`, τ 0.99, ρ 0.01 | 0.9386 | 0.9369 / 0.9246 / 0.9451 / 0.9436 / 0.9356 | 126,976 both | 91 vs 91–92 |
| ViT-L/14, 672 px, `upper_half`, τ 0.99, ρ 0.002 — **the shipped defaults** | 0.9462 | 0.9504 / 0.9428 / 0.9445 | 285,696 both | 300 vs 299–303 |

Identical patch counts, ranks inside the campaign's own range, and image AUROC 0.0014 and 0.0003 from
the campaign's mean — inside its seed spread in both cases. At the shipped defaults a fit over four
images is 120 s and inference is 649 ms/image on MPS, both measured while the MVTec phase was running
on the same machine, so both are upper bounds.

**What this does not settle.** No paired control on shared prepared pixels, which is why the maturity
label is `experimental`. Three axes were held fixed throughout because each costs a forward pass rather
than an arithmetic re-read — per-layer L2 before pooling, `concat` instead of `mean`, and
`final_norm=False` — and `rotation_fill=masked` was never measured at all. All four are in
[backlog.md](backlog.md).
