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
