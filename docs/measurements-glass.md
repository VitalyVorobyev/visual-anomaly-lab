# GLASS measurements

The evidence log for the second M11 method candidate. GLASS is a learned-anomaly-synthesis
reference: a frozen ImageNet feature extractor feeds a trainable projection and discriminator;
local image-space Perlin anomalies and globally perturbed feature embeddings teach the
discriminator without labelled defects.

## 2026-08-12 — Apple-Silicon resource gate

`scripts/glass-smoke-test.py` measures anomalib's real GLASS module before an application
wrapper is written. Each device/size/batch leg runs in a fresh process. Seeded synthetic
pixels are byte-identical across devices; the first update and inference are warm-up passes.
The measured training update includes image-space synthesis, feature-space perturbation,
the complete 20-step inner mining loop, backward, and both optimizer steps.

| Variable | Fixed value |
|---|---|
| Backbone | `wide_resnet50_2`, public ImageNet weights |
| Feature layers | `layer2`, `layer3` |
| Mining | enabled, 20 inner steps |
| Local anomaly source | built-in Perlin synthesis; no DTD download |
| Initialisation and pixels | torch seed 20260812 |
| Host | Apple Silicon, macOS 26.5.2, Python 3.12.12 |
| Runtime | anomalib 2.6.0, torch 2.13.0, timm 1.0.28 |
| Backbone SHA-256 | `6f38d2121a24ee7e2915587e7634138fbd0dfffd4653c4f23b1d51478d340e46` |

| Device | Prepared size | Batch | Centre pass | Train update | Inference | Peak RSS | MPS driver |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPS | 144 × 144 | 1 | 165.4 ms | 74.0 ms | 12.7 ms | 1.04 GiB | 1.20 GiB |
| CPU | 144 × 144 | 1 | 29.0 ms | 117.6 ms | 18.1 ms | 1.07 GiB | — |
| MPS | 288 × 288 | 1 | 171.4 ms | 200.4 ms | 30.9 ms | 1.05 GiB | 1.19 GiB |
| CPU | 288 × 288 | 1 | 88.6 ms | 303.6 ms | 43.2 ms | 1.69 GiB | — |
| MPS | 288 × 288 | 8 | 476.8 ms | 1,633.4 ms | 240.3 ms | 1.05 GiB | 4.30 GiB |
| CPU | 288 × 288 | 8 | 318.8 ms | 2,157.3 ms | 304.2 ms | 6.35 GiB | — |

The loaded graph has 28.80 million parameters; only the 3.94 million projection and
discriminator parameters belong to optimizers. Their resumable model and optimizer payload is
49.3 MB. The external backbone must be named and fingerprinted rather than copied into every
checkpoint.

Batch 8, used in the upstream benchmark, adds no meaningful throughput on this host: the MPS
update is 8.15 times slower than batch 1 and raises driver memory from 1.19 to 4.30 GiB. The
application reference should therefore use batch 1 and finite update counts. At 288 px, 5,000
updates are approximately 17 minutes of measured update time before data loading and periodic
centre passes. CPU is a credible fallback, though about 1.5 times slower for training.

### Integration invariants found by the gate

- The feature centre is a full pass over normal training images in upstream anomalib and is
  recomputed every epoch. The application plugin must bound and print this pass, sample with
  `evenly_spaced`, and define exactly when the centre is refreshed. A hidden full-dataset pass
  is not acceptable.
- Prepared pixels must bypass anomalib's spatial preprocessor. Only ImageNet channel
  standardisation belongs inside the model wrapper.
- The pretrained WRN-50 backbone is an experiment input. Its download policy, tensor
  fingerprint and package versions must be persisted and checked on reload.
- GLASS's augmentation and feature perturbation consume torch random streams. Exact resume
  requires saving/restoring CPU and MPS RNG state, and tests in both seed directions.
- The optional Describable Textures Dataset is an explicit public asset, not an incidental
  download. Anomalib identifies `dtd-r1.0.1.tar.gz` with SHA-256
  `e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205`. The first plugin will
  default to built-in Perlin synthesis; a paired public-data gate must prove DTD adds enough
  value before the app manages or downloads that corpus.
- Upstream category-specific `svd` choices must not enter the universal plugin as dataset-name
  conditionals. Any exposed synthesis distribution is generic experiment configuration and is
  held constant across the public gate.

**Decision:** proceed to a lazy GLASS plugin with batch 1, 288 × 288 as the documented
reference size, finite updates, a bounded centre sample, Perlin-only synthesis by default,
MPS preferred with CPU fallback, and an exact paired public-data quality gate before promotion.

## 2026-08-12 — Paired public-data quality gate

`scripts/glass-public-gate.py` ran the integrated plugin and the fixed PatchCore control on
the official VisA one-class splits for `candle` and `pcb1`. Within each class both methods
received the same identity-prepared 288 × 288 pixels and the same evaluation protocol. The
destination was an isolated app-data directory; source images under `/datasets/` remained
read-only.

| Variable | GLASS value |
|---|---|
| Updates | 5,000, batch 1 |
| Feature centre | 128 evenly spaced normals, refreshed every 392 absolute updates |
| Feature mining | 20 inner steps |
| Synthesis | sample-anchored global perturbation + built-in local Perlin regions |
| Learning rate | 1e-4 |
| Prepared input | identity, 288 × 288 |
| Seed | 20260812 |
| Checkpoint selection | final fixed-budget checkpoint; no labelled validation selection |

| Class | Method | Image ROC-AUC | Pixel ROC-AUC | AU-PRO |
|---|---|---:|---:|---:|
| `candle` | GLASS | 0.8293 | 0.9122 | 0.5505 |
| `candle` | PatchCore | 0.9371 | 0.9887 | 0.9467 |
| `pcb1` | GLASS | 0.7583 | 0.8850 | 0.6598 |
| `pcb1` | PatchCore | 0.8654 | 0.9934 | 0.8058 |
| **Mean** | **GLASS** | **0.7938** | **0.8986** | **0.6052** |
| **Mean** | **PatchCore** | **0.9013** | **0.9910** | **0.8762** |
| Promotion floor | GLASS required | 0.8000 | 0.8500 | 0.6000 |

GLASS passed the pixel ROC-AUC and AU-PRO floors but missed the image ROC-AUC floor by
0.0062. PatchCore exceeded it by 0.1075 image ROC-AUC, 0.0924 pixel ROC-AUC and 0.2710
AU-PRO on the same pixels. The result therefore does **not** promote GLASS as the
learned-synthesis reference.

| Class | Method | Train | Infer | Peak RSS | Model checkpoint | Full experiment artifacts |
|---|---|---:|---:|---:|---:|---:|
| `candle` | GLASS | 19m 06s | 15.5s | 1.34 GiB | 53 MiB | 1,197 MiB |
| `candle` | PatchCore | 16.3s | 10.0s | 1.62 GiB | 29 MiB | 1,173 MiB |
| `pcb1` | GLASS | 20m 03s | 14.8s | 1.07 GiB | 53 MiB | 1,199 MiB |
| `pcb1` | PatchCore | 16.7s | 12.9s | 1.62 GiB | 29 MiB | 1,175 MiB |

The approximately 1.2 GiB experiment footprint is predominantly persisted result maps,
not GLASS state; compact source-map persistence remains a separate cross-method problem.

### Why this is not the upstream benchmark protocol

Anomalib's GLASS reference documents 100 epochs, batch 8, at most 392 samples per epoch,
category-specific `svd`, and the checkpoint with the best labelled image-plus-pixel AUROC.
That is 39,200 image exposures and selects both distribution geometry and checkpoint using
category/test evidence. The workbench gate deliberately uses 5,000 image exposures, one
generic synthesis distribution and the final checkpoint. It is a bounded, honest reference,
not an attempt to reproduce a category-tuned paper table.

**Decision:** keep `glass_anomalib` available as an explicitly experimental comparison, not
a recommended reference. A follow-up may test a larger predeclared fixed budget or one paired
DTD ablation held constant across both classes. It must not tune `svd`, the stopping point or
the checkpoint separately per dataset category.
