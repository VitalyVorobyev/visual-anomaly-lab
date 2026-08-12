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
