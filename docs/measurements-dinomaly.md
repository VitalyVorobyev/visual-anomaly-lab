# Dinomaly measurements

The evidence log for the first M11 method candidate. Dinomaly is a transformer-reconstruction
reference: a frozen DINOv2 encoder supplies intermediate features, while a bottleneck and decoder learn
to reconstruct normal feature maps. This is a different detection principle from EfficientAD's
student–teacher distillation and PatchCore's nearest-neighbour memory bank.

## 2026-08-12 — Apple-Silicon resource gate

`scripts/dinomaly-smoke-test.py` measures the actual anomalib module before an application wrapper is
written. Every device/size leg runs in a fresh subprocess so peak resident memory is not polluted by a
previous model. Synthetic inputs are byte-identical across devices, the first step/pass is discarded as
warm-up, and the checkpoint measurement includes the trainable reconstruction network plus the
`StableAdamW` state needed to continue training.

| Variable | Fixed value |
|---|---|
| Encoder | `vit_small_patch14_reg4_dinov2`, public pretrained weights |
| Decoder | 8 layers, anomalib defaults otherwise |
| Batch | 1 synthetic RGB image, seed 1 |
| Initialisation | torch seed 0 before module construction |
| Host | Apple Silicon, macOS 26.5.2, Python 3.12.12 |
| Runtime | anomalib 2.6.0, torch 2.13.0, timm 1.0.28 |
| Encoder SHA-256 | `5f221c2cf60ece6b1569f41625f798b8af8fbf551ea33e3d962aad4e8324ec5c` |

| Device | Prepared size | Train step | Inference | Peak RSS | MPS driver memory |
|---|---:|---:|---:|---:|---:|
| MPS | 196 × 196 | 81.7 ms | 13.7 ms | 0.82 GiB | 0.52 GiB |
| CPU | 196 × 196 | 73.2 ms | 31.7 ms | 0.97 GiB | — |
| MPS | 392 × 392 | 122.4 ms | 42.3 ms | 0.82 GiB | 0.62 GiB |
| CPU | 392 × 392 | 130.7 ms | 62.1 ms | 1.13 GiB | — |

The model has 37.42 million parameters, of which 15.36 million are trainable. After one optimizer step,
the resumable reconstruction-network and optimizer payload is 245.9 MB. The external frozen encoder is
not duplicated into that payload; a checkpoint must instead name and fingerprint the exact resolved
encoder weights, as PatchCore already does.

MPS executes the complete forward, loss, backward, optimizer and inference graph. At the paper-shaped
392-pixel input it is only 1.07× faster than CPU for training, although inference is 1.47× faster. CPU is
therefore a credible fallback. The small profile is Mac-feasible without a special low-resolution mode;
quality, not resource pressure, will decide whether the base encoder deserves a later profile.

### Integration invariants found by the gate

- Prepared width and height must be divisible by the DINOv2 patch size, 14. A plugin must fail before
  training with a useful preprocessing-size message rather than letting a token reshape fail downstream.
- Anomalib validates `decoder_depth > 1`, but its default decoder fusion always indexes layers 0…7.
  Depths 2…7 therefore construct and then fail on the first batch. The first plugin fixes depth at eight;
  it will not expose a knob whose valid values depend on a hidden fusion topology.
- Anomalib's default `Resize(448) → CenterCrop(392)` preprocessor cannot run outside the shared spatial
  bridge. The wrapper must consume the already prepared pixels unchanged and apply only the encoder's
  ImageNet channel standardisation.
- Torch's global RNG initialises the trainable bottleneck and decoder. `seed` must be set immediately
  before construction and tested in both directions: same seed, same trained state; different seed,
  different state.
- The pretrained encoder is an experiment input. Downloads require an explicit policy, and saved state
  must carry the encoder name, package versions and the SHA-256 of its resolved tensor state.

**Decision:** proceed with a lazy `dinomaly_anomalib` plugin using the small registered DINOv2 encoder,
fixed depth eight, 392 × 392 as the documented reference size, finite steps, resumable optimiser state,
MPS preferred with CPU supported, and an exact public-data quality benchmark before M11 calls the method
integrated.

## 2026-08-12 — Paired public VisA quality gate

`scripts/dinomaly-public-gate.py` ran Dinomaly and PatchCore through the application's real import,
preparation, training, inference and evaluation path. The two methods received the same immutable
392 × 392 prepared pixels from the official VisA one-class split. Each method ran in a fresh child
process so peak RSS remained comparable. The gate used two structurally different public classes,
`candle` and `pcb1`, seed 20260812, and no private images.

| Method | Configuration |
|---|---|
| Dinomaly | small registered DINOv2 encoder, depth 8, batch 1, 5,000 steps |
| PatchCore | WRN-50, layer2+layer3, 256 training images, 50,000 candidates, 10% coreset |

| Class | Method | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | Train | Infer | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| candle | Dinomaly | 0.9618 | 0.9940 | 0.9541 | 611.7 s | 13.7 s | 1.13 GiB |
| candle | PatchCore | 0.7082 | 0.9437 | 0.8452 | 16.3 s | 13.6 s | 1.61 GiB |
| pcb1 | Dinomaly | 0.9649 | 0.9966 | 0.9488 | 596.0 s | 15.6 s | 1.15 GiB |
| pcb1 | PatchCore | 0.6710 | 0.9816 | 0.7560 | 16.8 s | 15.9 s | 1.61 GiB |

| Mean | Dinomaly | PatchCore | Difference |
|---|---:|---:|---:|
| Image ROC-AUC | 0.9634 | 0.6896 | +0.2738 |
| Pixel ROC-AUC | 0.9953 | 0.9626 | +0.0326 |
| AU-PRO | 0.9514 | 0.8006 | +0.1508 |

The predeclared promotion floors were mean image ROC-AUC 0.80, pixel ROC-AUC 0.85 and AU-PRO 0.60.
Dinomaly cleared all three and saved a resumable 234 MiB checkpoint per class. Its frozen encoder was
not duplicated into the checkpoint; the exact SHA-256 and anomalib, torch and timm versions were stored
and verified during reload. A real two-step MPS save/load check produced an identical score before and
after reload.

**Decision:** promote `dinomaly_anomalib` as the transformer-reconstruction reference. It is markedly
slower to fit than PatchCore, but it is bounded, Mac-feasible, resumable, and adds a useful high-quality
failure mode rather than another variation of the existing memory-bank family. GLASS is the next M11
resource gate.
