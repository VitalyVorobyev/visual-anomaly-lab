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
