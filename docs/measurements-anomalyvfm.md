# AnomalyVFM measurements

The evidence log for M11's zero-shot foundation-model candidate. AnomalyVFM adapts a
pretrained RADIO vision transformer with DoRA layers, a mask decoder and an image-level
predictor using synthetic anomalies outside the workbench. Application inference needs no
normal training images, but it does need the complete adapted public checkpoint.

## 2026-08-12 — Apple-Silicon resource and asset gate

`scripts/anomalyvfm-smoke-test.py` measures anomalib 2.6.0's real implementation in fresh
processes. It resolves and hashes the public asset once, then puts every worker into Hugging
Face offline mode. Seeded synthetic pixels are byte-identical across devices; the first
inference is a warm-up and the table reports the median of the next three.

| Variable | Fixed value |
|---|---|
| Model | AnomalyVFM with adapted RADIO |
| Parameters | 355.36 million |
| Batch | 1 |
| Precision | float32 |
| Seed and pixels | 20260812 |
| Host | Apple Silicon, macOS 26.5.2, Python 3.12.12 |
| Runtime | anomalib 2.6.0, torch 2.13.0 |
| Asset repository | `MaticFuc/anomalyvfm_radio` |
| Asset revision | `17654e763c8fae5ae1c44e2ec421a427783d6196` |
| Asset file | `model.safetensors`, 1,421,491,228 bytes |
| Asset SHA-256 | `50a219ba436ed656ad3c0405f9e81df8ad00b2e715c98be66d7e2edb62a83a37` |

| Device | Prepared size | Cached construction | Inference | Peak RSS | MPS driver |
|---|---:|---:|---:|---:|---:|
| MPS | 256 × 256 | 0.94s | 88.4 ms | 3.18 GiB | 2.08 GiB |
| CPU | 256 × 256 | 0.59s | 163.9 ms | 3.18 GiB | — |
| MPS | 512 × 512 | 0.80s | 260.5 ms | 3.18 GiB | 2.07 GiB |
| CPU | 512 × 512 | 0.58s | 448.3 ms | 3.18 GiB | — |
| MPS | 768 × 768 | 0.80s | 591.3 ms | 3.18 GiB | 2.07 GiB |
| CPU | 768 × 768 | 0.59s | 1,112.8 ms | 3.18 GiB | — |

MPS is 1.9 times faster than CPU at the published 768 px shape. Memory is dominated by
the 0.4-billion-parameter model rather than spatial activations over this range. This is a
large reference, but it is Mac-credible when loaded only on demand through the existing
single-resident-worker boundary. The quality gate should retain 768 px because the project
prefers reference quality over latency and the measured device has ample headroom.

### Integration invariants found by the gate

- Anomalib pins the correct Hugging Face revision but calls `hf_hub_download` with
  `local_files_only=False` inside its model constructor. The application wrapper must first
  resolve the named asset under the app-owned cache with the experiment's download policy,
  validate byte count and SHA-256, then construct under offline mode. A cached inference
  must never require the network.
- The 1.421 GB adapted checkpoint is the model, not a disposable backbone. Every experiment
  records its repository, revision, SHA-256 and dependency versions. A reload with a changed
  asset is refused rather than silently changing the detector.
- The model is zero-shot at application time. `fit` should validate and persist the external
  identity without iterating normal training images; `predict` uses the same shared prepared
  pixels as every other method.
- The spatial encoder has a 16 px patch size. The plugin validates prepared dimensions as
  positive multiples of 16 before importing torch.
- Anomalib explicitly reports export as unsupported. The workbench's portable-export contract
  must therefore report AnomalyVFM as unavailable until a standalone graph passes ONNX Runtime
  parity; it must not relay anomalib's no-op export method as success.

**Decision:** proceed to a lazy zero-shot plugin at a documented 768 × 768 reference size,
with MPS preferred and CPU fallback. Asset resolution/offline replay and a paired public
quality gate are mandatory before promotion.
