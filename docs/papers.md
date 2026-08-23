# Method references and candidate map

This is the compact reading list behind the model registry and M11. A paper appearing here is not an
integration decision: it must still fit the dataset-agnostic plugin boundary, run on Apple Silicon, expose
bounded resource planning, and produce source-frame maps under the shared evaluation protocol.

## Implemented references

| Method | Principle | Primary source |
|---|---|---|
| `efficientad_*` | student–teacher distillation + autoencoder | [EfficientAD](https://arxiv.org/abs/2303.14535) |
| `patchcore_anomalib` | pretrained patch features + coreset memory bank | [PatchCore](https://arxiv.org/abs/2106.08265) |
| `dinomaly_anomalib` | DINOv2 transformer feature reconstruction | [Dinomaly](https://arxiv.org/abs/2405.14325) |
| `pixel_reference` | robust per-pixel classical reference | internal floor, intentionally simple |
| `dino_memory` (`global_knn`) | frozen self-supervised patch features matched by nearest neighbour | [AnomalyDINO](https://arxiv.org/abs/2405.14529) (Damm et al., WACV 2025) — that a frozen DINOv2 patch bank plus a nearest-neighbour rule is a complete method, with no training and no adapter |
| `dino_memory` (`local_gaussian`) | one shrunk Gaussian per patch position, scored by Mahalanobis distance | [PaDiM](https://arxiv.org/abs/2011.08785) (Defard et al., 2020) — the per-position distribution and the shrunk covariance; the encoder and the dimension reduction differ |
| `dino_memory` (`local_knn`) | one bank per patch position, searched over a window | ours: the registration-aware middle between the two rows above |

The shared wrapper baseline is [anomalib](https://github.com/open-edge-platform/anomalib), currently pinned
to 2.6.0. It is an implementation source, not the specification of our model or evaluation contracts.

## M11 candidates

| Priority | Candidate | New principle | Integration and Apple-Silicon question |
|---:|---|---|---|
| 1 | [GLASS](https://arxiv.org/abs/2407.09359) | learned discriminator with image- and feature-level anomaly synthesis | Evaluated and integrated as experimental: the bounded public gate missed the image ROC-AUC floor and trailed PatchCore. |
| 2 | [AnomalyVFM](https://arxiv.org/abs/2601.20524) | zero-shot adapted vision foundation model | The pinned 1.421 GB RADIO asset and 768 px graph passed the CPU/MPS resource gate; app-managed offline loading and public quality remain. |
| — | [SuperADD](https://arxiv.org/abs/2605.14808) | training-free DINOv3 multi-layer memory bank under distribution shift | **Superseded by `dino_memory`** (ADR-0037). Its three stated preconditions — expose the hidden 100,000-vector database bound, test a smaller DINOv3 backbone, measure CPU/MPS placement — are what `plan_memory`, the `DinoBackbone` table and `scripts/dino-memory-smoke-test.py` now deliver in-house, without third-party config coupling. |

Additional useful comparisons stay below the first integration wave:

- [SuperSimpleNet](https://arxiv.org/abs/2408.03143), a compact discriminative synthetic-anomaly model
  that can use normal-only or labelled anomalies;
- [INP-Former++](https://arxiv.org/abs/2506.03660), which reconstructs intrinsic normal prototypes from
  each test image and reports strong single-, multi-, few- and zero-shot results;
- [WinCLIP](https://arxiv.org/abs/2303.14814), the established CLIP zero/few-shot baseline. Its semantic
  robustness under industrial colour and illumination shifts must be measured, not assumed.

## Selection rule

M11 integrates **different useful failure modes**, not the largest leaderboard number. Dinomaly is the
reconstruction reference; GLASS remains an experimental learned-synthesis comparison after missing its
promotion floor; `dino_memory` covers the frozen-backbone memory family in-house, with the failure mode
PatchCore structurally cannot have — a pattern that is normal in one place and an anomaly in another;
AnomalyVFM is next as the zero-shot foundation-model candidate. SuperADD is no longer a candidate: the
principle it represents is now ours, bounded and measured, rather than wrapped.
