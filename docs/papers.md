# Method references and candidate map

This is the compact reading list behind the model registry and M11. A paper appearing here is not an
integration decision: it must still fit the dataset-agnostic plugin boundary, run on Apple Silicon, expose
bounded resource planning, and produce source-frame maps under the shared evaluation protocol.

## Implemented references

| Method | Principle | Primary source |
|---|---|---|
| `efficientad_*` | student–teacher distillation + autoencoder | [EfficientAD](https://arxiv.org/abs/2303.14535) |
| `patchcore_anomalib` | pretrained patch features + coreset memory bank | [PatchCore](https://arxiv.org/abs/2106.08265) |
| `pixel_reference` | robust per-pixel classical reference | internal floor, intentionally simple |

The shared wrapper baseline is [anomalib](https://github.com/open-edge-platform/anomalib), currently pinned
to 2.6.0. It is an implementation source, not the specification of our model or evaluation contracts.

## M11 candidates

| Priority | Candidate | New principle | Integration and Apple-Silicon question |
|---:|---|---|---|
| 1 | [Dinomaly](https://arxiv.org/abs/2405.14325) | DINOv2 transformer feature reconstruction | Already in anomalib 2.6. Start with the small encoder and shallow decoder; measure MPS training memory and cache every weight by exact digest. |
| 2 | [GLASS](https://arxiv.org/abs/2407.09359) | learned discriminator with image- and feature-level anomaly synthesis | Already in anomalib 2.6. Synthetic texture inputs and generated anomaly budgets must be explicit, bounded experiment inputs. |
| 3 | [AnomalyVFM](https://arxiv.org/abs/2601.20524) | zero-shot adapted vision foundation model | Already in anomalib 2.6. It provides the most distinct no-in-domain-training reference, but RADIO weights and CPU/MPS cost need an asset and smoke-test decision first. |
| 4 | [SuperADD](https://arxiv.org/abs/2605.14808) | training-free DINOv3 multi-layer memory bank under distribution shift | Newly available in anomalib 2.6 and scientifically current, but it overlaps PatchCore's memory-bank principle. The upstream default uses a huge DINOv3 backbone and hides a 100,000-vector database cap from its public constructor, so resource planning comes before integration. |

Additional useful comparisons stay below the first integration wave:

- [SuperSimpleNet](https://arxiv.org/abs/2408.03143), a compact discriminative synthetic-anomaly model
  that can use normal-only or labelled anomalies;
- [INP-Former++](https://arxiv.org/abs/2506.03660), which reconstructs intrinsic normal prototypes from
  each test image and reports strong single-, multi-, few- and zero-shot results;
- [AnomalyDINO](https://arxiv.org/abs/2405.14529), a simple DINOv2 patch-similarity few-shot method;
- [WinCLIP](https://arxiv.org/abs/2303.14814), the established CLIP zero/few-shot baseline. Its semantic
  robustness under industrial colour and illumination shifts must be measured, not assumed.

## Selection rule

M11 integrates **different useful failure modes**, not the largest leaderboard number. The first wave is
Dinomaly → GLASS → AnomalyVFM because it yields reconstruction, learned synthetic-discrimination and
zero-shot foundation-model references beside the existing student–teacher and memory-bank families.
SuperADD is evaluated in parallel as the newest distribution-shift reference, but only moves ahead if a
bounded smaller-backbone configuration retains its value on the public protocol.
