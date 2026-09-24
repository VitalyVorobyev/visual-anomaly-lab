# Method references

A compact reading list behind the method registry. A paper here is not an integration decision: it must
still fit the dataset-agnostic plugin boundary, run on Apple Silicon, expose bounded resource planning, and
produce source-frame maps under the shared evaluation protocol. Gate results are in
[measurements.md](measurements.md).

## Implemented

| Method | Principle | Primary source |
|---|---|---|
| `efficientad_custom` | student–teacher distillation + autoencoder | [EfficientAD](https://arxiv.org/abs/2303.14535) |
| `patchcore_anomalib` | pretrained patch features + coreset memory bank | [PatchCore](https://arxiv.org/abs/2106.08265) |
| `dinomaly_custom` | DINOv2/DINOv3 transformer feature reconstruction | [Dinomaly](https://arxiv.org/abs/2405.14325) |
| `glass_anomalib` | learned discriminator with image- and feature-level anomaly synthesis | [GLASS](https://arxiv.org/abs/2407.09359) |
| `pixel_reference` | robust per-pixel classical reference | internal floor, intentionally simple |
| `dino_memory` (`global_knn`) | frozen self-supervised patch features matched by nearest neighbour | [AnomalyDINO](https://arxiv.org/abs/2405.14529) (Damm et al., WACV 2025) |
| `dino_memory` (`local_gaussian`) | one shrunk Gaussian per patch position, Mahalanobis distance | [PaDiM](https://arxiv.org/abs/2011.08785) (Defard et al., 2020); encoder and dimension reduction differ |
| `dino_memory` (`local_knn`) | one bank per patch position, searched over a window | ours: the registration-aware middle between the two rows above |
| `subspace_ad` | PCA of frozen patch features; a patch scores its residual against the normal subspace | SubspaceAD (Lendering et al., CVPR 2026). Its layer window, "layers 22-28 of 40", means two different things at any other depth (ADR-0038) |

The wrapper baseline is [anomalib](https://github.com/open-edge-platform/anomalib), pinned to 2.6.0 — an
implementation source, not the specification of our method or evaluation contracts (ADR-0029).

## Anomaly candidates

- [AnomalyVFM](https://arxiv.org/abs/2601.20524) — zero-shot adapted vision foundation model. The pinned
  RADIO asset passed the resource gate; offline app-managed loading and a public quality gate remain.
- [SuperSimpleNet](https://arxiv.org/abs/2408.03143) — compact discriminative synthetic-anomaly model, usable
  with normal-only or labelled anomalies.
- [INP-Former++](https://arxiv.org/abs/2506.03660) — reconstructs intrinsic normal prototypes from each test
  image; strong single-, multi-, few- and zero-shot results.
- [WinCLIP](https://arxiv.org/abs/2303.14814) — the established CLIP zero/few-shot baseline; robustness to
  industrial colour and illumination shifts must be measured, not assumed.
- [SuperADD](https://arxiv.org/abs/2605.14808) — training-free DINOv3 multi-layer memory bank; its principle
  is covered in-house by `dino_memory` (ADR-0037), so it is not a candidate.

## Few-shot segmentation

- **INSID3** (CVPR 2026, [arXiv 2603.28480](https://arxiv.org/abs/2603.28480),
  [code](https://github.com/visinf/INSID3)) — training-free matching over frozen DINOv3 features with
  positional debiasing. Role: baseline.
- **RSRM** (ECCV 2026) — training-free DINOv3 segmentation with semantic re-fusion and hybrid prototypes.
  Role: reference.
- **FSSDINO** ([arXiv 2602.07550](https://arxiv.org/abs/2602.07550),
  [code](https://github.com/hussni0997/fssdino)) — class prototypes from frozen DINO features, refined with
  Gram-matrix similarity. Role: baseline.
- **FSS-SAM3** (2026) — support and query placed on one canvas and segmented by a frozen SAM3. Role: idea
  only.
- **UINO-FSS** (2025) — DINOv2 features with a distilled SAM, fused by 4D correlation. Role: reference.
- **Matcher** — DINOv2 correspondence matching turned into SAM prompts. Role: reference.
- **VRP-SAM / CAT-SAM** — visual reference prompts and conditional tuning that adapt SAM to a few examples.
  Role: idea only.

## Selection rule

Integrate **different useful failure modes**, not the largest leaderboard number. Dinomaly is the
reconstruction reference; GLASS is an experimental learned-synthesis comparison; `dino_memory` covers the
frozen-backbone memory family, including the failure mode PatchCore cannot have — a pattern normal in one
place and anomalous in another; AnomalyVFM is the zero-shot foundation-model candidate.
