# Choosing a method

Choose a small experiment set that spans different assumptions. More architectures are not automatically
more information.

## Start with the floor

`pixel_reference` is fast, transparent, and sensitive to alignment. A strong result suggests the object is
stable and anomalies are pixel-local. Its failure map is a useful diagnostic even when it loses: edges and
shadows point to registration or illumination, while diffuse texture variation points toward feature-based
methods.

## Memory-bank reference

`patchcore_anomalib` stores representative normal patch embeddings from a frozen backbone. It is attractive
when training defects are unavailable and local appearance matters. Fit is short, but candidate selection is
quadratic and inference scales with bank size. Inspect the printed plan before running; compare backbones,
layers, and coreset budget as separate experiments.

## Compact student–teacher

`efficientad_custom` combines local teacher/student disagreement with a reconstruction branch. It is much
smaller at inference than transformer references and supports exact continuation. Teacher identity and
calibration materially affect ranking. Use several seeds before claiming a small accuracy delta; the checked
evidence historically supported a speed advantage over the retired anomalib wrapper, not a general accuracy
win — see `docs/measurements.md` for the recorded comparison, which is no longer a re-runnable in-app
baseline (ADR-0029).

## Transformer reconstruction

`dinomaly_anomalib` reconstructs frozen DINOv2 features with a trainable decoder. On the checked two-class
VisA gate it achieved the strongest mean quality among the measured candidates, at roughly ten minutes of
training per class. Choose it when a high-quality semantic reference is worth the fitting and checkpoint
cost. Its verified graph embeds normalization and emits the method's smoothed top-one-percent score.

## Learned synthesis

`glass_anomalib` learns a discriminator from synthetic feature anomalies. It is experimental: the bounded
public gate passed pixel floors but missed the predeclared image ROC-AUC promotion floor and took about
twenty minutes per class. Keep it for research on synthesis-based failure modes, not as the default.
Its experimental quality status is independent of portability: the fitted projection, discriminator, map,
and method score have a verified ONNX export.

## Frozen patch memory

`dino_memory` holds a frozen DINOv2 or DINOv3 encoder's patch features as the model, and one `scoring`
field decides what that memory is: a coreset bank over every position (`global_knn`, PatchCore's rule over
transformer features), one small bank per patch position searched over a window (`local_knn`), or one
shrunk Gaussian per position scored by Mahalanobis distance (`local_gaussian`). Nothing is trained, so a
fit is a single bounded pass and the plan is printed before it starts.

Choose `local_knn` only when the capture is **registered** — pixel (i, j) meaning roughly the same thing
across the dataset. That is the one thing a global bank structurally cannot see: a pattern that is normal
in one place and misplaced in another. On unregistered data it is a worse global bank. No public quality
gate has been run against any of the three rules yet, so treat it as a research comparison rather than a
recommended default, and read its scores only against its own run (ADR-0028).

The DINOv3 backbone entries are licence-gated and need approved Hugging Face access; the default is an
ungated Apache-2.0 DINOv2 that needs no account.

## Resource-gated candidates

AnomalyVFM is measured but not integrated. Its pinned 355M-parameter adapted RADIO asset runs on the target
Mac, but at about 1.4 GB of weights and 591 ms/image at 768 px it belongs behind explicit asset and resource
planning. “Runs once” is not the same as “fits the workbench contract.”

## A practical matrix

| Need | First choice | Contrast |
|---|---|---|
| Fast data/geometry sanity check | `pixel_reference` | identity vs localized region |
| No gradient training | `patchcore_anomalib` | different backbone/layer/bank budget |
| Compact deployable deep model | `efficientad_custom` | PatchCore |
| High-quality semantic reference | `dinomaly_anomalib` | PatchCore under same pixels |
| Study learned synthetic anomalies | `glass_anomalib` | PatchCore and Dinomaly |
| Position matters on registered capture | `dino_memory` (`local_knn`) | the same fit at `global_knn` |

Keep the comparison interpretable: same split, same prepared geometry, one hypothesis changed per run, and
failure samples inspected before the next configuration sweep.
