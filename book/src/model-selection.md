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

`dinomaly_custom` reconstructs frozen DINOv2 (or DINOv3) features with a trainable decoder, with the
encoder and the decoder depth as configuration fields. On the checked two-class VisA gate it achieved the
strongest mean quality among the measured candidates, at roughly ten minutes of training per class —
matching the anomalib-wrapped `dinomaly_anomalib` it was measured against to the third decimal on every
metric, so that wrapper has since retired (`docs/measurements.md`, ADR-0029). Choose it when a
high-quality semantic reference is worth the fitting and checkpoint cost. It does not export to ONNX yet
(`docs/backlog.md`).

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
in one place and misplaced in another. On unregistered data it is a worse global bank. At `global_knn` it
cleared the paired public gate and beat PatchCore on it (`docs/measurements.md`), which is why it is the
anomaly task's recommended default and the method a new experiment starts from; `local_knn` and
`local_gaussian` have no gate of their own. Read its scores only against its own run (ADR-0028).

The DINOv3 backbone entries are licence-gated and need approved Hugging Face access; the default is an
ungated Apache-2.0 DINOv2 that needs no account.

## Frozen patch subspace

`subspace_ad` asks a different question of the same frozen encoders: instead of keeping the training
patches and measuring a distance to them, it keeps what they **span**. Patch tokens are pooled over a
band of transformer blocks, PCA is fitted to the normals, and a patch scores the part of itself the
normal subspace cannot reconstruct. Nothing is trained and nothing is stepped — a fitted model is a
mean vector and an orthonormal basis, and a fit over sixteen images finishes in a minute or two.

**Reach for it first on a new dataset.** It is the cheapest thing here that is not a baseline: no
training horizon to choose, no bank budget to plan, and it works from a handful of normal images
rather than from a full training set, because each one is augmented with thirty random rotations.

Two fields decide most of what it does. `layers` is the band of blocks, expressed as a position in
the encoder's depth rather than as a count — `upper_half`, the default, is blocks 6–12 of a ViT-S and
12–24 of a ViT-L, and it beat the other eight windows at both depths. `tail_fraction` is how much of
the map the image score averages: small finds a small defect that a whole-map mean would drown, large
is steadier on a diffuse one, and its best value moves down as the encoder gets larger.

Set `rotations` to 0 when the part's orientation carries meaning — a component with a printed label,
a keyed connector — because a rotated copy of it is not a normal example and admitting one puts a
false direction into the subspace.

The default backbone is ViT-L, and it is also the most expensive entry in the table. Its margin
depends on the data more than anything else in the sweep did: against ViT-B it wins 2.5 points of
image AUROC on VisA, decisively and in eleven categories of twelve, and it wins **nothing** on
MVTec-AD, where the two are a tie within noise. ViT-B costs about half. Start there on data that
looks more like MVTec than VisA, and start there in any case while an experiment is still taking
shape. Its
defaults are the verdict of a parameter sweep rather than a paper's suggestion (`docs/measurements.md`,
ADR-0038), but no public **promotion gate** has been run against it, which is why it is marked
experimental — read its scores only against its own run (ADR-0028).

## Zero-shot reference

`anomalyvfm_anomalib` is AnomalyVFM: a 355M-parameter RADIO encoder adapted once, by its authors, to find
anomalies anywhere, and used here exactly as published. It reads no normal images — an experiment scores
without a train job, and every dataset gets the same weights — so it answers a question no other method
here can: **how much do my normal images buy?** A fitted method that does not beat it on the same pixels
has learned little from them.

It is the heaviest entry in the table: a 1.42 GB checkpoint, fetched once into the app cache and checked by
size and SHA-256 on every run, and about 0.6 s per image at 768 × 768 on the target Mac, the frame it was
measured at. That is its native size, so a run that names none reads it; the patch size is 16, so any
other frame must be a multiple of 16.
It cleared the public floors on VisA without seeing a normal image, so it is supported as the zero-shot
reference (`docs/measurements.md`), and like every other method its scores are read only against its own
run (ADR-0028).

## A practical matrix

| Need | First choice | Contrast |
|---|---|---|
| Fast data/geometry sanity check | `pixel_reference` | identity vs localized region |
| No gradient training | `patchcore_anomalib` | different backbone/layer/bank budget |
| Compact deployable deep model | `efficientad_custom` | PatchCore |
| High-quality semantic reference | `dinomaly_custom` | PatchCore under same pixels |
| Study learned synthetic anomalies | `glass_anomalib` | PatchCore and Dinomaly |
| Position matters on registered capture | `dino_memory` (`local_knn`) | the same fit at `global_knn` |
| Few normals, no training, answer today | `subspace_ad` | `dino_memory` (`global_knn`) on the same pixels |
| No normals at all, or what normals buy | `anomalyvfm_anomalib` | any fitted method on the same pixels |

Keep the comparison interpretable: same split, same prepared geometry, one hypothesis changed per run, and
failure samples inspected before the next configuration sweep.
