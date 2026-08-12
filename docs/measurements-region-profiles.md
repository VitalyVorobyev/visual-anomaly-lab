# Region-profile measurements

The evidence log behind **ADR-0033** and M10. Localising a dominant object before anomaly detection
is an experiment variable, not an assumed improvement. This file records the paired public-data gate
that decides whether any localiser is safe enough to replace the identity profile as the default.

## Protocol and decision rule

`scripts/region-value-gate.py` creates an isolated app-data directory, adopts VisA's official one-class
split for `candle` and `pcb1`, and builds identity and localised profile revisions for every source image.
It then creates a fresh PatchCore subprocess for each leg so peak resident memory is comparable. The two
legs within a class have the same source manifest, split, prepared size, padding, model configuration and
seed; only the region extractor changes.

| Variable | Fixed value |
|---|---|
| Dataset | VisA `candle` and `pcb1`, official `1cls` split |
| Method | `patchcore_anomalib`, pretrained `wide_resnet50_2`, `layer2+layer3` |
| Memory bank | 256 images, 50,000 candidate vectors, coreset ratio 0.1 |
| Prepared input | 256 × 256 RGB, 5% region padding |
| Seed | 20260812 |
| Host | Apple Silicon, macOS 26.5.2, Python 3.12.12 |
| Runtime | anomalib 2.6.0, torch 2.13.0, torchvision 0.28.0, timm 1.0.28, numpy 2.5.1 |

The rule was fixed before reading the result: localisation needs at least **+0.01 mean pixel ROC-AUC**
and **+0.01 mean AU-PRO**, may lose no more than **0.02** on either primary metric in either class, and
must build with zero failures. Image ROC-AUC is reported but cannot overrule source-frame pixel quality:
a crop that makes image ranking easier by removing real defect pixels is not an improvement.

## 2026-08-12 — foreground-threshold baseline

The classical extractor completed all 2,204 images, but selected only 14.1% of the source frame on
`candle` and 31.4% on `pcb1`. It therefore tests an aggressive object-only hypothesis rather than a mild
background trim.

| Class | Profile | Build s | Mean crop | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | Covered pixels | Missed defect pixels |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `candle` | identity | 9.2 | 100.0% | 0.867 | 0.963 | 0.890 | 100.0% | 0 / 405,291 |
| `candle` | threshold | 102.0 | 14.1% | 0.620 | 0.511 | 0.192 | 14.1% | 347,202 / 405,291 |
| `pcb1` | identity | 9.3 | 100.0% | 0.717 | 0.982 | 0.713 | 100.0% | 0 / 1,398,914 |
| `pcb1` | threshold | 40.1 | 31.4% | **0.834** | 0.833 | 0.456 | 29.6% | 351,910 / 1,398,914 |

Mean localised-minus-identity deltas were **−0.300 pixel ROC-AUC** and **−0.478 AU-PRO**. The `pcb1`
image ROC-AUC improvement of +0.117 is the important trap: without inverse projection and uncovered-pixel
accounting, it would look like evidence for localisation even though the crop omitted 25.2% of all defect
pixels and substantially reduced both localisation metrics.

Preparation was also 4.3–11.1× slower than identity. Training and inference stayed in the same band
(12.5–15.6 s training, 8.5–11.2 s inference; 1.74–1.76 GB peak RSS), as expected: both profiles become the
same 256 × 256 model input. Source-frame float maps cost about 1.23 GB per run for these 200-image test
sets, independently of crop size; compact source-map persistence is a storage follow-up, not a reason to
change the value verdict.

**Decision:** identity remains the default. `foreground_threshold` remains available as an explicit,
previewable experiment variable.

## 2026-08-12 — MobileSAM automatic grid

The paired MobileSAM run follows the same protocol. Its checkpoint is accepted only through the fixed asset
catalogue (40,728,226 bytes, SHA-256
`6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`), and its automatic grid is bounded
to 4 × 4 prompts with batches of 16.

The real MPS smoke—not merely model construction—hit an upstream float64 prompt conversion that MPS cannot
execute. The extractor now recognises that specific `TypeError`, rebuilds once on CPU, and does not swallow
unrelated type failures. On five public `candle` images, the first CPU-fallback result took 1.91 s and the
steady state took 0.41–0.44 s/image.

On `candle`, MobileSAM's selected binary mask covered a mean 63.2% of pixels; on `pcb1`, 71.4%. In both
cases its bounding box covered 100%. The spatial profile operates on an invertible rectangular crop, so both
produced byte-equivalent prepared inputs and exactly identical identity metrics.

| Class | Profile | Build s | Mean mask | Mean crop | Image ROC-AUC | Pixel ROC-AUC | AU-PRO |
|---|---|---:|---:|---:|---:|---:|---:|
| `candle` | identity | 9.5 | 100.0% | 100.0% | 0.867 | 0.963 | 0.890 |
| `candle` | MobileSAM | 510.3 | 63.2% | 100.0% | 0.867 | 0.963 | 0.890 |
| `pcb1` | identity | 9.5 | 100.0% | 100.0% | 0.717 | 0.982 | 0.713 |
| `pcb1` | MobileSAM | 505.8 | 71.4% | 100.0% | 0.717 | 0.982 | 0.713 |

All 2,204 profile entries succeeded. Training, inference and peak RSS stayed in the identity band because
the model inputs were identical; the localizer added **16.9 minutes** of CPU preparation and zero metric
gain. This is a useful negative result about the current selection rule: “largest credible mask” often
selects a background region, and mask coverage alone does not imply a useful object bounding box. The rule
must not be tuned against the test metrics after seeing this; the follow-up in the backlog calibrates a
boundary/objectness rule on training normals and validates it on different classes.

**Final M10 decision:** identity remains the default spatial input. Both `foreground_threshold` and
`mobile_sam` remain explicit, immutable, previewable options; neither silently changes existing experiments.
