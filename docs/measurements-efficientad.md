# EfficientAD measurements

The running log behind **ADR-0029**: `efficientad_custom` is our implementation and
`efficientad_anomalib` is the baseline it beats or does not, and *which* is a question this file
answers with numbers rather than an argument anyone wins.

**Append-only.** A row is never edited once written, and never deleted because it turned out
inconvenient — a negative result is a result, and a change that did not help is worth exactly as much
here as one that did. The roadmap keeps only the headline; this is where the working goes.

## The protocol

Fixed in ADR-0029 so it cannot drift between one measurement and the next. Anything compared against
anything else in this file was run under all of it:

| | |
|---|---|
| **Dataset** | VisA, imported with the `csv_table` adapter against `split_csv/1cls.csv` |
| **Split** | that file's official one-class split, unmodified — **it has no `val` subset**, which is the published protocol and the regime the calibration hypotheses target |
| **Objects** | `pcb1` (904 train / 100 + 100 test) and `capsules` (542 / 60 + 100) |
| **Preprocessing** | 256 × 256, RGB, bilinear — the default, and EfficientAD's own |
| **Steps** | 4000 unless a row says otherwise |
| **Seeds** | 3 for anything that decides something; 1 for exploration, marked as such |
| **Reported** | median and full range. Never mean ± sd at n = 3 |
| **Floor** | `pixel_reference` on the same split, once per object |

**A change that moves sample ROC-AUC by less than the seed range is not evidence.** That rule is the
reason the seed column exists, and it is the one most likely to be quietly broken by someone in a
hurry — including a later version of me.

### Two things to state when reading these against published numbers

- **The step budget is not the paper's.** 4000 against 70000, chosen so a run finishes in about ten
  minutes on this machine. Any gap to a published number includes that, and the step-budget curve
  below is what separates it from the rest.
- **There is no centre crop.** Official MVTec results crop the centre 76.6% of each image; the
  EfficientAD paper disables that for its own comparisons on the grounds that the crop encodes
  knowledge of where the test defects are. Our preprocessing is a straight resize, so we are on the
  paper's honest protocol already.

## Why these two objects

`pcb1` is rigid, well registered and single-instance, so `pixel_reference` is a meaningful floor on
it and EfficientAD has real headroom — an improvement has to show in the 0.95+ regime, which is where
the calibration fit and the score reducer live. `capsules` is the hardest object in VisA, with many
small objects, small defects and the smallest normal test set (60), so seed noise is largest there: it
is the one that punishes a change that only helps easy data, and the one that tests whether the
variance discipline above is real.

## Runs

No rows yet — the head-to-head is the next thing to run. Columns, when they arrive:

| Date | Implementation | Object | Steps | Seed | Sample ROC-AUC | Sample AP | Image ROC-AUC | Pixel ROC-AUC | AU-PRO | ms/step | ms/image | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Hypotheses, and what would settle them

Each is a configuration field on `EfficientAdCustomConfig` whose default reproduces the published
behaviour, so measuring one is a single changed number and nothing else.

| Field | Hypothesis | Why it might be true |
|---|---|---|
| `calibration_holdout` | Fitting the score normalization on normals the student has *not* memorized raises sample ROC-AUC. | The two branch weights come from different quantile pairs, so the fit sets their **ratio**, and the ratio reorders images. M3 measured 0.744 → 0.769 for held-out versus training-fitted quantiles — and that already paid for 90 fewer training images. VisA's official split has no `val`, so this is live on the primary protocol. |
| `score_reduction` | The mean of the hottest `score_top_k` pixels beats a single maximum. | `amax` over 65 536 pixels is one pixel's opinion, and `pixel_reference` already reduces by a percentile for that reason. Sign genuinely unknown: a *percentile* would be wrong here — the 99.5th is the top 328 pixels, an 18 × 18 region, and many VisA defects are smaller than that — which is why the alternative offered is a small top-k rather than a quantile. |
| `student_teacher_weight` | The fixed 0.5 / 0.5 branch blend is not optimal. | Report as an **oracle upper bound only**. It cannot be tuned on normals alone, and tuning it on the test set is cheating (ADR-0029 rules it out as a shipped default). Its value is as a ceiling on what a better calibration could recover. |
| `pretrained_teacher` | The published ImageNet-distilled teacher is worth its download. | The distillation mechanism works against any fixed nonlinear feature extractor — a random teacher separates a synthetic defect perfectly in the test suite. A negative result here would make the method fully offline, so it is worth asking even though the expected answer is "yes, keep it". |
| `max_steps` | More steps is the largest single move available. | 4000 against the paper's 70000. Nearly free to measure because training resumes: one run gives 4000 → 8000 → 16000 by continuing it, and the curve calibrates how much every other change is worth. **Run this first.** |
