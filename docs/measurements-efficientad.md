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

### Exploratory — `candle`, generated split, n=1. **Not the protocol above.**

Run before the protocol sweep, on the `candle` dataset and `normal_only_train` split that were
already in the workbench, because an `efficientad_anomalib` baseline was already trained on them:
the comparison cost one training run instead of two. Single seed, and the split is a generated
60/20/20 rather than VisA's official one-class file — so these rows are **evidence about the
implementation, not about the method's accuracy**, and nothing here may be compared against a
published number or against a protocol row below.

| Date | Implementation | Steps | Seed | Test sample ROC-AUC | Test pixel ROC-AUC | AU-PRO | ms/image |
|---|---|---|---|---|---|---|---|
| 2026-08-08 | `pixel_reference` (floor) | — | 0 | 0.790 | 0.891 | 0.819 | 7.8 |
| 2026-08-08 | `efficientad_anomalib` | 4000 | 0 | 0.727 | 0.875 | 0.523 | 48.7 |
| 2026-08-08 | `efficientad_custom` | 4000 | 0 | **0.751** | **0.900** | **0.560** | **25.8** |
| 2026-08-08 | `efficientad_anomalib` | **50** | 0 | **0.227** | 0.680 | 0.086 | 45.5 |
| 2026-08-08 | `efficientad_custom` | **50** | 0 | **0.226** | 0.689 | 0.110 | 25.6 |

At 4000 steps ours is ahead on every metric — sample ROC-AUC +0.024, pixel +0.025, AU-PRO +0.037 —
and takes **1.9× less time per image**. Seeds 1 and 2 are needed before any of that counts as
evidence under the rule above, and are in flight; the accuracy gaps are small enough that the seed
range may well swallow them.

**The speed difference is not noise and does not need seeds.** The wrapper computes the branch maps
twice per image: once inside `model(image)` for the score and again in `get_maps` for the two
per-branch diagnostics. Ours computes them once and reduces the combined map separately, which is
what `reduce_score` exists for. 48.7 → 25.8 ms is that second forward pass through teacher, student
and autoencoder, and nothing else.

Training cost is the same to within measurement noise: **130 ms/step on MPS** against the 123 the
wrapper recorded. Nothing here is faster because it does less work per step.

**The two 50-step rows are the most useful thing measured so far.** Both implementations land at
0.226 / 0.227 — a difference of 0.002 — on the same split, the same preprocessing and the same
budget, having been written independently. Two things follow.

First, the implementation is right where it can be checked against something: the equivalence tests
pin the arithmetic given identical weights, and this pins the whole path — statistics fitting,
training loop, calibration, scoring, evaluation — end to end on real data, which no unit test
reaches.

Second, **both are inverted at 50 steps**, scoring defects *below* normals, and that is the method
rather than either implementation. Checked further on our side: each branch inverts on its own
(`map_st` 0.262, `map_stae` 0.296), and the unnormalized maps invert identically to the normalized
ones — so the calibration is monotone as designed and is not the cause. The reading that fits is
that an untrained student's error tracks the teacher's activation magnitude, which tracks local
texture; candle's defects are locally *less* textured than its normals, so an undertrained model
measures "how much texture is here" and gets the sign backwards. It is worth knowing that this
method passes through an actively wrong regime on the way up rather than merely a weak one — a run
stopped early is not a worse detector, it is an inverted one.

Also note the baseline at 4000 steps sits **below the numpy floor** on this split (0.727 against
0.790). That is not a claim about EfficientAD, which reports 97.5 on VisA at 70000 steps; it is a
claim about 4000 steps on a generated split, and it is the clearest possible argument for running
the step-budget curve before anything else.

### The diagnostics contract, checked on these runs rather than on a fixture

M6's third exit criterion is that every M4 view works on the new method with no new code. The
automated half is a unit test; this is the same question asked of two real runs on real data, by
reading their diagnostics indexes:

| | `efficientad_anomalib` (exp 2) | `efficientad_custom` (exp 5) |
|---|---|---|
| Keys, kinds and scopes | `architecture` graph/model, `score_normalization` table/model, `teacher_features_pca` image/model, `teacher_features_grid` grid/model, `teacher_magnitude` map/model, `map_student_teacher` map/image, `map_autoencoder` map/image | **identical** |
| Architecture nodes | 37, all carrying real shapes | 37, all carrying real shapes |
| Architecture edges | distillation, reconstruction | identical |
| `total_parameters` | 8 058 628 | 8 057 856 |

The parameter counts differ by exactly **772**, and the difference is a finding rather than a
discrepancy: 384 + 384 + 4 is the size of anomalib's `mean_std` and `quantiles` `ParameterDict`s.
Those are fitted *statistics*, not learnable parameters, and counting them inflates the number the
Architecture tab prints. Ours holds them as buffers, so the count is what the optimizer actually
touches. A twelve-line decision made for correctness turns out to be visible on screen.

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
