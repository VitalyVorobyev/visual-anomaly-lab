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
| 2026-08-08 | `efficientad_anomalib` | 4000 | 1 | 0.731 | 0.890 | 0.534 | 45.6 |
| 2026-08-08 | `efficientad_anomalib` | 4000 | 2 | 0.746 | 0.880 | 0.548 | 46.7 |
| 2026-08-08 | `efficientad_custom` | 4000 | 0 | 0.751 | 0.900 | 0.560 | 25.8 |
| 2026-08-08 | `efficientad_custom` | 4000 | 1 | 0.764 | 0.878 | 0.498 | 25.7 |
| 2026-08-08 | `efficientad_custom` | 4000 | 2 | 0.792 | 0.893 | 0.612 | 25.9 |
| 2026-08-08 | `efficientad_anomalib` | **50** | 0 | **0.227** | 0.680 | 0.086 | 45.5 |
| 2026-08-08 | `efficientad_custom` | **50** | 0 | **0.226** | 0.689 | 0.110 | 25.6 |

### What the three seeds actually support

| Metric | `efficientad_custom` median (range) | `efficientad_anomalib` median (range) | Gap | Verdict |
|---|---|---|---|---|
| Sample ROC-AUC | 0.764 (0.751–0.792) | 0.731 (0.727–0.746) | +0.033 | **not established** |
| Pixel ROC-AUC | 0.893 (0.878–0.900) | 0.880 (0.875–0.890) | +0.013 | no claim |
| AU-PRO | 0.560 (0.498–0.612) | 0.534 (0.523–0.548) | +0.026 | no claim |
| ms/image | 25.8 (25.7–25.9) | 46.7 (45.6–48.7) | **−20.9** | **evidence** |

**Only the speed result is claimed.** The two implementations are 1.8× apart on inference with
non-overlapping ranges an order of magnitude smaller than the gap, and there is a mechanism rather
than a correlation: the wrapper computes the branch maps twice per image, once inside `model(image)`
for the score and again in `get_maps` for the two per-branch diagnostics. Ours computes them once and
reduces the combined map separately, which is what `reduce_score` exists for. Training cost is
unchanged — **130 ms/step on MPS** against the 123 the wrapper recorded — so nothing here is fast
because it does less work.

**On accuracy the rule says no, and the rule wins.** The gap in sample ROC-AUC is +0.033 and the
widest seed span is 0.041, so by the rule set in ADR-0029 before any of this was run, it is not
evidence. Pixel ROC-AUC and AU-PRO are worse than that — their ranges overlap outright.

Worth stating precisely, because it is the one place the rule and the data disagree: **all three of
our runs beat all three of the wrapper's** on sample ROC-AUC (0.751 > 0.746), and a clean separation
of three against three is the smallest p-value that design can produce — one-sided p = 1/20. So the
result is more suggestive than "not evidence" makes it sound, and it is still not a claim. The
resolution is more seeds, not a softer rule; a rule that gets relaxed the first time it is
inconvenient was never a rule. It is also a fair criticism of the rule itself, recorded here rather
than quietly reinterpreted: comparing a gap of medians against a within-group *span* conflates spread
with uncertainty of the median, and a rank test would use the data better. Changing it is a new ADR,
not an edit.

### Our seed spread is twice the wrapper's, and that is probably our doing

0.041 against 0.019 on sample ROC-AUC. The likely cause is in this file's own configuration: the
quantile fit samples `quantile_pixel_budget // quantile_images` pixels per map, which at the
compared settings (2²² over 128 images) is 32 768 of each map's 65 536 — **half the pixels, drawn at
random** — where the wrapper pools whole maps and takes an exact quantile. At 128 images the full set
is 8.4M elements, comfortably under `torch.quantile`'s 2²⁴ limit, so the subsampling bought nothing
here and cost run-to-run stability.

That makes the top hypothesis a defect in a default rather than an idea: **the budget should be spent
only when the exact fit does not fit.** It is deliberately not fixed yet, because changing it would
invalidate the six runs above; it is the first thing to measure next, and it should reduce variance
rather than move the median.

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

### The step-budget curve — one trajectory, read at six points

**Why one run and not five.** Five independent runs at 4000 / 8000 / 16000 / 30000 / 50000 / 70000
cost 178 000 steps to answer a question that 70 000 answers, because every shorter run re-walks
ground the longer one already covered. Training resumes (ADR-0025), so it does not have to. The chain
forks the seed-0 checkpoint already measured at 4000 steps, so **the curve's first point is a number
that was already in this file** rather than a new one that has to be trusted.

`candle`, generated split, seed 0 — the exploratory protocol above, not ADR-0029's. One dataset and
one seed on purpose: this measures *where the metric stops moving*, and that question does not need
a second object to answer.

| Steps | Sample ROC-AUC | Pixel ROC-AUC | AU-PRO | ms/image | Wall clock |
|---|---|---|---|---|---|
| 4 000 | 0.751 | 0.900 | 0.560 | 25.8 | — (inherited) |
| 8 000 | 0.715 | 0.904 | 0.573 | 26.0 | 9.3 min |
| 16 000 | | | | | |
| 30 000 | | | | | |
| 50 000 | | | | | |
| 70 000 | | | | | |

**What the schedule does along this chain, stated rather than hidden.** `fit_more` rebuilds the
`StepLR` against the *new* total, so each leg anneals over its own last 5% and returns to base rate
at the next resume. A point on this curve is therefore "a run of length N that annealed at the end",
which is the right thing to compare — but the legs *before* it also annealed briefly, where a single
N-step run would have held 1e-4 through its first 95%. The perturbation is small and it is the price
of not paying for 178 000 steps.

**The first leg found a bug that would have made the whole curve a measurement of itself.**
`StepLR.get_lr` is multiplicative on the param group's *current* rate, and `Adam.load_state_dict`
restores the rate the previous leg ended on — which is always the decayed one. So every continuation
started a tenfold low and dropped again at its own boundary: **1e-5 instead of 1e-4 on the first
resume, and 1e-9 by the fifth.** Both EfficientAD plugins had it, with a comment in each claiming the
opposite. Caught by reading the learning-rate metric sixty seconds into leg one, fixed by computing
the rate from the schedule's closed form at the resume point, and pinned by two tests. Every row
above except the inherited 4 000 was produced after the fix; the 4 000 run was a fresh `fit` and
never touched the resume path.

Worth stating plainly: **the resume feature has been quietly wrong since it was written**, and
nothing in the application would have shown it. The loss curve continues, the step counter is
correct, the run finishes. It took a measurement that depended on the learning rate being right to
make it visible — which is the argument for this file existing.

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
| `quantile_pixel_budget` | **Spending the budget only when an exact fit does not fit reduces the seed spread**, without moving the median. | Measured above: our spread is twice the wrapper's, and the one stochastic step the wrapper does not have is this sampling. At 128 calibration images the exact fit is 8.4M elements, under `torch.quantile`'s limit, so the sampling bought nothing. **Run this first — it is a defect in a default, not an idea, and a smaller spread is what makes every other hypothesis measurable.** |
| `calibration_holdout` | Fitting the score normalization on normals the student has *not* memorized raises sample ROC-AUC. | The two branch weights come from different quantile pairs, so the fit sets their **ratio**, and the ratio reorders images. M3 measured 0.744 → 0.769 for held-out versus training-fitted quantiles — and that already paid for 90 fewer training images. VisA's official split has no `val`, so this is live on the primary protocol. |
| `score_reduction` | The mean of the hottest `score_top_k` pixels beats a single maximum. | `amax` over 65 536 pixels is one pixel's opinion, and `pixel_reference` already reduces by a percentile for that reason. Sign genuinely unknown: a *percentile* would be wrong here — the 99.5th is the top 328 pixels, an 18 × 18 region, and many VisA defects are smaller than that — which is why the alternative offered is a small top-k rather than a quantile. |
| `student_teacher_weight` | The fixed 0.5 / 0.5 branch blend is not optimal. | Report as an **oracle upper bound only**. It cannot be tuned on normals alone, and tuning it on the test set is cheating (ADR-0029 rules it out as a shipped default). Its value is as a ceiling on what a better calibration could recover. |
| `pretrained_teacher` | The published ImageNet-distilled teacher is worth its download. | The distillation mechanism works against any fixed nonlinear feature extractor — a random teacher separates a synthetic defect perfectly in the test suite. A negative result here would make the method fully offline, so it is worth asking even though the expected answer is "yes, keep it". |
| `max_steps` | More steps is the largest single move available. | 4000 against the paper's 70000, where both implementations sit *below* the numpy floor. **Being measured now** — one trajectory to 70 000, read at six points; see the step-budget curve above. Until it lands, no other hypothesis is worth running: the seed spread at 4000 (0.041) is larger than every effect being chased (0.01–0.03), so nothing measured at that budget could reach evidence. |
