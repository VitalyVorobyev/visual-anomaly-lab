# ADR-0028: Comparing runs whose scores are not in the same units

**Status:** Accepted (2026-08-08)

Extends **ADR-0011** (metrics come from a job; the evaluation layer is model-independent) and
the run-wide display range that makes two images of one run comparable by eye (handbook
diagnostics.md). Neither says
what happens when the two things being compared are two *runs*, and that turns out to be a different
question.

## Context

The comparison screen is the milestone that makes this workbench worth having: several methods, one
split, one evaluation protocol, read side by side. The evaluation layer already produces every number
it needs, computed by one implementation that never imports a model (ADR-0011). So the obvious design
is a screen that fetches N experiments and lays their metrics out in columns.

That works exactly as far as the metrics that are unit-free, and no further.

**A score has no meaning outside its own run.** `pixel_reference` scores a pixel as a robust z — how
many MADs it sits from the training median — and a typical map peaks around 8. `efficientad_anomalib`
scores the sum of a distilled student's error and an autoencoder's, quantile-normalized against a
reference set, and peaks around 1.3. The two numbers describe the same claim ("this looks wrong") on
scales that have nothing to do with each other, and nothing in the system relates them, because
nothing can: there is no shared physical quantity underneath. A third method could peak at 4000.

Two consequences, both of which reach the screen:

- **A single numeric threshold applied to N runs is meaningless.** At 1.3, EfficientAD is at its
  operating point and `pixel_reference` calls everything normal. A comparison table with one slider
  above it would print a confusion matrix per method, all of them true, describing operating points
  nobody chose — and it would look exactly like a fair comparison.
- **A single display range applied to N maps is meaningless in the same way.** The run-wide range
  (handbook diagnostics.md) exists so two *images* are comparable; stretching two *runs* over one range renders the
  lower-scaled method's map as flat background, which reads as "this method found nothing".

The threshold-independent metrics have no such problem. ROC-AUC, average precision and AU-PRO are
functions of the *ranking*, not of the values, so they are directly comparable across methods and
across datasets — which is exactly why the literature reports them.

Three designs were considered for the threshold-dependent half:

**Rescale every run into a common unit.** Min-max or quantile-normalize each run's scores to `[0, 1]`
and threshold that. It makes the table look tidy, and it is a lie about units: the normalization is
fitted per run, so "0.5" is a different claim per method, and the number on screen now implies a
comparability the underlying quantity does not have. It also breaks the moment a run's score
distribution is skewed, which one-class methods' distributions generally are.

**Compare only what is unit-free.** Drop the confusion matrix from the comparison entirely. Honest,
and it discards the thing an engineer actually asks — *at a usable operating point, which method
raises fewer false alarms* — which is not answerable from an AUROC.

**One shared rule, resolved per run.** Each run gets its own threshold, derived by the same stated
rule from its own score distribution; the rule is named on screen and each run's resolved value is
printed beside its confusion matrix. This is what the results screen already does for a single run —
`suggest_threshold` returns a number *and* the sentence explaining how it was chosen, precisely
because a silently-opened slider invites being read as a recommendation.

## Decision

**Nothing is compared in score units. What is shared across runs is a rule, never a number.**

- **Threshold-independent metrics are compared directly**, unchanged and unnormalized: sample and
  image ROC-AUC, average precision, pixel ROC-AUC, AU-PRO, and the timing summary. They are read from
  the stored metric sets a job wrote (ADR-0011) — the comparison **recomputes nothing** and re-runs
  no inference.
- **Threshold-dependent outputs are resolved per run by one shared operating-point rule**, and the
  rule is part of the request. Two rules ship:
  - **`f1`** — each run at its own F1-optimal threshold, which is `suggest_threshold`'s first case
    and therefore the same rule, the same implementation and the same rationale sentence the results
    screen shows. Every method at its own best, which is the fairest reading when the question is
    "which method".
  - **`recall`** — the *highest* threshold at which a run still reaches a target recall, so every
    method is held to the same detection rate and the comparison is of what that rate *costs* in
    false alarms. Highest rather than any: recall only falls as the threshold rises, so every lower
    cut reaches the target too, and the highest one is the only choice that does not hand a method
    false alarms it did not need.
    This is the operating point an inspection engineer actually specifies, and it is the one that
    makes the confusion matrix say something the AUROC does not. A run that cannot reach the target
    at any threshold reports no operating point rather than its closest attempt.
- **Every resolved threshold is printed in its run's own units, next to the rule that produced it.**
  A confusion matrix with no threshold beside it is a claim about an operating point the reader
  cannot name.
- **Anomaly maps are drawn each on its own run-wide range, and both ranges are printed.** What
  transfers across runs is the *fraction* of the range, not the value: one cut slider drives N
  segmentations at the same relative operating point, which is the same reasoning `ResultsState.cut`
  already encodes for a single run. The comparison this supports is of **where** the methods fire and
  where they disagree — spatial structure — never of how hot one map is against another.
- **A comparison is refused across datasets or across splits, and warned across preprocessing.** A
  different split is a different question and the numbers are not commensurable, so it is a 422 and
  not a caveat. Preprocessing is a warning because two runs at different resolutions are a legitimate
  experiment — but the AUROC difference then partly measures the resize, which is the exact failure
  the shared preprocessing bridge exists to prevent, so it is stated loudly on the screen rather than
  in a tooltip.
- **Where the methods disagree is a first-class output, not a client-side derivation.** The per-sample
  agreement table is computed server-side at the resolved thresholds, one row per sample with one
  outcome per run. The rule "a score at or above the threshold is a defect" already exists once, in
  Python; N runs' worth of it re-derived in TypeScript is the same class of bug `ThresholdReport`
  was shaped to avoid, multiplied by N.
- **It is N-way, not two-way, at every layer** — the request takes a list, the response is
  index-aligned with it, and the screen lays out columns. "A/B" is the N = 2 case. M6 and M7 each add
  a method, and neither may cost a line here.
- **The agreement table is not capped.** The standing rule is to bound anything whose cost is linear
  in the dataset *and whose value is not* — here the value is linear too, because every disagreeing
  sample is one the reader wants. What *is* capped is the number of runs, at six, because a table
  wider than that is not read.

**Ruled out:** normalizing scores into a common unit (a fabricated comparability, and the tidiest of
the three designs is the one that misleads); a single threshold slider over the comparison (true
numbers, unchosen operating points, and it looks fair); a shared colour scale across maps (renders
the lower-scaled method blank); dropping the confusion matrix (honest and useless); and computing
anything the comparison shows from the anomaly maps on disk — pixel-level *curves* stay absent here
for the same reason they are absent from the benchmark tab, because the accumulator streams its
histograms and discards them (handbook evaluation.md), and reconstructing one would mean re-reading every map.

## Consequences

Two methods can be read against each other on one split under one protocol, and the screen states
which parts of that reading are unit-free and which are a rule applied twice.

Negative consequences, accepted honestly:

- **The confusion matrices in a comparison are at N different thresholds.** That is the correct
  answer to an incommensurable-units problem, and it is still a table whose columns are not what a
  reader assumes at a glance. It rests on the rule being named on screen and the resolved value being
  printed — remove either and the table becomes the misleading one this record rejected.
- **`f1` flatters every method equally.** Each run at its own optimum is the best case, chosen with
  the labels of the subset being reported, which is an optimistic operating point in exactly the way
  fitting anything on the test set is. `recall` is the honest counterweight and is one control away,
  but the default is the flattering one because it is the one that needs no parameter.
- **The F1 rule is quadratic in the number of samples.** `suggest_threshold` evaluates a full
  confusion matrix per candidate score, and the comparison runs it once per experiment. It is
  reused rather than reimplemented, because a second F1-optimal search would be free to drift from
  the one the results screen shows — the cost is accepted to keep one implementation.
- **Nothing here makes two runs' *maps* quantitatively comparable, and no future screen should
  claim otherwise.** The fraction-of-range cut is a shared *relative* operating point; it is not a
  shared unit, and stacking two methods' maps into a difference image would be exactly the
  fabricated comparability this record rejects.
- **The comparison reads the stored metric sets, so it is only as current as the last evaluation.**
  A run whose scores were re-evaluated under a different aggregation shows the newer numbers; one
  that was retrained without being re-scored shows the older ones. `POST /reevaluate` is the fix and
  the screen names it, but nothing detects the staleness — the same gap ADR-0011 accepts for the
  single-run results screen, now visible in a table where one column can be stale beside a fresh one.

## Changelog

**2026-08-23 — the illustration named a retired method.** `efficientad_anomalib` is no longer
registered (ADR-0029 changelog); the two-methods-two-scales point in Context above still holds
verbatim for a surviving pair — `pixel_reference` around 8, and `efficientad_custom`'s own
quantile-normalized sum on a scale unrelated to it. Nothing about the decision changes: a score
still has no meaning outside its own run.
