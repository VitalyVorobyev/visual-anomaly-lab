# ADR-0028: Comparing runs whose scores are not in the same units

**Status:** Accepted (2026-08-08)

## Context

The comparison screen puts N runs of one split side by side under one evaluation protocol. The
evaluation layer already produces every number it needs (ADR-0011), so the obvious design fetches N
metric sets and lays them out in columns.

That works only for metrics that are unit-free. **A score has no meaning outside its own run.**
`pixel_reference` scores a pixel as a robust z against the training median; `efficientad_custom`
reports a quantile-normalized sum of two branch errors. Both say "this looks wrong" on scales that
nothing relates, because there is no shared physical quantity underneath. So a single numeric
threshold applied to N runs describes operating points nobody chose, and a single display range
applied to N maps renders the lower-scaled method as blank. ROC-AUC, average precision and AU-PRO
are functions of the ranking and compare directly.

For the threshold-dependent half, three designs were live. **Rescale every run into a common unit**
(min-max or quantile to `[0, 1]`): tidy, and a lie, since the normalization is fitted per run and
"0.5" is a different claim per method. **Compare only what is unit-free**: honest, and it discards
the question an engineer actually asks — at a usable operating point, which method raises fewer
false alarms. **One shared rule, resolved per run**: each run gets its own threshold from the same
stated rule, and the rule and the resolved values are shown.

## Decision

**Nothing is compared in score units. What is shared across runs is a rule, never a number.**

- **Threshold-independent metrics are compared directly**, read from the stored metric sets. The
  comparison recomputes nothing and re-runs no inference.
- **Threshold-dependent outputs are resolved per run by one shared operating-point rule** that is
  part of the request: `f1` (each run at its own F1-optimal threshold, the same implementation the
  single-run results screen uses) or `recall` (the highest threshold at which a run still reaches a
  target recall; a run that cannot reach it reports no operating point). Every resolved threshold
  is printed in its run's own units beside the rule that produced it.
- **Each map is drawn on its own run-wide range.** What transfers across runs is a *fraction* of
  each range, never a value; the comparison is of where methods fire, not how hot.
- **A comparison is refused across datasets or splits, and warned across preprocessing.** A
  different split is a different question; a different resize is a legitimate experiment whose AUROC
  difference partly measures the resize.
- **Where runs disagree is computed server-side**, per sample at the resolved thresholds, so the rule
  "a score at or above the threshold is a defect" exists once, in Python.
- **It is N-way at every layer.** A new method may not cost a line here.

**Ruled out:** a common score unit; one threshold slider over N runs; one colour scale over N maps;
dropping the confusion matrix; and computing anything shown here from the maps on disk.

## Consequences

- Runs of one split can be read against each other, and the screen states which parts are unit-free
  and which are one rule applied N times.
- **The confusion matrices sit at N different thresholds.** That is correct, and still not what a
  reader assumes at a glance; it depends on the rule and the resolved values staying on screen.
- **`f1` flatters every method equally**: it is chosen with the labels of the subset being reported.
  `recall` is the honest counterweight, one control away, but the default needs no parameter.
- **The F1 search is quadratic in the number of samples** and runs once per experiment. It is reused
  rather than reimplemented so it cannot drift from the results screen.
- **Nothing makes two runs' maps quantitatively comparable**, and no future screen should claim it; a
  difference image of two methods' maps would be exactly the fabricated comparability rejected here.
- **The comparison is only as current as each run's last evaluation**, and nothing detects a column
  that is stale beside a fresh one.
