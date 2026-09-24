# ADR-0031: The teacher is an experiment variable, and we produce it

**Status:** Accepted (2026-08-08)

## Context

EfficientAD cannot train from a dataset alone. It distils a student against a **pretrained
teacher**: a small PDN that was itself taught to reproduce a WideResNet-101's local features on
ImageNet. Every implementation ships one, and the natural treatment is a fixed public asset —
fetched once, part of the method, not a subject of study.

That premise is false. The teacher anomalib ships and the one released with an independent
reproduction share an architecture and tensor shapes, and differ element by element: they are two
distillation runs, not one file. Swapping one for the other, changing nothing else, moved sample
ROC-AUC and AU-PRO by more than the seed range and by more than many times the training budget
could (`docs/measurements.md`). The single largest determinant of accuracy was a file nobody here
could inspect or vary.

Three alternatives were live. **Pin one published teacher as part of the method**: simpler, and
it keeps implementations exactly comparable, but leaves the most important input outside the
workbench. **Distil ad hoc in a script**: cheaper, and it would need its own progress, cancellation
and resume, all of which the job system already has. **Make the teacher a property of the method**,
one registry entry per teacher: it multiplies the method picker by an axis that is not a method.

## Decision

**The teacher is configuration of the experiment, and the workbench can produce one.**

- **`teacher_source` is a field of `efficientad_custom`'s config**, defaulting to the teacher that
  measured better. Published teachers cache side by side; each download is pinned and checksummed,
  so an upstream change is a named failure rather than a silent change of teacher between two runs
  that read as comparable. The anomalib teacher remains a value only so the experiments recorded
  against it stay loadable; no default produces it.
- **A `distill` job produces a teacher**: a frozen source model distilled into the PDN, resumable,
  written as one described artifact with its weights, feature-normalization statistics and full
  configuration. `teacher_source: "distilled"` names it, and loading validates the recorded
  architecture and preprocessing first.
- **The source model is behind a protocol** (`FeatureSource`). The WideResNet is one
  implementation; a frozen DINOv2 would be another, without the loop, checkpoint or student side
  moving.
- **The source model is training-only.** What ships is the same PDN at the same inference cost.
- **A continuation cannot change teacher.** `fit_more` refuses, by name, a teacher different from
  the one the checkpoint records.

## Consequences

- The teacher becomes measurable, and alternative source encoders become reachable.
- **A head-to-head against another EfficientAD implementation must pin the same teacher**, or it
  measures two teachers. Since no default produces the anomalib teacher, that pin is now a
  deliberate exercise rather than an accident.
- **A default change rewrites the meaning of old rows** unless they record what they used, so
  experiments predating the field were backfilled. Any future default change owes the same.
- **Our own distilled teacher will likely trail the published one for a long time**: the full
  reference recipe is measured in days on this hardware, and Imagenette is a small fraction of
  ImageNet. The value is that the teacher is measurable, not that the first one wins.
- **The teacher interacts with every other hypothesis.** A null result under one teacher is only
  evidence about that configuration, and at least one has already had to be withdrawn for this.
