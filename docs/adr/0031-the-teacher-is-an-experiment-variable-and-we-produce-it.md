# ADR-0031 — The teacher is an experiment variable, and we produce it

**Status:** accepted · 2026-08-08 · extends ADR-0007, ADR-0008, ADR-0029

## Context

EfficientAD cannot train from a dataset alone. It distils a student against a **pretrained
teacher**: a 2.7M-parameter PDN that has itself been taught to reproduce the local features of
a WideResNet-101 on ImageNet. Every implementation ships one, and this workbench treated it the
way it treats the ImageNette penalty set — a fixed public asset, fetched once, not a subject of
study. `efficientad_assets.py` said so in its docstring: *"these are the assets the paper's
authors released and the reference uses, so fetching different bytes would make every
comparison partly a measurement of a different teacher."*

The premise was false, and the measurement that found it was looking for something else.

Two implementations of EfficientAD in this repository agreed at 0.73 sample ROC-AUC on VisA
`candle`, where the paper reports 0.975 on VisA overall. Two independent codebases agreeing is
strong evidence that the *code* is not the cause, which correctly pointed at the data and the
protocol — and quietly skipped the one other thing both implementations load from the same
place. anomalib's `pretrained_teacher_small.pth` and the teacher bundled with
nelson1425/EfficientAD have identical architecture and identical tensor shapes and **differ
element by element by up to 1.4 in absolute value**. They are two distillation runs, not one
file under two names.

Measured on `candle` at a fixed 4000-step budget, three seeds each, changing nothing else:

| Teacher | Sample ROC-AUC | AU-PRO |
|---|---|---|
| `anomalib` | 0.764 (span 0.041) | 0.560 (span 0.114) |
| `nelson1425` | **0.888 (span 0.007)** | **0.914 (span 0.002)** |

Non-overlapping ranges — the bar ADR-0029 fixed in advance, cleared for the first time. The
teacher was worth more than 26 000 additional training steps, at identical inference cost, and
the better teacher was also six times more stable.

## Decision

**The teacher is configuration of the experiment, and one this workbench can produce.**

1. **`teacher_source` is a field of `EfficientAdCustomConfig`**, defaulting to the one that
   measured better. Both published teachers cache side by side, and a run can be repeated
   against either without a refetch. The reproduction's URL is pinned to a commit and
   checksummed: an asset that changes upstream must be a named failure, not a silent change of
   teacher between two runs that read as comparable.

2. **A `distill` job produces one.** A frozen source model, patch-aggregated to the PDN's
   output width and grid, MSE into the PDN, resumable, writing weights plus the source's
   feature-normalization statistics plus the full configuration as one described artifact in
   the model cache. `teacher_source: "distilled"` names it, and the load validates the recorded
   architecture, channel count and preprocessing before using it.

3. **The source model is behind a protocol** — `FeatureSource`, four members. The WideResNet is
   one implementation; a frozen DINOv2-S is a second, and the loop, the checkpoint, the
   manifest and the student side do not move.

4. **The source model is training-only.** What ships is the same PDN at the same inference
   cost. This is what makes (2) affordable at all.

## Alternatives

- **Pin one published teacher and treat it as part of the method.** What we had. It is simpler,
  and it makes the two implementations exactly comparable by construction. It also means the
  single largest determinant of accuracy is a file nobody here can inspect, improve, or even
  compare against an alternative — and the reason we know that is that trying it was the only
  way to find out. Rejected because a workbench whose most important input is outside it is not
  measuring what it claims to.
- **Distil ad hoc, in a script outside the application.** Cheaper today. It would need its own
  progress reporting, its own cancellation and its own resume, all of which the job system
  already has — which CLAUDE.md names as the argument for reusing it. Rejected.
- **Make the teacher a property of the *method* rather than the experiment**, e.g. a second
  registry entry per teacher. Rejected: it multiplies the method picker by an axis that is not
  a method, and the comparison screen already knows how to diff configuration.

## Consequences

- **A head-to-head against `efficientad_anomalib` must pin `teacher_source="anomalib"`.**
  ADR-0029 makes the wrapper the baseline; that only means something if both sides see the same
  teacher. This is a real cost of the default change and the most likely way to misread a
  future comparison. It is stated in the plugin's own docstring as well as here.
- **Every experiment recorded before the field existed had to be backfilled** (migration 003).
  An absent field takes the *current* default, so the default change would otherwise have
  silently rewritten what those runs claim to have done — on screen, and in `fit_more`, which
  reloads the teacher from configuration.
- **A continuation cannot change teacher.** `fit_more` reloads from configuration and does not
  refit the teacher statistics, so switching would combine a student trained against one
  network, the weights of another, and the first one's normalization. Refused by name; the
  checkpoint records what it was distilled against.
- **`distill` cost a migration**, because the `job.kind` CHECK constraint is the one place the
  queue's kind-agnosticism stops at the database. ADR-0009's claim that a new kind costs one
  entry and one handler is true of the runtime and not quite true of the schema.
- **Our own teacher will probably be worse than nelson1425's for a long time**, and that is
  expected rather than a regression: Imagenette is 13 394 images against ImageNet's 1.28M, and
  a full reference-recipe distillation is measured in days on this hardware. The value is that
  the teacher becomes measurable, and that DINOv2 becomes reachable — not that the first one
  wins.
- **Two teachers is a comparison axis that did not exist**, and it interacts with every other
  hypothesis in `measurements-efficientad.md`. One result has already had to be withdrawn
  because of it: the score-aggregation sweep found no effect under the weaker teacher and a
  0.125 effect under the better one. A null result is only evidence about the configuration it
  was measured in.

## Changelog

**2026-08-08 — new runs stop using the `anomalib` teacher entirely.** The decision above made
the teacher a variable and set a better default; this narrows it further. The anomalib teacher
remains a value of `teacher_source` **only so the five experiments recorded against it stay
loadable** — an experiment's configuration is the record of what it did, and withdrawing a
value would make those rows unreadable rather than merely obsolete. It is listed last in the
field and described as what it is.

The consequence above stands but changes character: pinning the teacher for an
implementation-versus-implementation head-to-head is now a **deliberate exercise** rather than
something anyone stumbles into, because no default produces it. ADR-0029 still makes the
wrapper the baseline this method is measured against; it is a baseline this method has left
behind rather than one it is tracked against run by run.

Amended rather than superseded, per ADR-0030: the decision is unchanged, its scope is narrower.
