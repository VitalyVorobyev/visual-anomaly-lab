# ADR-0029: `efficientad_custom` is our implementation; anomalib is the baseline it beats or does not

**Status:** Accepted (2026-08-08)

Extends **ADR-0008** (both implementations, in sequence, behind one interface) and supersedes its
framing of the custom one as a method *whose correctness is measured against* the wrapper. Nothing
else in ADR-0008 is reversed: the wrappers stay, the order stays, the MPS orientation stays.

## Context

ADR-0008 committed to two implementations of EfficientAD behind one interface, and said the custom
one's "correctness is then measurable against the battle-tested one, in-app, on our own data". Read
literally, that makes the wrapper the specification and the goal a second copy of a known number.

Building it made a different question unavoidable: once the from-scratch implementation exists and is
correct, what is it *for*? A faithful clone produces no new information. This is a research
workbench, and by M6 it has everything an evidence loop needs — the plugin boundary (**ADR-0007**),
one shared preprocessing bridge so two methods see identical pixels, one evaluation protocol
(**ADR-0011**, handbook evaluation.md), resumable training (handbook jobs.md), comparison in non-commensurable
score units (**ADR-0028**), and a public benchmark with masks and an official split (**ADR-0015**).

Four measured facts sharpen it, all found while writing the implementation:

- **The wrapper is a soft baseline, not a ceiling.** An independent reimplementation of the paper
  matches or beats its published numbers (VisA-S 97.6 against the paper's 97.5), while anomalib's
  EfficientAD carries open, unresolved performance issues.
- **Implementation choices reorder images.** M3 measured the score-normalization *fit* moving sample
  ROC-AUC by 0.025 — on a threshold-free metric — because the two branch weights come from different
  quantile pairs and the fit sets their ratio.
- **The reference's quantile fit is already an estimate.** It subsamples randomly above `torch.quantile`'s
  2²⁴-element limit, which is a hard failure on MPS. An exact fit over a full training set is not
  available to anyone.
- **The reference has latent numerical defects.** `sqrt(E[x²] − E[x]²)` in float32 can go negative
  from cancellation and yield a `NaN` deviation that silently poisons every score in a run; the map
  normalization divides by `qb − qa` unguarded.

Alternatives considered. **A faithful clone**: produces a second copy of a known number and nothing
else. **Forking anomalib's torch module**: inherits its constraints and its defects, and couples our
stored configs to a third-party schema — the leakage ADR-0008 already warns about. **Improving the
wrapper in place**: same coupling, and it destroys the baseline we would be measuring against.

## Decision

**`efficientad_custom` is our own implementation of arXiv:2303.14535. anomalib is the floor it is
measured against, not the specification it must match.**

- **Equivalence with the reference is split into two kinds, and they are never mixed.** *Contract*:
  the PDN forward, pinned at `rtol=0, atol=0`, permanently — it must load the published pretrained
  teacher, so a mismatch means loading a public asset into the wrong network, which does not raise
  and merely costs accuracy. *Bring-up*: the autoencoder, the losses and the map computation,
  establishing once that we read the paper correctly, and then held as a regression net for the
  verified core.
- **A bring-up pin is retired by replacement, never by loosening.** The only sanctioned retirement
  deletes the test in the same change that implements the divergence, tests the new behaviour, and
  records its measured effect. **Growing a tolerance is forbidden.** A test whose `atol` quietly went
  from 0 to 1e-2 is the failure mode this whole record exists to prevent.
- **Every improvement is a configuration field whose default reproduces the published behaviour.**
  Options are free here — the picker and every form are generated from the plugin's JSON Schema — so
  each improvement is an ablation the workbench itself can run and put beside its own baseline on the
  comparison screen. An untouched run *is* the verified core.
- **A divergence lands with a measurement or it does not land.** The protocol is fixed here so it
  cannot drift: VisA, the official one-class split unmodified, 256×256 RGB bilinear, a fixed step
  budget, three seeds, median and range reported, `pixel_reference` as the floor. **A change that
  moves sample ROC-AUC by less than the seed range is not evidence.**
- **Measurements accumulate in `docs/measurements.md`**, append-only. The roadmap keeps
  only the headline, because that file is for what still constrains new work.
- **The diagnostic keys are the wrapper's**, pinned by a test that reads them out of the wrapper's own
  source rather than restating them — so the wrapper is the executable specification of the
  diagnostics contract, which is the coordination cost **ADR-0018** said M6 would have to pay.

**Ruled out:** cloning the reference; forking its torch module; making these changes to the wrapper
instead; and tuning any *discriminative* parameter — the branch weighting — on the test set. Such
sweeps are reported as oracle upper bounds on what a better calibration could recover, never shipped
as defaults.

## Consequences

The workbench answers "is this change better?" about a method we own, on a public benchmark, with a
floor and a reference in the same screen. That is the research payoff ADR-0008 promised and could not
itself deliver. It also makes the reference's defects visible rather than inherited: the guards, the
float64 accumulation and the refusals in `efficientad_custom` are all things found by writing the
implementation rather than reading about them.

Negative consequences, accepted honestly:

- **Two implementations diverging on purpose are harder to reason about than two that agree.** "Which
  one is right?" becomes "which one is better, and on what?", and that question has no answer outside
  the protocol.
- **A gap now has more than one candidate explanation.** Under ADR-0008 there was a presumption of
  equality, so a difference was a bug to find. There is no longer any such presumption, which means an
  *undocumented* divergence is worse here than it was before — nothing will notice its absence.
- **The bring-up pins will rot, by design.** A reader later finds tests that were deliberately
  deleted rather than fixed. This record, and the measurement log, are the only things that make that
  legible rather than alarming.
- **Three seeds is thin.** It is what fits a spare-time compute budget at ten minutes a run, and it
  means small real improvements will be indistinguishable from noise and correctly rejected. That is
  the right error to make, and it is a real limit on what this loop can find.
- **The protocol is now a frozen thing that will eventually be wrong.** Two VisA objects are not the
  benchmark, and changing them invalidates comparability with every row already recorded.
- **`Capabilities` did not grow, but the config model did.** `calibration_holdout`, `score_reduction`,
  `score_top_k`, `pretrained_teacher` and `quantile_pixel_budget` are hypotheses wearing form
  controls. ADR-0007 predicted the capability flags would become "a growing, weakly-typed catalogue of
  exceptions"; the same pressure now applies one level down, to a method's own options, and the same
  discipline is owed — a field that never gets measured should be deleted, not kept.

## Changelog

**2026-08-23 — `efficientad_anomalib` is retired; two honest degradations follow.** The wrapper
this record measures `efficientad_custom` against is removed from the registry (the roadmap's
method count drops to five). Two things this record relied on change character rather than
disappear:

1. **The baseline leg becomes a recorded historical measurement, not a re-runnable in-app
   comparison.** The numbers in `docs/measurements.md` stand; nothing new can be measured against
   the wrapper without reinstalling code that is now deleted. A future comparison against
   anomalib's EfficientAD would mean re-adding the wrapper, not flipping a flag.
2. **The diagnostic-key contract is no longer read from the wrapper's source.** "The diagnostic
   keys are the wrapper's, pinned by a test that reads them out of the wrapper's own source"
   (Decision, above) described a live read of `efficientad_anomalib.py`; that file is gone, so the
   keys are now a frozen literal set in
   `test_dl_efficientad_custom.py::test_the_diagnostic_keys_are_the_ones_the_views_expect`. The
   keys themselves are unchanged; the executable specification moved from a source file to a
   written list.

The rest of the record is unaffected: `efficientad_custom` is still our own implementation, still
measured against a floor rather than cloning a specification, and the equivalence and divergence
discipline still applies to it alone.
