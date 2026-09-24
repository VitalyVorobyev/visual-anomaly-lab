# ADR-0029: `efficientad_custom` is our implementation; anomalib is the baseline it beats or does not

**Status:** Accepted (2026-08-08)

## Context

ADR-0008 puts a wrapped library implementation of a method first and our own second, behind one
interface. Read literally, it made the wrapper the specification: the custom implementation was
correct when it reproduced the wrapper's number. Once our implementation exists and is correct, a
faithful clone produces no new information, and this is a research workbench with everything an
evidence loop needs — the plugin boundary (ADR-0007), one preprocessing bridge, one evaluation
protocol (ADR-0011), comparison across score units (ADR-0028) and public benchmarks with official
splits (ADR-0015).

Writing `efficientad_custom` also showed that the reference is a soft baseline rather than a
ceiling: implementation choices such as the score-normalization fit reorder images on a
threshold-free metric, the reference's quantile fit is itself an estimate, and it carries latent
numerical defects (a float32 variance that can go negative, an unguarded division). The figures are
in `docs/measurements.md`.

Alternatives considered: **a faithful clone** (a second copy of a known number); **forking the
library's torch module** (inherits its defects and couples stored configs to a third-party schema,
the leakage ADR-0008 warns about); **improving the wrapper in place** (the same coupling, and it
destroys the baseline being measured against).

## Decision

**An in-house implementation of a published method — `efficientad_custom`, and `dinomaly_custom` on
the same terms — is ours. The library wrapper is the baseline it is measured against, not the
specification it must match.**

- **Equivalence with the reference is of two kinds, never mixed.** *Contract* pins are exact and
  permanent: where our network must load a published pretrained asset, a mismatch loads it into the
  wrong network without raising. *Bring-up* pins establish once that the paper was read correctly
  and then serve as a regression net for the verified core.
- **A bring-up pin is retired by replacement, never by loosening.** The change that implements a
  divergence deletes the pin, tests the new behaviour and records its measured effect. Growing a
  tolerance is forbidden.
- **Every improvement is a configuration field whose default reproduces the published behaviour**,
  so each is an ablation the comparison screen can run; an untouched run is the verified core.
- **A divergence lands with a measurement or it does not land**, under a protocol predeclared in
  `docs/measurements.md`. A change smaller than the seed range is not evidence.
- **A wrapper is removed from the registry once ours reaches parity under that protocol.** Its
  numbers then stand as a recorded measurement rather than a re-runnable comparison.

**Ruled out:** cloning the reference; forking its module; changing the wrapper instead; and tuning a
discriminative parameter on the test set. Such sweeps are reported as oracle upper bounds, never
shipped as defaults.

## Consequences

- The workbench answers "is this change better?" about methods we own, on public benchmarks, and the
  reference's defects become visible rather than inherited.
- **Two implementations diverging on purpose are harder to reason about than two that agree.** A gap
  has no presumption of being a bug, so an undocumented divergence is worse than before: nothing
  notices its absence.
- **Bring-up pins disappear by design**, and only this record and the measurement log make that
  legible rather than alarming.
- **Few seeds is thin.** Small real improvements will be indistinguishable from noise and rejected;
  that is the right error, and a real limit on what the loop can find.
- **A predeclared protocol will eventually be wrong**, and changing it invalidates every row already
  recorded against it.
- **Once a wrapper is retired, the baseline is history.** Measuring anything new against it means
  re-adding the wrapper.
- **The config model grows instead of `Capabilities`.** Each option is a hypothesis wearing a form
  control, and a field that is never measured should be deleted, not kept.
