# ADR-0029: We own the methods we keep; external libraries are baselines to measure against

**Status:** Accepted (2026-09-26)

## Context

Every published anomaly-detection method the workbench wants can be obtained two ways: wrap a
maintained library (anomalib provides most of them), or implement it from the paper in PyTorch.

A wrapper gets a working, credible method quickly and gives a number that can be trusted as a
baseline. It cannot be changed, instrumented or reasoned about, and this is a research workbench,
not only a runner: its purpose is to answer "is this change better?". A reference implementation
is also a soft baseline rather than a ceiling — implementation choices such as a score-normalization
fit reorder images on threshold-free metrics, and libraries carry latent numerical defects. Once an
implementation of our own is correct, a faithful clone of the reference produces no new
information.

Compute is a single Apple Silicon Mac. MPS is the accelerator: no CUDA, and a PyTorch backend with
operator gaps that surface only when a kernel is reached, often mid-training.

## Decision

**A method the workbench keeps is ours. An external library is a baseline it is measured against,
never the specification it must match.** Both sit behind the one plugin interface (see ADR-0007),
under separate registry keys.

- **A wrapper translates; it does not pass through.** Its configuration is our pydantic model,
  mapped onto the library's. The library's option schema never becomes our form or our stored
  experiment configuration, so a library upgrade cannot rewrite what a past experiment means.
- **A standalone probe runs before any plugin is written against a new library or the
  accelerator** (`scripts/mps-smoke-test.py` and a `scripts/<name>-smoke-test.py` per method). It
  establishes operator coverage on MPS, the memory footprint, and which device each stage belongs
  on, as a script that can be re-run rather than a discovery made halfway through an integration.
  Its findings become the plugin's defaults, device placement and caps.
- **Defaults are MPS-oriented**, with a CPU fallback when the device or an operator is unavailable,
  and the deep-learning dependencies sit behind an optional extra so that everything else runs
  without torch.
- **Equivalence with a reference is of two kinds, never mixed.** *Contract* pins are exact and
  permanent: where our network must load a published pretrained asset, a mismatch loads it into the
  wrong network without raising. *Bring-up* pins establish once that the paper was read correctly
  and then serve as a regression net for the verified core.
- **A bring-up pin is retired by replacement, never by loosening.** The change that implements a
  divergence deletes the pin and tests the new behaviour. Growing a tolerance is forbidden.
- **Every improvement is a configuration field whose default reproduces the published behaviour**,
  so each is an ablation the comparison screen can run; an untouched run is the verified core.
- **A divergence lands with a measurement or it does not land**, under a protocol predeclared in
  `docs/measurements.md`. A change smaller than the seed range is not evidence.
- **A wrapper is removed from the registry once ours reaches parity under that protocol.** Its
  numbers then stand as a recorded measurement rather than a re-runnable comparison.

## Alternatives considered

- **Wrap everything.** Fast, and the workbench never owns a number it can explain.
- **Implement everything from scratch.** Slow, and with no reference to be measured against.
- **A faithful clone of the reference.** A second copy of a known number.
- **Forking the library's module, or improving the wrapper in place.** Both inherit the reference's
  defects and couple stored configurations to a third-party schema, and the second destroys the
  baseline being measured against.
- **Tuning a discriminative parameter on the test set.** Such sweeps are reported as oracle upper
  bounds, never shipped as defaults.

## Consequences

- The workbench answers "is this change better?" about methods it owns, on public benchmarks, and a
  reference's defects become visible rather than inherited.
- **A wrapped library is a heavy, churning dependency** while it stays: anomalib pulls in PyTorch
  Lightning and a large transitive tree, has broken its public API between minor versions, and sits
  between us and the training loop. Pinning is mandatory and upgrades mean wrapper rework.
- **Two implementations diverging on purpose are harder to reason about than two that agree.** A gap
  has no presumption of being a bug, so an undocumented divergence goes unnoticed.
- **Bring-up pins disappear by design**, and only this record and the measurement log make that
  legible rather than alarming.
- **Few seeds is thin.** Small real improvements will be indistinguishable from noise and rejected;
  that is the right error, and a real limit on what the loop can find.
- **A predeclared protocol will eventually be wrong**, and changing it invalidates every row already
  recorded against it.
- **Once a wrapper is retired, the baseline is history.** Measuring anything new against it means
  re-adding the wrapper.
- **MPS is the least-tested PyTorch backend.** Silent numerical differences and unimplemented
  operators are plausible, and a probe run once goes stale as PyTorch moves.
- **The config model grows instead of `Capabilities`.** Each option is a hypothesis wearing a form
  control, and a field that is never measured should be deleted, not kept.
