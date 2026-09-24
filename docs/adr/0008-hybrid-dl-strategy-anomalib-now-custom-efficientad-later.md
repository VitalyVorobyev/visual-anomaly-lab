# ADR-0008: Hybrid deep-learning strategy — wrap a maintained library first, own a method later

**Status:** Accepted (2026-08-06)

## Context

Every published anomaly-detection method this workbench wants can be obtained two ways: wrap a
maintained library (anomalib provides most of them), or implement it from the paper in PyTorch.

The two serve different goals. A wrapper gets a working, credible method quickly and gives a
reference whose number can be trusted as a baseline. An implementation of our own is where the
understanding of the method lives, and it is the only kind the workbench can change, instrument and
reason about — this is a research workbench, not only a runner.

Compute is a single Apple Silicon Mac. MPS is the accelerator: no CUDA, and a PyTorch backend with
operator gaps that surface only when a kernel is reached, often mid-training.

The alternatives were to wrap everything (fast, and the workbench never owns a number it can
explain) or to implement everything from scratch (slow, and with no reference to be measured
against).

## Decision

**Both, in sequence, behind the one plugin interface (see ADR-0007): a method may enter as a thin
wrapper, and one we need to reason about is then implemented as our own, under a separate key.**

- **A wrapper translates; it does not pass through.** Its configuration is our pydantic model,
  mapped onto the library's. The library's own option schema never becomes our form or our stored
  experiment configuration, so a library upgrade cannot rewrite what a past experiment means.
- **Our implementation is measured against the wrapper as a baseline, not a specification**, and a
  wrapper that our implementation matches can be retired (see ADR-0029).
- **A standalone probe runs before any plugin is written against a new library or the
  accelerator.** It establishes operator coverage on MPS, the memory footprint, and which device
  each stage belongs on, as a script that can be re-run — not as a discovery made halfway through an
  integration. Its findings become the plugin's defaults and caps.
- **Defaults are MPS-oriented.** A method prefers MPS where its tensor work benefits, falls back to
  the CPU when the device or an operator is unavailable, and uses moderate input sizes so a run
  fits unified memory in a tolerable time.
- **The deep-learning dependencies sit behind an optional extra**, so everything else — the floor
  method, evaluation, and every other test — runs without torch.

## Consequences

Real deep-learning behaviour exists before an in-house implementation does, so jobs, results and
comparison are built against it rather than stubs, and every in-house method inherits a finished
evaluation and comparison apparatus plus a baseline to beat.

- **A heavy dependency.** anomalib pulls in PyTorch Lightning and a large transitive tree. Install
  size and import time grow, and Lightning's abstractions sit between us and the training loop,
  making failures harder to diagnose and progress awkward to route into our contexts.
- **API churn.** The library has broken its public API between minor versions; pinning is
  mandatory and upgrades mean wrapper rework.
- **Two implementations of one algorithm are ongoing maintenance** until the wrapper retires, and a
  retired wrapper's baseline survives only as a recorded number that cannot be re-run.
- **MPS is the least-tested PyTorch backend.** Silent numerical differences and unimplemented
  operators are plausible; a probe run once can go stale as PyTorch moves.
- **Some methods need external assets** — pretrained weights, a penalty image set — which is a
  first-run network dependency in a tool that is otherwise local.
