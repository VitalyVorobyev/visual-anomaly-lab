# ADR-0008: Hybrid deep-learning strategy — anomalib now, custom EfficientAD later

**Status:** Accepted (2026-08-06)

## Context

The brief asks for EfficientAD and PatchCore. Both can be obtained two ways: wrap a maintained
library (anomalib provides both), or implement them from scratch in PyTorch.

The two options serve different goals. Wrapping gets a working, credible workbench quickly and
provides a reference implementation whose numbers can be trusted as a baseline. Implementing from
scratch is where the actual learning about the method lives — and this is a research workbench whose
purpose includes understanding these methods, not only running them.

Compute is a single Apple Silicon Mac. MPS is the available accelerator: no CUDA, no multi-GPU, and
a PyTorch backend with known operator coverage gaps.

## Decision

**Both, in sequence, behind the same interface. Ship the wrappers first.** (User decision.)

- **Now:** `efficientad_anomalib` and `patchcore_anomalib` are thin wrappers around anomalib,
  implementing the `AnomalyModel` ABC of ADR-0007. They translate our config models into anomalib's,
  run its training/inference, and emit per-image scores and float32 anomaly maps in our format.
- **Later:** `efficientad_custom` — a from-scratch PyTorch EfficientAD — is a roadmap item, added as
  a **second implementation behind the same interface**, not a replacement. Because both register
  under distinct keys and produce identical result shapes, the app can run them on the same split
  and compare them directly. The custom implementation's correctness is then measurable against the
  battle-tested one, in-app, on our own data.
- **MPS-oriented defaults:** device selection prefers MPS with an automatic CPU fallback for
  unsupported operations; default image sizes are moderate rather than maximal, chosen so a training
  run fits comfortably in unified memory and completes in a tolerable time on one machine.

## Consequences

A working end-to-end deep-learning path exists early, so the rest of the workbench (jobs, results,
evaluation, comparison UI) can be built and validated against real model behaviour rather than
stubs. When the custom implementation arrives, it inherits a fully built evaluation and comparison
apparatus, and "is my implementation right?" becomes a question the tool itself answers. If the
custom version never gets written, the product is still complete against the brief.

Negative consequences, accepted honestly:

- **A heavy dependency.** anomalib pulls PyTorch Lightning and a large transitive tree into the
  backend. Install size and cold-start import time grow substantially, and Lightning's abstractions
  sit between us and the training loop, making failures harder to diagnose and progress reporting
  awkward to route into our `TrainContext` callbacks (ADR-0007).
- **API churn.** anomalib has broken its public API between minor versions before. Version pinning
  is mandatory, upgrades will require wrapper rework, and our wrappers will accumulate
  version-conditional code.
- **EfficientAD needs an ImageNet subset.** Training requires downloading an external dataset — a
  first-run network dependency in a tool that is otherwise strictly local (ADR-0003), and a manual
  setup step that will confuse anyone who expects offline operation.
- **MPS is the least-tested PyTorch backend.** Silent numerical differences, unimplemented
  operators, and memory behaviour unlike CUDA are all plausible; some anomalib configurations may
  simply not run, or may fall back to CPU and be slow enough to be impractical.
- **Two implementations of one algorithm is ongoing maintenance.** Once `efficientad_custom` exists,
  divergence between the two is a permanent source of "which one is wrong?" investigations, and the
  interface must keep accommodating both.
- **Wrapper leakage.** Config models risk becoming pass-throughs of anomalib's own options, which
  would couple our UI and stored experiment configs to a third-party schema.

## Changelog

**2026-08-23 — the EfficientAD leg of "wrappers first, custom later" is complete, and the
wrapper is retired.** `efficientad_anomalib` shipped first, as this record set out to do;
`efficientad_custom` then shipped behind the same interface and, per ADR-0029, became the
implementation this workbench measures and improves rather than a second copy of a known
number. With the in-house implementation established and preferred, the wrapper no longer earns
its maintenance cost (API churn, Lightning coupling, a heavy dependency) and has been removed
from the registry; its recorded numbers remain in `docs/measurements.md` as a historical
baseline. `patchcore_anomalib`, `dinomaly_anomalib` and `glass_anomalib` are unaffected — the
wrappers-first strategy stands for them.

**2026-08-24 — the Dinomaly leg follows the same arc, and the second of the two anomalib
wrappers this record shipped is now retired.** `dinomaly_anomalib` shipped first, as this
record set out to do; `dinomaly_custom` then shipped behind the same interface and, per
ADR-0029's pattern extended a second time, became the implementation this workbench measures
and improves. It reached VisA parity with the wrapper's recorded run (means matching to the
third decimal on all three metrics), so under the predeclared rule the wrapper has been
removed from the registry; its recorded numbers remain in `docs/measurements.md` as a
historical baseline. `patchcore_anomalib` and `glass_anomalib` are the two methods still wrapped
through anomalib — the wrappers-first strategy stands for them.
