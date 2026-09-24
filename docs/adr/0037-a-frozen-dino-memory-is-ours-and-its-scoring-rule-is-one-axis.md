# ADR-0037: A frozen DINO memory is ours, and its scoring rule is one axis

**Status:** Accepted (2026-08-23)

## Context

The workbench has a frozen self-supervised encoder table (`models/dino_backbone.py`) and a reusable
coreset selection (`models/coreset.py`) that a method may hold without importing anomalib. The next
method puts them together; the question is which one, and who writes it.

The nearest literature is three methods over the same frozen patch features, differing only in what
the memory *is*: a global patch bank with nearest-neighbour scoring ([AnomalyDINO][dino]), a
per-position Gaussian ([PaDiM][padim]), and a coreset bank ([PatchCore][pc]). They fail differently
in a way inspection data cares about. A global bank is position-blind by construction — a pattern
normal somewhere is normal everywhere — so on registered captures, where a pattern in the wrong
place is a defect, it structurally cannot see a whole class of anomaly.

Three alternatives were seriously considered:

- **Wrap anomalib's SuperADD**, a training-free DINOv3 memory bank. Free to adopt and recent, but its
  default backbone is large enough that a run measures the encoder rather than the method, its bank
  cap is hidden from the constructor (the opposite of bounding a memory before it runs), it couples
  stored configs to a third-party schema (ADR-0008, ADR-0029), and it has no per-position mode.
- **Wrap the AnomalyDINO research repository.** A faithful source for the global rule only, with no
  bounded plan or cancellation, and not a dependency this workbench can carry.
- **Three separate plugins**, one per rule. Each would read simply, but the encoder path, channel
  fusion, plan, checkpoint, map post-processing and diagnostics are identical, so three plugins are
  three copies of everything but the lines that differ, and three picker entries that hide that they
  are one method.

## Decision

**`dino_memory` is our own plugin: one module, one registry entry, one frozen encoder from the
shared backbone table, and one `scoring` field — `global_knn`, `local_knn`, `local_gaussian` — that
decides what the memory is.**

- **The scoring axis is one enum, not a layout × distance product.** The product has an invalid
  cell: a global Gaussian is one distribution over every patch the encoder sees, a model of the
  dataset's marginal rather than of normality.
- **Frozen encoders come from the shared backbone table, never from a library's model**, so every
  frozen-feature method measures the same encoders.
- **Every footprint the three rules imply is resolved and printed before the encoder is built**,
  and each kernel runs on the device a standalone probe measured for it (ADR-0008).
- **Three rules under one key are safe only because nothing is compared in score units**
  (ADR-0028): the rules report different distances and meet only through threshold-free metrics.

**Ruled out:** wrapping SuperADD or AnomalyDINO; three plugins; a second scoring axis.

## Consequences

- The frozen-backbone memory family is owned the way ADR-0029 owns EfficientAD. Encoder, layers,
  bank layout, window and shrinkage are fields, so each is an ablation measured against
  `patchcore_anomalib`.
- **A fourth rule is a new enum value and a code change**, not a second control. That is the price
  of refusing the product, and it holds only while the invalid cell stays invalid.
- **One plugin with three memory layouts reads harder than three plugins with one each**: `fit`
  branches three ways and the checkpoint carries one of three payloads. It is the less bad option.
- **`local_knn` cannot be proven on public data.** VisA is unregistered, so the mode whose premise is
  that position matters looks like a weaker global bank there. A synthetic test demonstrates the
  property; real evidence needs registered captures that cannot be published.
- **`local_gaussian` is shrinkage-dominated at its defaults**, closer to a scaled Euclidean distance
  than its name suggests. The shrinkage is a diagnostic map so this is visible, not implied.
- **The config model is large**, with fields inert under two of three rules. A field nobody measures
  should be deleted.

[dino]: https://arxiv.org/abs/2405.14529
[padim]: https://arxiv.org/abs/2011.08785
[pc]: https://arxiv.org/abs/2106.08265
