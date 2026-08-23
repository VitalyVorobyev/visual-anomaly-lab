# ADR-0037 — A frozen DINO memory is ours, and its scoring rule is one axis

**Status:** Accepted (2026-08-23)

Extends **ADR-0008** (wrap first, own what we need to reason about) and **ADR-0029** (a wrapper is a
baseline, not a specification) to the frozen-backbone memory family. Supersedes the SuperADD evaluation
item that `docs/backlog.md` and `docs/papers.md` carried.

## Context

`patchcore_anomalib` gave the workbench a memory-bank method, and `models/dino_backbone.py` and
`models/coreset.py` gave it a frozen self-supervised encoder and a reusable coreset selection that a
method may hold without importing anomalib. The obvious next step is a method that puts the two
together. The question is which one, and who writes it.

Three facts made this a real question rather than a formality.

**The nearest useful literature is three papers, not one.** [AnomalyDINO][dino] shows that a frozen
DINOv2 patch bank plus a nearest-neighbour rule is a complete method with nothing trained. [PaDiM][padim]
shows that a per-patch-position Gaussian over pretrained features is another complete method over the
same inputs. [PatchCore][pc] is the coreset bank we already have. All three read the same features and
differ only in what the memory *is* — and they fail differently, which is what
[`docs/papers.md`](../papers.md)'s selection rule actually asks for.

**They fail differently in a way our data cares about.** A global bank is position-blind by
construction: a pattern that is normal *somewhere* is normal everywhere. On registered inspection
captures — which the showcase corpus is, and which most fixtured industrial capture is — a pattern in
the wrong place is a defect, and no global bank can see it. That is not a tuning difference. It is a
class of anomaly one rule structurally cannot detect.

**anomalib 2.6 ships SuperADD**, a training-free DINOv3 multi-layer memory bank, and the backlog held
it behind three preconditions: expose and plan its hidden 100 000-vector database bound, test a smaller
DINOv3 backbone, and measure CPU/MPS behaviour. Those preconditions were written before
`dino_backbone.py` existed. They now describe work this repository does for itself.

**A standalone probe ran first, as ADR-0008 requires** (`scripts/dino-memory-smoke-test.py`). Nine
stages, both devices, no downloads. Its verdicts are facts about this machine rather than uncertainties
carried into a plugin: the encoder forward wants MPS (~2×); `topk` over a 100 000-wide row is ~7×
*slower* on MPS and breaks exact ties the other way; batched Cholesky is 16× slower on MPS where the
kernel exists at all; the distance and einsum kernels and the map post-processing run on either device.

### Alternatives seriously considered

**Wrap anomalib's SuperADD.** Zero implementation cost, and it is the newest published work in the
family. Against it: its default is a large DINOv3 backbone, so a run measures the encoder rather than
the method; its 100 000-vector database cap is hidden from the public constructor, which is precisely
the "bound it before it runs, not after" property `plan_bank` exists to provide; adopting it couples our
stored experiment configs to a third-party schema, the leakage ADR-0008 warns about and ADR-0029 refused
for EfficientAD; and it has no per-position mode at all, so the one failure class that motivated this
work would still be missing.

**Wrap the AnomalyDINO research repository.** A faithful source for the global rule, and nothing else —
a research codebase is not a dependency this workbench can carry, has no bounded-plan or cancellation
story, and still leaves the per-position rules unwritten.

**Three separate plugins** (`dino_global`, `dino_local`, `dino_gaussian`). Each module would then hold
one memory and read straightforwardly. Against it: the encoder path, the channel fusion, the plan, the
checkpoint, the map post-processing and the diagnostics are identical across all three, so three plugins
means three copies of everything that is not the twenty lines that differ — and the method picker would
offer three entries whose names do not say that they are one method under one encoder.

## Decision

**`dino_memory` is our own plugin: one module, one registry entry, one frozen encoder taken from the
shared `DinoBackbone` table, and one `scoring` field with three values — `global_knn`, `local_knn`,
`local_gaussian` — that decides what the memory is. We do not wrap SuperADD or the AnomalyDINO
repository, and we do not split the three rules into three plugins. The scoring axis is a single enum
rather than a layout × distance product, because that product has an invalid cell: a global Gaussian is
one distribution fitted to the union of every patch the encoder ever sees, which is a model of the
dataset's marginal rather than a model of normality. Every footprint the three rules imply is resolved
by a pure, torch-free `plan_memory` and printed before the first forward pass, and every kernel is
placed on the device the smoke test measured for it.**

- **Frozen encoders come from `dino_backbone.py`, never from a library's model.** Five entries, three
  ungated; the default is ungated Apache-2.0 so a fresh machine gets a result rather than a 401. Gated
  DINOv3 weights are reached through an **ambient `HF_TOKEN`** and never through a config field: a
  licence is a permission the user holds, not an input to the experiment, and a token on a form would be
  stored in the database, printed into job logs and carried into every comparison export.
- **The plan precedes the encoder, not just the pass.** The grid comes from `patch_grid`'s arithmetic
  and the width from the backbone table times the layer count, so `describe()` is logged before the
  encoder is constructed. The first real batch verifies both and refuses a disagreement by name.
- **Nothing is compared in score units** (ADR-0028), which is what makes three rules under one key safe:
  a `local_gaussian` run reports Mahalanobis distances and a `global_knn` run reports squared Euclidean
  ones, and neither is ever read against the other except through threshold-independent metrics.
- **`channel_fusion=feature_concat` makes the banking unit a sample.** Per-channel vectors are
  concatenated in `channel_order`'s stable sorted order and L2-normalized again, so the squared distance
  between fused vectors is the *average* of the per-channel distances and the score scale is independent
  of the channel count. `capabilities().channel_aware` is `True` regardless of the field, because the
  capability says the model *may* consume channel metadata.
- **`portable_formats` is empty**, not conditional. `feature_concat` has no single-input graph, and the
  export offer is made from the registry before any configuration is read.
- **`supports_resume` is `False` and `SupportsResume` is not implemented.** There are no steps to
  continue; a memory is either built or it is not.

**Ruled out:** wrapping SuperADD or AnomalyDINO; three plugins; a second axis; a `hf_token` config
field; and declaring a portable format that only one configuration of the method can honour.

## Consequences

The workbench now owns the frozen-backbone memory family the way ADR-0029 made it own EfficientAD. The
encoder, the layer depth, the bank layout, the window radius, the reduction width and the shrinkage rule
are all fields, so each is an ablation the comparison screen can already run — and each is measured
against `patchcore_anomalib`, which is the same principle through somebody else's library. SuperADD's
three preconditions are satisfied in-house and its backlog item is deleted rather than deferred.

Negative consequences, accepted honestly:

- **The scoring enum cannot express a fourth combination without a new value.** A windowed Gaussian, or
  a coreset *within* each position, is a code change and a new enum entry rather than a second control.
  That is the cost of refusing the product, and it is only the right trade for as long as the invalid
  cell stays invalid.
- **One plugin holding three bank layouts reads harder than three plugins holding one each.** `fit` has
  a three-way branch, the checkpoint carries one of three payloads, and a reader has to hold all three in
  mind to change the shared feature path. The alternative was three copies of everything shared; this is
  the less bad option, not a good one.
- **The evidence for `local_knn` cannot come from public data.** VisA is unregistered — objects sit
  wherever they were photographed — so the mode whose entire premise is that position means something
  will look like a worse global bank on the only public benchmark this workbench uses. The property is
  demonstrated by a synthetic test that arranges the case deliberately, and the real evidence will have
  to come from registered multi-view captures that cannot be published. A reader is entitled to treat
  `local_knn` as unproven on public data, because it is.
- **The method ships with no public number beside it.** The VisA gate is an open roadmap item, and until
  it runs `dino_memory` is `experimental` in `docs/benchmarks/results.json` for the same evidence reason
  `glass_anomalib` is.
- **`local_gaussian` is shrinkage-dominated at its defaults.** 64 samples of 128 dimensions is 65 short
  of full rank, so λ is large and the Mahalanobis distance is closer to a scaled Euclidean one than the
  name suggests. `sample_deficit` is a field and the per-position λ is a diagnostic map precisely so this
  is visible rather than implied — but it does mean the mode's headline description overstates what the
  defaults compute.
- **The config model is large.** Seventeen fields, of which several apply to exactly one scoring value
  and are inert under the other two. ADR-0007 predicted the capability flags would become a weakly-typed
  catalogue of exceptions and ADR-0029 found the same pressure one level down in a method's options; this
  is that pressure again, and the same discipline is owed — a field nobody measures should be deleted.

[dino]: https://arxiv.org/abs/2405.14529
[padim]: https://arxiv.org/abs/2011.08785
[pc]: https://arxiv.org/abs/2106.08265
