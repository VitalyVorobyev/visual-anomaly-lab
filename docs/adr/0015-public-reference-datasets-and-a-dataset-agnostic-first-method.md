# ADR-0015: Public reference datasets, and a dataset-agnostic method first

**Status:** Accepted (2026-08-07)

**Supersedes:** the sequencing claim in ADR-0010's context, and `system-design.md` §10's
statement that `classical_circular` is the vertical slice's first model.

## Context

The stated goal has always been a universal anomaly-detection explorer for arbitrary image
datasets, with the private showcase tree as its first reference dataset rather than its
scope. The delivery order contradicted that. M3 put `classical_circular` — the one method
deliberately specific to the showcase geometry — on the critical path as the *first* model,
which meant the first end-to-end proof of the architecture would be a proof that it works
for one dataset.

The design had absorbed that dataset in several other places besides. A channel vocabulary
sat in an adapter's defaults. "Image-level metrics only" was written into the evaluation
layer as a property of the system, when it was a property of the one dataset on hand having
no masks. Split guidance was reasoned around 98 normal and 91 defective samples. The media
layer was documented around 1280×1024 BMPs.

None of that is wrong for one dataset. All of it is wrong for a universal tool, and each
piece gets harder to remove the more code is built on top of it.

Two public benchmarks are now available locally: **VisA** (12 object classes, ~1000 normal
and 100 anomalous images each, **with pixel-level masks**, and official split tables) and
the **GKN Blade Surface Defect Dataset** (203 good, 48 nick, 149 scratch, no masks). Both
are CC BY 4.0. They are large and freely downloadable, so keeping copies in the repository
would add gigabytes to buy nothing.

## Decision

**The vertical slice is proved on a dataset-agnostic method against public benchmarks.
`classical_circular` moves off the critical path.**

- **EfficientAD via anomalib is the first real method**, per ADR-0008's hybrid strategy.
  A wrapper around a maintained implementation produces a number on VisA that can be
  checked against a published one, which is what makes the surrounding machinery
  trustworthy. The from-scratch `efficientad_custom` follows, with that number as its
  yardstick.
- **A dataset-agnostic floor baseline, `pixel_reference`, takes the slice's baseline
  slot.** Per-pixel median and MAD over the training normals, a z-map, and a
  high-percentile score: numpy and Pillow only, trains in seconds, works on any dataset.
  It is the geometry-free core of ADR-0010, so reviving `classical_circular` later means
  adding a circle-fit front-end rather than starting over.
- **`classical_circular` is deferred to a later, optional milestone.** ADR-0010 stays
  accepted and its algorithm stays valid; only its position in the order changes.
- **Public reference datasets are downloaded, never committed.** `/datasets/` is
  gitignored, the README says how to obtain each and credits both, and
  `scripts/check-repo-safety.sh` fails if anything under it is ever staged.
- **Correctness is measured against published numbers**, not only against our own
  regression baselines. A method whose reported figure is far from the paper's on the
  paper's own split has a bug, and that is a signal no self-consistent test suite can give.

## Consequences

The first end-to-end proof of the architecture is a proof that it works for datasets in
general, and the plugin boundary of ADR-0007 gets exercised by a real deep model rather
than by a method written to fit the interface. Pixel-level metrics become computable,
because VisA ships masks — which is what makes ADR-0017 possible at all.

Negative consequences, accepted honestly:

- **The critical path now depends on torch, anomalib and MPS.** The slice cannot complete
  if an operator has no MPS kernel, where `classical_circular` would have needed nothing
  but numpy. `pixel_reference` mitigates this — the whole results path is exercisable
  before torch is involved — but it does not remove the dependency.
- **A published number is a weaker check than it looks.** Preprocessing, resolution and
  aggregation all move the figure, so "in the neighbourhood" is the most this buys. A gap
  is acceptable; an *unexplained* gap is not, and telling the two apart takes judgement.
- **The showcase dataset is no longer what the system is proved against day to day.** Its
  irregularities — a capture group with two channels instead of three, mixed bit depths —
  are exactly the cases a universal tool must handle, and they now live only in an opt-in
  test that CI never runs.
- **`classical_circular` may never be built.** Deferring an optional milestone in a
  spare-time project is close to cancelling it, and ADR-0010 should be read as a design
  that exists rather than one that ships.
- **The reference datasets are not reproducible from the repository.** A clone plus
  `uv sync` does not get you a runnable benchmark; it gets you instructions. That is the
  price of not committing gigabytes, and the README carries the cost.
