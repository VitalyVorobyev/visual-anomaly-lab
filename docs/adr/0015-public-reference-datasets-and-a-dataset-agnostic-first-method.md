# ADR-0015: Public reference datasets, and a dataset-agnostic method first

**Status:** Accepted (2026-08-07)

## Context

The goal is a universal anomaly-detection explorer for arbitrary image datasets; the private
showcase dataset is one reference dataset, not the scope. The original delivery order contradicted
that: it put a classical method built for the showcase parts' circular geometry on the critical path
as the *first* model, so the first end-to-end proof of the architecture would have proved it for
one dataset.

The showcase dataset had leaked into the design elsewhere too — a channel vocabulary in an
adapter's defaults, "image-level metrics only" written as a property of the evaluation layer when it
was a property of one dataset with no masks, split guidance reasoned around its counts. None of it
is wrong for one dataset; all of it is wrong for a universal tool, and it gets harder to remove as
code builds on it.

Public benchmarks exist that fix this: **VisA** (with pixel masks and official splits) and the
**GKN** blade-surface set, both CC BY 4.0. They are large and freely downloadable, so committing
them would add gigabytes and buy nothing.

The alternatives were to keep the showcase-first order and generalize later, or to vendor the
reference datasets for reproducibility.

## Decision

**The workbench is proved on dataset-agnostic methods against public benchmarks.**

- **The baseline floor is `pixel_reference`**: per-pixel median and MAD over the training normals,
  numpy and Pillow only, trains in seconds on any dataset. It is the geometry-free core of the
  classical method, so adding one later means a front-end, not a new method.
- **`classical_circular` is optional and deferred.** It is the only method that may assume anything
  about the showcase dataset's geometry, and nothing else may.
- **Public reference datasets are downloaded, never committed.** `/datasets/` is gitignored, the
  README says how to obtain each and credits it, and the repository safety check fails if anything
  under it is staged.
- **Correctness is measured against published numbers**, not only against our own regression
  baselines. A method far from the paper's figure on the paper's own split has a bug, and no
  self-consistent test suite can say so.

## Consequences

The architecture is proved for datasets in general, and the plugin boundary (see ADR-0007) is
exercised by real deep models rather than one written to fit it. Pixel-level metrics are computable
because the benchmarks ship masks.

- **The critical path depends on torch and MPS.** `pixel_reference` lets the whole results path run
  without torch, but deep methods cannot be proved without it.
- **A published number is a weaker check than it looks.** Preprocessing, resolution and
  aggregation all move the figure. A gap is acceptable; an *unexplained* gap is not, and telling
  them apart takes judgement.
- **The showcase dataset is no longer what the system is proved against day to day.** Its
  irregularities — a two-view capture group, mixed bit depths — are exactly what a universal tool
  must handle, and they live in an opt-in test CI never runs.
- **`classical_circular` may never be built.** Deferring an optional item in a spare-time project is
  close to cancelling it.
- **The reference datasets are not reproducible from the repository.** A clone gets instructions,
  not a runnable benchmark.
