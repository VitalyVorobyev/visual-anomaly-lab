# ADR-0038: Research runs outside the app, and only its verdict ships

**Status:** Accepted (2026-09-17)

## Context

Adding [SubspaceAD][paper] leaves the implementer choices no reading can settle: encoder family and
size, which band of transformer blocks to pool and whether the paper's band is relative or fixed,
input resolution, the explained-variance threshold and the tail fraction. Each is a measurement, and
they interact. That is a sweep of hundreds of configurations per benchmark category, across several
benchmarks and encoders. The question is where it runs.

**The workbench can already do this**, which is what makes it a real question. A script that created
one `Experiment` per configuration and enqueued one `Job` each would reuse datasets, splits, the
queue, `MetricSet` and the comparison screen, and needs no new results format. Three facts rule it
out:

- **The app's unit of work destroys the sweep's economy.** Almost every axis is free inside one
  forward pass — rank, tail fraction and layer band are prefixes or slices of what one pass already
  computed, and nested k-shot draws share a pass. Only the backbone and the resolution cost a pass.
  A `Job` is one configuration over one split and cannot share a pass with its siblings, so through
  the app the sweep is orders of magnitude more expensive.
- **The axes are not method options, and schema v1 is frozen.** Comparable across runs, each axis
  would become a persisted column and, under ADR-0004, a numbered migration — for variables that
  exist to be eliminated.
- **The benchmarks' protocol is not the app's data model.** Per-category official splits and a
  k-shot draw with augmented copies are not `Split`s, and importing them would make the import layer
  part of the measurement.

## Decision

**Research lives in `backend/research/`, and it is not the application.** It sits inside `backend/`
so one `ruff`, `mypy` and `pytest` cover it, and outside `backend/src/` so it is never packaged.

- **The dependency is one-way.** `research` imports `anomaly_lab`; `anomaly_lab` never imports
  `research`, and no registry, job handler or route refers to it. A research module is not a plugin.
- **What research needs from the product, it imports rather than reimplements.** Preprocessing goes
  through the application's own spatial transform and image preparation; metrics through
  `eval/metrics.py` and `eval/pixel.py`; and a shipped method's own arithmetic lives in
  `anomaly_lab/models/` and is imported back by the campaign, so the verdict is measured on the code
  the product runs. Where a research need does not fit, the application helper grows an optional
  parameter rather than being copied.
- **Results are self-describing JSON lines**, one row per arm per category, appended as produced. The
  file is the resume point; there is no database row to look a configuration up in.
- **Only the verdict ships:** one configurable plugin whose *defaults* are the winning configuration,
  one entry in `docs/measurements.md` recording protocol and number, and what the handbook needs.
  Losing arms become a sentence in the measurement record, not plugin options "in case".

## Consequences

- The sweep becomes affordable on a laptop, and an arm's pixels are the pixels a run would get.
- **Research gets none of the app's operational machinery** — no queue, cancellation, progress
  stream or results screen, only a resume flag and a log. A sweep has one user who can read a log.
- **Two code paths read images and agree only because they share code.** If someone "optimizes" the
  campaign by decoding images directly, every campaign number silently stops being comparable to a
  run. That is the failure mode to watch in review.
- **Nothing in a sweep is reproducible from the database**, so rows must stay self-describing; a row
  that names a configuration by abbreviation cannot be read next year.
- **The campaign can measure axes the product cannot express.** If the winner needs one, the plugin
  grows the field; the plugin never becomes a sweep harness with a UI.
- **Research is never packaged, so nothing verifies it against a released wheel.**

[paper]: ../papers.md
