# ADR-0038 — Research runs outside the app, and only its verdict ships

**Status:** Accepted (2026-09-17)

Extends **ADR-0007** (a method costs one module and one registry entry) and **ADR-0029** (a wrapper is
a baseline, not a specification) to the work that happens *before* a method exists. Constrained by
**ADR-0004** (schema v1 is frozen) and **ADR-0022** (private data lives outside the tree).

## Context

Adding [SubspaceAD][paper] is not the same shape of task as adding `dino_memory` was. The paper leaves
several things to the implementer that we have no prior on and cannot settle by reading: which encoder
family and size, which band of transformer blocks to pool, at what input resolution, whether the
paper's "layers 22–28 of 40" is a *relative* band or a fixed count of seven, and what the explained
variance threshold and the TVaR fraction should be. Every one of those is a measurement, and the
measurements interact — the right layer band on a 12-block ViT-S is not the right band on a 24-block
ViT-L.

That is a sweep of roughly 360 configurations per benchmark category, across three benchmarks, six
encoders and two input resolutions. The question is where it runs.

**The workbench can already do this, and that is what makes it a real question.** It has `Dataset`,
`Split`, `Experiment`, `Job`, a FIFO queue with cancellation and log tee-ing, `MetricSet`, and a
comparison screen that puts N runs of one split side by side. A `scripts/run-matrix.py` that created
one `Experiment` per configuration and enqueued one `Job` each would reuse all of it, would put every
arm's scores in the results UI where they could be browsed, and would need no new results format. It
was the first plan.

Three facts ruled it out.

**The app's unit of work destroys the sweep's economy.** The campaign is affordable only because of
what is free *inside* one forward pass. Because the retained basis is orthonormal, the residual at
every rank is a prefix of one running sum, so the variance threshold τ costs nothing once the
projection is computed at the largest rank any threshold selects. The image score is a prefix mean of
the sorted patch scores, so the TVaR fraction ρ costs nothing. The k-shot draws are nested, so one
pass over the fit pool snapshots k = 1, 2 and 4. The encoder returns the union of every view's blocks,
so the layer band is a slice. What actually costs a forward pass is the backbone and the input
resolution, and nothing else. A `Job` is one method configuration scored over one split — it cannot
share a pass with the 359 arms that differ from it only in a number read off the same features. Run
through the app, the sweep is not somewhat slower; it is about two orders of magnitude more expensive,
and it stops being a thing that can run on a laptop overnight.

**The axes are not method options, and schema v1 is frozen.** Which transformer blocks were pooled,
whether the encoder's final LayerNorm was applied to the intermediates, whether rotated pixels were
filled with zeros or masked out of the fit, how many rotations were used — these are the sweep's
independent variables. Through the app each would have to become a persisted column to be comparable
across runs, and ADR-0004 makes every one of them a numbered migration. Paying migration cost for
variables that exist to be eliminated is the wrong trade: the campaign's whole purpose is to answer
these once and then never ask again.

**The benchmarks are not datasets the app holds.** VisA is twelve categories with official one-class
splits, MVTec-AD is fifteen, and the paper's protocol draws k normals with thirty augmented copies
each. Expressing that as twenty-eight imported `Dataset` rows plus hand-built `Split`s would make the
import layer part of the measurement, and the protocol's k-shot draw is not a `Split` at all.

## Decision

**Research lives in `backend/research/`, and it is not the application.** It sits inside `backend/`
so that one `ruff`, one `mypy` and one `pytest` invocation cover it on the same terms as `src/`, and
outside `backend/src/` so that it is never packaged — the wheel lists `src/anomaly_lab` alone.

**The dependency is one-way and that is the load-bearing invariant.** `research` imports
`anomaly_lab`; `anomaly_lab` never imports `research`, and nothing in `models/registry.py`,
`jobs/handlers.py` or the API refers to it. A research module is not a plugin, gets no registry entry,
and cannot appear in the method picker.

**What research reuses, it reuses rather than reimplements.** Preprocessing goes through the
application's own `SpatialTransform.resolve` and `prepare_image`, so an arm's pixels are the pixels a
run would get; metrics go through `eval/metrics.py` and `eval/pixel.py`. This is the point of keeping
the code in one repository instead of a notebook: the alternative is a second normalization and a
second ROC-AUC, and then every comparison between a campaign number and a run number is partly a
comparison of two implementations. Where a research need does not fit — the per-arm cost of
recomputing a mask's connected components, say — the application's helper grows an optional parameter
rather than acquiring a copy in `research/`.

**Results are JSON lines, one row per arm per category, appended as they are produced.** Not SQLite,
not `MetricSet`. A sweep runs for hours on a machine that may be closed, so the file is the resume
point: a run skips the `(backbone, resolution, benchmark, category)` groups already present.

**Only the verdict ships.** The campaign's output reaching the product is exactly three things: one
configurable `subspace_ad` plugin whose *defaults are the winning configuration*, one entry in
[`docs/measurements.md`](../measurements.md) recording the protocol and the number, and whatever the
handbook needs to say about the method. Losing arms do not become plugin options "in case"; they
become a sentence in the measurement record.

## Consequences

**The sweep becomes affordable, and stays honest about preprocessing.** Phase 1 is hours rather than
weeks, and the identity-region path means an arm's normalization is the application's, not the
campaign's.

**Research gets none of the app's operational machinery, and reimplements a little of it.** No job
queue, no cancellation, no WebSocket progress, no results screen — a `--resume` flag and a log file
instead. That is a real loss and it is accepted: the machinery exists to make *user-initiated* runs
observable, and a sweep has one user who can read a log.

**Two code paths now read images, and they agree only because they share code.** The invariant is
enforced by construction rather than by test: research calls the same functions. If someone
"optimizes" the campaign by decoding images directly, every campaign number silently stops being
comparable to a run. That is the failure mode to watch for in review.

**Nothing in the sweep is reproducible from the database.** An arm's row carries its own
configuration — backbone, resolution, blocks, τ, ρ, k, seed, rotation fill, final-norm — because there
is no `Experiment` row to look it up in. Rows must therefore stay self-describing; a row that names a
configuration only by an abbreviation is a row that cannot be read next year.

**The campaign can measure things the product cannot express, and that is a hazard.** It sweeps
per-layer L2 normalization and the relative-versus-fixed reading of the layer band; if the winner
turns out to need an axis the plugin has no field for, the plugin grows the field. The rule is that
the plugin's *defaults* are the verdict, not that the plugin is a sweep harness with a UI.

**Research is excluded from the packaged artifact, so it is untested by deployment.** A `research`
module that imports something only present in the `dl` extra fails in the torch-free CI job like any
other, which is the intended coupling; but nothing verifies that `research/` still runs against a
released wheel, because it never does.

## Changelog

### 2026-09-18 — The method's own arithmetic moved into the package, and research imports it back

"What research reuses, it reuses rather than reimplements" was written about the application's
*helpers* — `SpatialTransform`, `eval/metrics.py`. Shipping the plugin turned it into a sharper
question: the covariance, the residual identity, the tail mean, the rotation augmentation and the
pixel map are the method, and they were sitting in `research/subspace_ad/`. Duplicating them in a
plugin would have meant the campaign's verdict was measured on code the product does not run, which
is the failure this record exists to prevent, one level deeper than it was originally aimed.

So `subspace.py` and `maps.py` moved to `anomaly_lab/models/subspace.py` and
`anomaly_lab/models/score_map.py`, `LayerBand` moved to `models/dino_backbone.py` beside the encoder
table it is about, and `research/subspace_ad/features.py` imports all of them back. The one-way
dependency is unchanged and is what makes the move legal in this direction and illegal in the other.

The check that this was worth doing is in [`measurements.md`](../measurements.md): on VisA `candle`
the plugin and the campaign produce identical patch counts, ranks inside each other's range, and
image AUROC within 0.0014 — with genuinely different fit-image selection, rotation streams and pixel
loading paths. Only the arithmetic is shared, and it is the part that agrees exactly.

[paper]: ../papers.md
