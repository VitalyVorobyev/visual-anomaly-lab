# Architecture Decision Records

This directory records **choices that had a live alternative** — what was decided, why, and what it
cost. It is not the system's documentation. To learn how the workbench works, read
[the handbook](../architecture/README.md); come here to find out why it is shaped that way.

**The bar for a record** (ADR-0030): *would a competent engineer plausibly have chosen otherwise, and
would changing it now cost more than a refactor?* Both must be yes. A contract detail, a helper, a
read path for something already decided, or a new option on an existing seam is handbook material.

**Records are amendable.** An accepted record may be edited when the decision it describes is
refined, with a dated entry in a `## Changelog` section naming what changed. A **reversal** still
gets a new number and supersedes the old record explicitly — the superseded reasoning is what makes
the replacement legible.

**Numbers are permanent.** Citations of the form `ADR-NNNN` exist across the backend, the
frontend and the docs. A number is an address; it is never reused and never withdrawn.

## Decisions

The twenty-one that still settle something.

| # | Title | Area |
|---|---|---|
| [0002](0002-monorepo-layout.md) | Monorepo layout | Structure |
| [0003](0003-tauri-to-python-boundary-is-a-fastapi-sidecar.md) | Tauri-to-Python boundary is a FastAPI sidecar | Structure |
| [0004](0004-persistence-with-sqlite-and-filesystem-artifacts.md) | Persistence with SQLite and filesystem artifacts | Storage |
| [0005](0005-sample-owns-label-and-split-channel-is-data-not-schema.md) | Sample owns label and split; channel is data, not schema | Domain |
| [0006](0006-import-via-pluggable-adapters-and-reviewable-manifest.md) | Import via pluggable adapters and a reviewable manifest | Import |
| [0007](0007-common-model-plugin-interface-with-capability-flags.md) | Common model plugin interface with capability flags | Methods |
| [0008](0008-hybrid-dl-strategy-anomalib-now-custom-efficientad-later.md) | Hybrid deep-learning strategy — anomalib now, custom EfficientAD later | Methods |
| [0009](0009-job-execution-subprocess-per-job-single-fifo-queue.md) | Job execution — subprocess per job, single FIFO queue | Jobs |
| [0011](0011-evaluation-protocol-for-grouped-samples.md) | Evaluation protocol for grouped samples | Evaluation |
| [0012](0012-frontend-stack-and-generated-api-client.md) | Frontend stack and a generated API client | Frontend |
| [0015](0015-public-reference-datasets-and-a-dataset-agnostic-first-method.md) | Public reference datasets, and a dataset-agnostic method first | Steering |
| [0018](0018-model-diagnostics-as-a-declarative-capability.md) | Model diagnostics as a declarative capability | Diagnostics |
| [0021](0021-design-token-layer-and-primitive-set.md) | A design token layer, and primitives for the controls Tailwind does not have | Frontend |
| [0022](0022-private-source-data-lives-outside-the-working-tree.md) | Private source data lives outside the repository working tree | Safety |
| [0026](0026-a-resident-inference-worker-beside-the-job-queue.md) | A resident inference worker beside the job queue | Jobs |
| [0028](0028-comparing-runs-whose-scores-are-not-in-the-same-units.md) | Comparing runs whose scores are not in the same units | Evaluation |
| [0029](0029-anomalib-is-the-baseline-not-the-specification.md) | `efficientad_custom` is our implementation; anomalib is the baseline it beats or does not | Methods |
| [0030](0030-decisions-are-amendable-and-the-handbook-holds-current-truth.md) | Decisions are amendable, and the handbook holds current truth | Process |
| [0031](0031-the-teacher-is-an-experiment-variable-and-we-produce-it.md) | The teacher is an experiment variable, and we produce it | Methods |
| [0032](0032-annotation-truth-is-versioned-and-source-frame.md) | Annotation truth is versioned and source-frame | Annotation |
| [0033](0033-region-profiles-pin-an-invertible-source-transform.md) | Region profiles pin an invertible source transform | Spatial input |

## Folded

Eleven records whose current truth now lives in the handbook. Each keeps its number, its file and
its full text — the reasoning is still worth reading, and 658 citations still resolve. What changed
is that **you no longer have to read them to learn how the system works.**

Most exist because the old immutability rule left no other way to refine a decision: four of the
five diagnostics records are extensions of ADR-0018 that would today be an amendment to it.

| # | Title | Read instead |
|---|---|---|
| [0010](0010-classical-circular-part-baseline-algorithm.md) | Classical circular-part baseline algorithm | [methods](../architecture/methods.md) |
| [0013](0013-import-rescan-and-commit-semantics.md) | Import re-scan and commit semantics | [import](../architecture/import.md) |
| [0014](0014-shell-capabilities-are-injected-not-imported.md) | Shell capabilities are injected, not imported | [frontend](../architecture/frontend.md) |
| [0016](0016-adapters-for-public-datasets-masks-and-imported-splits.md) | Adapters for public datasets, masks in the catalog, imported splits | [import](../architecture/import.md) |
| [0017](0017-pixel-level-evaluation-at-constant-memory.md) | Pixel-level evaluation, at constant memory | [evaluation](../architecture/evaluation.md) |
| [0019](0019-serving-diagnostic-payloads-through-the-index.md) | Serving diagnostic payloads through the index, on a recorded scale | [diagnostics](../architecture/diagnostics.md) |
| [0020](0020-metric-series-are-replayed-from-the-job-log.md) | Metric series are replayed from the job log, not buffered | [jobs](../architecture/jobs.md) |
| [0023](0023-raw-values-beside-the-rendered-picture.md) | Raw values are served beside the rendered picture | [diagnostics](../architecture/diagnostics.md) |
| [0024](0024-layer-level-introspection-as-a-shared-helper.md) | Layer-level introspection is a shared torch helper | [diagnostics](../architecture/diagnostics.md) |
| [0025](0025-training-is-resumable-as-a-declared-capability.md) | Training is resumable as a declared capability, and steps are absolute | [jobs](../architecture/jobs.md) |
| [0027](0027-on-demand-diagnostics-are-first-class-and-deletable.md) | On-demand diagnostics are first-class in the index, and are deletable | [diagnostics](../architecture/diagnostics.md) |

## Superseded

| # | Title | Superseded by |
|---|---|---|
| [0001](0001-private-data-never-leaves-the-machine.md) | Private data never leaves the machine | [0022](0022-private-source-data-lives-outside-the-working-tree.md) |

## Conventions

- **Filename:** `NNNN-kebab-case-title.md`, `NNNN` the next unused number, zero-padded.
- **Status:** `Accepted`, `Folded into <page>`, `Superseded by ADR-NNNN` or `Deprecated`, with the
  date it was reached.
- **Length:** one page. If a record does not fit, the decision is probably two decisions.
- **Cross-references:** cite related records inline as `(see ADR-0007)`.
- **Consequences are honest.** Negative consequences are stated plainly. A record with only upsides
  has not been thought through — and a record whose Consequences section has aged into a to-do list
  should be amended, not reissued.

## Template

```markdown
# ADR-NNNN: Title in sentence case

**Status:** Accepted (YYYY-MM-DD)

## Context

The forces at play: the problem, the constraints, the facts about the domain or the dataset that
make this a real question. Written so that a reader who was not in the room understands why a
decision was needed. Name the alternatives that were seriously considered.

## Decision

What was decided, in the active voice and the present tense: "The boundary is a local FastAPI
server", not "we will probably use HTTP". Be specific enough that the decision constrains code.
State explicitly what is ruled out, and why the rejected options were rejected.

## Consequences

What becomes easier, and what becomes harder. State the negative consequences honestly — the costs
accepted, the risks taken on, the work deferred to a backlog. This section is what makes the record
worth re-reading later.

## Changelog

Only once a record has been amended. One dated line per amendment, naming what changed and why.
Never used to soften a claim that turned out to be wrong — that belongs in Consequences, stated.
```
