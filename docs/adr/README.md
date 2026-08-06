# Architecture Decision Records

This directory records the architectural decisions behind **visual-anomaly-lab**: what was decided,
why, and what it costs. Records are immutable once accepted — a decision that no longer holds is
superseded by a new record rather than edited in place.

The format is **MADR-lite**: one page, three sections, no ceremony.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-private-data-never-leaves-the-machine.md) | Private data never leaves the machine | Accepted |
| [0002](0002-monorepo-layout.md) | Monorepo layout | Accepted |
| [0003](0003-tauri-to-python-boundary-is-a-fastapi-sidecar.md) | Tauri-to-Python boundary is a FastAPI sidecar | Accepted |
| [0004](0004-persistence-with-sqlite-and-filesystem-artifacts.md) | Persistence with SQLite and filesystem artifacts | Accepted |
| [0005](0005-sample-owns-label-and-split-channel-is-data-not-schema.md) | Sample owns label and split; channel is data, not schema | Accepted |
| [0006](0006-import-via-pluggable-adapters-and-reviewable-manifest.md) | Import via pluggable adapters and a reviewable manifest | Accepted |
| [0007](0007-common-model-plugin-interface-with-capability-flags.md) | Common model plugin interface with capability flags | Accepted |
| [0008](0008-hybrid-dl-strategy-anomalib-now-custom-efficientad-later.md) | Hybrid deep-learning strategy — anomalib now, custom EfficientAD later | Accepted |
| [0009](0009-job-execution-subprocess-per-job-single-fifo-queue.md) | Job execution — subprocess per job, single FIFO queue | Accepted |
| [0010](0010-classical-circular-part-baseline-algorithm.md) | Classical circular-part baseline algorithm | Accepted |
| [0011](0011-evaluation-protocol-for-grouped-samples.md) | Evaluation protocol for grouped samples | Accepted |

## Conventions

- **Filename:** `NNNN-kebab-case-title.md`, where `NNNN` is the next unused number, zero-padded.
- **Status:** one of `Proposed`, `Accepted`, `Superseded by ADR-NNNN`, `Deprecated`, with the date
  the status was reached.
- **Length:** one page. If a record does not fit, the decision is probably two decisions.
- **Cross-references:** cite related records inline as `(see ADR-0007)`.
- **Consequences are honest.** Negative consequences are stated plainly. A record with only upsides
  has not been thought through.

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
```
