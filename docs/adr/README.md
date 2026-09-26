# Architecture Decision Records

These records hold **choices that had a live alternative** — what was decided, why, and what it
cost. They are not the system's documentation: to learn how the workbench works, read
[the handbook](../architecture/README.md); come here to find out why it is shaped that way.

## Decisions

| # | Decision |
|---|---|
| [0003](0003-tauri-to-python-boundary-is-a-fastapi-sidecar.md) | The Tauri-to-Python boundary is a FastAPI sidecar |
| [0004](0004-persistence-with-sqlite-and-filesystem-artifacts.md) | SQLite for metadata, files for artifacts; one schema script until a catalogue is worth keeping |
| [0005](0005-sample-owns-label-and-split-channel-is-data-not-schema.md) | The sample owns label and split; a channel is data, not schema |
| [0006](0006-import-via-pluggable-adapters-and-reviewable-manifest.md) | Import goes through pluggable adapters and a reviewable manifest |
| [0007](0007-common-model-plugin-interface-with-capability-flags.md) | Every method is one plugin behind one interface, with capability flags |
| [0009](0009-job-execution-subprocess-per-job-single-fifo-queue.md) | Every job is its own subprocess, drawn from one FIFO queue |
| [0012](0012-frontend-stack-and-generated-api-client.md) | Hash routing, TanStack Query, and an API client generated from OpenAPI |
| [0021](0021-design-token-layer-and-primitive-set.md) | Semantic tokens and primitives live once, in the shared `lab-ui` package |
| [0022](0022-private-source-data-lives-outside-the-working-tree.md) | Private source data lives outside the working tree |
| [0026](0026-a-resident-inference-worker-beside-the-job-queue.md) | One resident inference worker beside the queue, kept off the device by a lock |
| [0028](0028-comparing-runs-whose-scores-are-not-in-the-same-units.md) | Nothing is compared in score units; runs share a rule, never a number |
| [0029](0029-anomalib-is-the-baseline-not-the-specification.md) | We own the methods we keep; external libraries are baselines to measure against |
| [0032](0032-annotation-truth-is-versioned-and-source-frame.md) | Annotation truth is versioned and in the source frame |
| [0033](0033-region-profiles-pin-an-invertible-source-transform.md) | Region profiles pin an invertible source transform |
| [0034](0034-portable-models-are-verified-deployment-bundles.md) | Portable models are verified deployment bundles |
| [0038](0038-research-runs-outside-the-app-and-only-its-verdict-ships.md) | Research runs outside the app, and only its verdict ships |
| [0039](0039-a-task-is-frozen-on-the-experiment-and-chooses-its-evaluator.md) | A task is frozen on the experiment, and it chooses the evaluator |
| [0040](0040-few-shot-segmentation-is-a-task-and-its-references-are-a-split.md) | Few-shot segmentation is a task, and its references are a split |

Missing numbers belong to records that were removed; numbers are never reused.

## The rules of the record set

- **The handbook says what the code does now; a record says why it was chosen.** When they
  disagree, the handbook is right about *what* and the record about *why*.
- **The bar for a record has two parts, and both must be yes:** would a competent engineer
  plausibly have chosen otherwise, and would changing it now cost more than a refactor? A contract
  detail, a helper, a read path or a new option on an existing seam is handbook material.
- **A record is edited in place when its decision is refined**, so it reads as one coherent choice.
  It carries no changelog, no narrative of what a change did, no milestone tags and no measured
  numbers: git history is the changelog, and figures live in [measurements.md](../measurements.md).
- **A reversal gets a new number** and supersedes the old record explicitly, restating whatever it
  keeps from it.
- **A record whose truth has moved into the handbook is removed**, and every citation of it — in
  docs, the book, code comments, tests, skills, `CLAUDE.md` and `AGENTS.md` — is repointed in the
  same change. `scripts/check-doc-links.py` fails on a citation of a removed number.

Editing in place means a record can be rewritten to look prescient; what was believed at the time
lives only in git history, and a removed argument that comes back is recovered from `git log`. That
is taken knowingly.

## Format

- **Filename:** `NNNN-kebab-case-title.md`, `NNNN` the next unused number, zero-padded.
- **Status:** `Accepted`, or `Superseded by ADR-NNNN`, with the date it was reached.
- **Sections:** Context, Decision, Alternatives considered, Consequences — each a page or less.
- **Cross-references:** cite related records inline as `(see ADR-0007)`, and the handbook by page.
- **Consequences are honest.** A record with only upsides has not been thought through.

```markdown
# ADR-NNNN: Title in sentence case

**Status:** Accepted (YYYY-MM-DD)

## Context

The forces that make this a real question, for a reader who was not in the room.

## Decision

What was decided, in the active voice and the present tense, specific enough to constrain code.

## Alternatives considered

Each option that was live, and why it lost.

## Consequences

What becomes easier and what becomes harder — the costs accepted and the risks taken on.
```
