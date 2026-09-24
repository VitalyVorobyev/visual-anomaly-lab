# ADR-0018: Model diagnostics as a declarative capability

**Status:** Accepted (2026-08-07)

## Context

A score and a heatmap say whether a model works, not *why*, and "why" is what separates a research
workbench from a batch script. A student-teacher method, for instance, fails in two distinguishable
ways, and the combined map it stores averages them together, discarding the one thing that would
explain the failure.

The requirement is sharper than "show some intermediate tensors": every diagnostic view must work
for a method that did not exist when the view was written. The obvious implementations — a typed
response model per method, a route per method, a frontend switch on the method's name — each make a
new method a UI change, which is what ADR-0007 exists to prevent.

There was also a live temptation to add a second event channel for training telemetry, since
per-step losses and learning rate are diagnostics in every ordinary sense.

## Decision

**Diagnostics are an optional, declared capability. A model pushes them into a self-describing
index, and the UI renders by `kind`, never by method name.**

- **A capability flag** says a method produces diagnostics; the UI offers the views from the flag.
- **One context call is the whole authoring surface**: a key, a title, a `kind` and a payload.
  `kind` says how to draw it — arrays (`map`, `image`, `grid`) are written as float32 `.npy`, and
  `graph` and `table` payloads are inline JSON. A diagnostic is scoped to one image or to the run.
- **Everything lives under the experiment's artifact directory**, described by an index file. No
  schema change, and deleting the experiment deletes its diagnostics. The index is written once,
  after the model returns, so a crashed run has arrays and no index — which reads correctly as
  "nothing usable".
- **Scalar series reuse the job protocol's `metric` event** (see ADR-0009). No second channel.
- **A disabled writer accepts every call and does nothing.** A plugin never asks whether
  diagnostics are wanted, so the capability is not a conditional threaded through it.
- **Per-image diagnostics are budgeted in the writer, and the truncation is recorded**, because a
  silent cap reads as "this is all there was".
- **Architecture is captured from a real forward pass**, not drawn, so it cannot go stale against
  the model it describes.

**Ruled out:** typed per-method response models and per-method routes (each makes a new method a UI
change); diagnostics in SQLite (a migration, for blobs that belong beside the maps); always-on
diagnostics (a long inference would spend most of its disk on them).

## Consequences

The views are written once, against the index, and a method gets all of them by emitting the same
keys. A method that emits nothing renders nothing, with no branch anywhere. The floor method emits
diagnostics with numpy alone, so the contract is testable without the deep-learning extra.

- **The contract is weakly typed by design.** `kind` says how to draw a payload and nothing about
  what it means; a misleading title produces a plausible, wrong picture.
- **Key agreement is by convention, not schema.** A view that compares two named maps silently
  shows less if a method names them differently; tests have to pin the keys.
- **Disk cost grows with usefulness.** The budget bounds it, so the answer to "show me this image's
  diagnostics" is often "that image was not in the sample".
- **`Capabilities` grew again**, exactly as ADR-0007 predicted.
- **Raw float32 arrays repeat ADR-0007's storage cost**, several per image rather than one.
