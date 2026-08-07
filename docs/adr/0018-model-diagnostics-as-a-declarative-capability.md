# ADR-0018: Model diagnostics as a declarative capability

**Status:** Accepted (2026-08-07)

Extends **ADR-0007**. Nothing in that record is reversed: the plugin interface, the capability flags
and the registry stay as decided, and this adds one flag and one context method to them.

## Context

A score and a heatmap say whether a model works. They do not say *why*, and "why" is what separates
a research workbench from a batch script. EfficientAD in particular fails in two distinguishable
ways — the student failing to reproduce the teacher, and the autoencoder disagreeing with the student
— and the combined map that gets stored deliberately averages them together. Looking at the combined
map alone throws away the one thing that would explain the failure.

The visualization milestone (**M4**) is built entirely on this, and **M6** then reimplements
EfficientAD from scratch as `efficientad_custom`. So the requirement is sharper than "show some
intermediate tensors": every view M4 builds must work on a method that did not exist when the view
was written. The obvious implementations all fail that test — a `EfficientAdDiagnostics` response
model, a `/api/experiments/{id}/teacher-features` route, or a frontend `switch` on `model_type` each
make M6 a UI change, which is exactly what ADR-0007 exists to prevent.

There was also a live temptation to add a second event channel. Per-epoch losses and learning rate
are diagnostics in every ordinary sense of the word, and it would be natural to give them a
`diagnostic` event beside `progress` and `log`.

## Decision

**Diagnostics are an optional, declared capability that a model pushes into a self-describing index.
The UI renders by `kind` and never by method name.**

- **`Capabilities.produces_diagnostics`** joins the existing flags. The UI offers the views from the
  flag, not from a list of method names.
- **`ctx.emit_diagnostic(key, title, kind, payload)`** is the whole authoring surface. `kind` is one
  of `map`, `image`, `grid` (float32 arrays, written as `.npy`) or `graph`, `table` (inline JSON).
  Passing `image_id` scopes a diagnostic to one image; omitting it scopes it to the run.
- **Everything lands under `artifacts/exp-<id>/diagnostics/`, indexed by `diagnostics.json`.** No
  schema change, no migration, and deleting the experiment directory deletes the diagnostics with
  it. The index is written once, by the job handler, after the model returns — a run that crashed
  halfway has arrays but no index, which reads correctly as "this produced no usable diagnostics".
- **Scalar series reuse the existing `metric` event** from ADR-0009's protocol. `loss_st`,
  `loss_ae`, `loss_stae` and the learning rate are already exactly what that event is for, and
  inventing a second channel for the same data would have been the wrong kind of completeness. That
  the job protocol carried a neural network's training telemetry without modification is a test
  ADR-0009 passed.
- **A disabled writer accepts every call and does nothing.** A plugin never asks whether diagnostics
  are wanted; it calls `emit_diagnostic` unconditionally. That is the difference between a capability
  and a conditional threaded through the plugin.
- **Per-image diagnostics are budgeted, and the truncation is recorded.** Three float32 maps per
  image over a few hundred images is hundreds of megabytes. The budget is enforced in the writer, in
  one place, and the index records how many images were dropped — a silent cap would read as "this
  is all there was".
- **Architecture is captured from a real forward pass**, not hand-drawn, so M4's diagram cannot go
  stale against the model it claims to describe.

**Ruled out:** typed per-method response models (make M6 a UI change); per-method routes (same);
storing diagnostics in SQLite (a migration, for blobs that belong on the filesystem beside the maps);
and always-on diagnostics (a long inference run would spend most of its disk on them).

## Consequences

M4's visualizations are written once, against the index, and `efficientad_custom` gets all of them
in M6 by emitting the same keys. A method that emits nothing renders nothing, with no branch anywhere.
`pixel_reference` produces diagnostics with numpy and Pillow alone, which is what makes the contract
testable without the optional deep-learning group installed.

Negative consequences, accepted honestly:

- **The contract is weakly typed by design.** `kind` says how to draw a payload and nothing about
  what it means. A model emitting a `map` under a misleading `title` produces a plausible, wrong
  picture, and nothing here catches it.
- **Key agreement is by convention, not by schema.** M4's overlay comparison expects
  `map_student_teacher` and `map_autoencoder`; if `efficientad_custom` names them differently in M6
  the views will silently show less. That is a real coordination cost, and the honest mitigation is
  that M6's exit criteria say so out loud.
- **Diagnostics cost disk in proportion to how useful they are.** The budget bounds it, which means
  the default answer to "show me the diagnostics for this image" is "that image was not one of the
  first twelve".
- **`Capabilities` grew again.** ADR-0007 predicted the flag struct would become "a growing,
  weakly-typed catalogue of exceptions". This is one more, exactly as predicted, and the prediction
  is worth re-reading before the next one is added.
- **Storing raw float32 arrays repeats ADR-0007's storage cost** at several arrays per image rather
  than one.
