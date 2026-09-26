# ADR-0007: Common model plugin interface with capability flags

**Status:** Accepted (2026-08-06)

## Context

Methods must be isolated behind one interface, so that a new one can be added without touching the
rest of the application, and evaluation must stay independent of any particular method.

The methods are genuinely dissimilar. A pixel-statistics floor trains in seconds on the CPU. A
memory-bank method fits nothing and builds a bank. A student-teacher network trains for minutes on
the accelerator and needs pretrained assets. Some want every channel of a sample; one would be
specific to a single dataset's geometry. An interface shaped after any one of them forces the
others into it.

The differences are not only internal: the UI must render a configuration form per method, decide
whether to offer an anomaly-map overlay, and know whether a train step exists at all.

## Decision

**One `AnomalyModel` abstract base class, declarative capability flags, and a name-keyed
registry.**

- **Lifecycle:** `fit`, `predict`, `save`, `load`.
- **Configuration** is a pydantic model. Its JSON Schema is served to the frontend, which generates
  the form. A hyperparameter is a Python field, never a UI change.
- **Capabilities** are declared flags — whether the method trains, produces a map, produces
  diagnostics, is channel-aware, is dataset-specific, can resume, can export, which tasks it
  supports, which device it prefers. The UI and the job layer branch on flags, **never on a
  registry key**.
- **The registry** maps stable keys to plugins; the key is what an experiment persists (see
  ADR-0005). It loads plugins lazily, so a heavy import stays inside its plugin.
- **Contexts** carry progress, cancellation and logging into `fit` and `predict`. A plugin knows
  nothing about subprocesses, queues or WebSockets (see ADR-0009).
- **`predict` returns per-image results.** Reducing a sample's images to one score is the evaluation
  layer's job (see the handbook's [evaluation](../architecture/evaluation.md) page); a
  channel-aware method may consult channel metadata but still emits per-image results.
- **Maps are stored raw, as float32 arrays** (see ADR-0004). Colormap, normalization and blending
  are applied at view time and never baked into stored data.

A new method is one module and one registry entry. If it needs a route, a schema change or a line
of TypeScript, the boundary is wrong and is fixed there, not in the caller.

## Alternatives considered

- **Per-method routes and screens.** Simple for two methods and unbounded after; every method
  becomes a frontend change and a new place for the evaluation protocol to diverge.
- **A UI that branches on the method's name.** Cheaper than a route per method, and it spreads
  knowledge of every method across every screen that shows one.

## Consequences

Several methods, and several implementations of one algorithm, coexist under their own keys and are
compared inside the app under one protocol. Because results stay per-image and maps stay raw,
evaluation and visualization choices remain open after the expensive computation is done.

- **The interface was designed from a few examples.** A method that does not fit `fit`/`predict` —
  online, or trained on negatives — will strain it, and may force a breaking change.
- **Capability flags proliferate.** Every "the UI needs to know whether…" becomes a flag, and the
  struct is a growing, weakly-typed catalogue of exceptions. Re-read this before adding one.
- **Generated forms are generic forms.** JSON Schema gives types and bounds, not conditional
  visibility or grouping; the result is plainer than a hand-built screen.
- **`dataset_specific` legitimizes non-generalizing code.** It is honest, and it makes adding more
  special cases easy.
- **Progress and cancellation are cooperative.** A plugin that never checks cannot be stopped
  politely; enforcement lives in ADR-0009's process boundary.
- **Raw maps cost storage**, several megabytes per full-resolution map.
