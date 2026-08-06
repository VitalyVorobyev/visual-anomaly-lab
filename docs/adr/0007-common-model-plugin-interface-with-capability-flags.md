# ADR-0007: Common model plugin interface with capability flags

**Status:** Accepted (2026-08-06)

## Context

The brief requires that "model implementations be isolated behind a common interface so that
additional methods can be added later without modifying the rest of the application", and that the
evaluation layer stay independent of individual models.

The three initial methods are genuinely dissimilar. The classical baseline is CPU-only, trains in
seconds, is deliberately specialized to the showcase dataset's circular parts, and wants to see all
channels of a sample. PatchCore builds a memory bank; EfficientAD trains a student-teacher network on
a GPU for minutes and needs an ImageNet subset. A fourth, `efficientad_custom`, will later reimplement
EfficientAD from scratch (ADR-0008). An interface that assumed any one of these shapes would force
the others into it.

The differences are also not merely internal: the UI must render a configuration form per model,
decide whether to offer an anomaly-map overlay, and know whether a "train" step exists at all.

## Decision

**A single `AnomalyModel` abstract base class, plus declarative capability flags, plus a name-keyed
registry.**

- **Lifecycle:** `fit(...)`, `predict(...)`, `save(path)`, `load(path)`.
- **Configuration:** `config_model()` returns a **pydantic model**. Its JSON Schema is served to the
  frontend, which **auto-generates the configuration form**. Adding a hyperparameter is a Python
  field, not a UI change.
- **Capabilities:** a `Capabilities` struct declaring `requires_training`,
  `produces_anomaly_map`, `channel_aware`, `dataset_specific`, `preferred_device`. The UI and the
  job layer branch on these flags rather than on model names.
- **Registry:** models register under stable keys — `classical_circular`, `efficientad_anomalib`,
  `patchcore_anomalib`, `efficientad_custom` — which are what an Experiment persists (ADR-0005).
- **Contexts:** `TrainContext` and `InferContext` carry a progress callback, a cancellation check,
  and a logger. A model reports progress and honours cancellation by calling these; it knows nothing
  about subprocesses, queues, or WebSockets (ADR-0009).

**Contract:** `predict` returns a **per-image** score, and optionally a per-image anomaly map.
Cross-channel aggregation to a sample-level score is **not** the model's job — it belongs to the
evaluation layer (see ADR-0011). A `channel_aware` model may consult channel metadata internally
(the classical baseline builds per-channel references), but it still emits per-image results.

**Anomaly maps** are written as **float32 `.npy`** (ADR-0004) as the source of truth. Colormapped
PNGs are rendered on demand for display; overlay opacity is a UI control, applied at view time.
Nothing about colormap, normalization, or blending is baked into stored data.

## Consequences

A new method is a file plus a registry entry: no route, schema, or UI change. Two implementations of
the same algorithm can coexist behind one key each and be compared inside the app — the point of
ADR-0008. Because scores stay per-image and maps stay raw, evaluation and visualization decisions
remain changeable after the expensive computation has been done.

Negative consequences, accepted honestly:

- **The interface is designed from three examples.** A method that does not fit `fit`/`predict` — an
  online or few-shot learner, or one needing negative examples during training — will strain it, and
  the first such addition will likely force a breaking change to the ABC.
- **Capability flags will proliferate.** Every new "the UI needs to know whether…" becomes a flag;
  the struct is a growing, weakly-typed catalogue of exceptions and will eventually encode
  distinctions no longer aligned with reality.
- **Auto-generated forms are generic forms.** JSON Schema gives us field types and ranges, not
  conditional visibility, grouping, or good defaults presentation. The result will look plainer
  than a hand-built screen — the brief asks for a professional engineering tool, and this is a
  concession against that.
- **`dataset_specific=True` is a legitimizing label for non-generalizing code.** It is honest, but
  it makes it easy to add more special-cased methods without confronting the cost.
- **Progress and cancellation are cooperative.** A model that never calls the cancellation check
  cannot be stopped politely; enforcement lives entirely in ADR-0009's process boundary.
- **Storage cost of raw maps** is significant (~5 MB per 1280x1024 float32 map).
