# Model plugin interface

Every anomaly-detection method is a plugin behind one interface (ADR-0007). The rest of the application knows
only this interface and the registry key.

```python
# backend/src/anomaly_lab/models/base.py

class Capabilities(BaseModel):
    requires_training: bool          # PatchCore/EfficientAD yes; a pure-reference method may say no
    produces_anomaly_map: bool       # drives whether the UI offers overlay controls
    produces_diagnostics: bool       # drives whether the UI offers the inspector views (ADR-0018)
    channel_aware: bool              # model consumes channel metadata internally
    dataset_specific: bool           # True for classical_circular — surfaced as a UI warning
    preferred_device: Literal["cpu", "mps", "cuda"]


class ImageRecord(BaseModel):
    image_id: int
    sample_id: int
    channel: str | None              # canonical channel name, None for single-view datasets
    path: Path                       # absolute path to the source image, read-only


class Prediction(BaseModel):
    image_id: int
    score: float                     # higher = more anomalous
    anomaly_map: Path | None         # float32 .npy written into ctx.artifact_dir / "maps"
    inference_ms: float


class AnomalyModel(Protocol):
    @classmethod
    def config_model(cls) -> type[BaseModel]: ...
    @classmethod
    def capabilities(cls) -> Capabilities: ...

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None: ...
    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]: ...
    def save(self, artifact_dir: Path) -> None: ...
    def load(self, artifact_dir: Path) -> None: ...


# The registry is a table of *lazy loaders*, so opening the method picker does not import
# torch. That only holds while each plugin keeps its heavy imports inside its functions.
LOADERS: dict[str, Callable[[], type[AnomalyModel]]] = {
    "pixel_reference":       lambda: PixelReferenceModel,      # numpy + Pillow, the floor
    "efficientad_anomalib":  lambda: EfficientAdAnomalibModel,
    # "efficientad_custom":  M6 — second implementation, same interface (ADR-0008)
    # "patchcore_anomalib":  M7
    # "classical_circular":  M8, optional (ADR-0015)
}
```

A method whose optional dependencies are missing reports `availability.available = false` with the
command that installs them, and is **listed rather than hidden**: "why can't I pick EfficientAD"
should be answerable from the screen.

## Schema-driven configuration

`config_model()` returns a **pydantic model**, which the API exposes as JSON Schema at
`GET /api/experiments/model-types`. The frontend renders the experiment configuration form **directly from
that schema** — field types, defaults, ranges and descriptions all come from the Python side. Adding a
hyperparameter to a model therefore requires no frontend change at all, which is what makes "add a method
without touching the rest of the app" true in practice rather than aspirational.

## Contexts

`TrainContext` and `InferContext` carry everything a long-running plugin needs and must not invent for itself:

- **`artifact_dir`** — the experiment's directory; the only place a model writes its own outputs;
- **`cache_dir`** — shared, app-managed storage for downloaded assets (pretrained weights, the ImageNette
  penalty set). Separate from `artifact_dir` because these belong to the *method* and are reused across every
  experiment that runs it — a copy per run would be absurd;
- **`preprocessing`** — the resize and colour policy every method is made to share (below);
- **`progress(fraction, message)`** — forwarded to the job event stream ([the job system](jobs.md));
- **`metric(name, value, step)`** — scalar series; becomes a `metric` event, which is what per-epoch losses use;
- **`should_cancel()` / `raise_if_cancelled()`** — cooperative cancellation, polled at batch boundaries;
- **`log`** — structured logger whose records become `log` events in the job stream;
- **`emit_diagnostic(...)`** — the diagnostics contract (below, ADR-0018);
- `TrainContext.val` — the held-out normals, **empty when the split has no `val` subset**;
- `InferContext.write_map(image_id, array)` — persists one float32 map and accumulates the run's display range.

Everything above is *injected* (ADR-0014). Models never touch SQLite, never read application settings, and
never write outside `artifact_dir` and `cache_dir` — which is also what makes a plugin unit-testable with a
`NullReporter` and no job system at all.

## Preprocessing is configuration of the experiment, not of the model

A comparison between two methods only means something if they were shown the same pixels. Left to themselves
the libraries disagree, and the resulting difference in AUROC would partly measure the resize.

So `PreprocessingConfig` — size, colour mode, resampling filter — is stored on the `Experiment`, handed to the
plugin in its context, and **every plugin loads its pixels through one function**. A model that decodes an
image any other way is a bug, not a variation. Aspect ratio is deliberately not preserved: the resize goes
straight to the configured size, which makes an anomaly map a plain stretch back onto the source image, so an
overlay aligns without the UI reconstructing letterbox offsets.

## Diagnostics

A method that declares `produces_diagnostics` can show what it did as well as what it concluded, by
pushing into a self-describing index the UI renders **by `kind`, never by method name**. That is one
capability flag and one context method; the whole contract — authoring, the two read paths, the
architecture tree, on-demand entries and deletion — is on its own page:
**[diagnostics](diagnostics.md)**.

## Contract: scores are per-image

> **Models emit per-image scores. Cross-channel aggregation belongs to the evaluation layer ([the evaluation layer](evaluation.md)).**

This is the seam that keeps evaluation model-independent. A model *may* use channel metadata internally —
`ImageRecord.channel` is provided, and the classical baseline relies on it to keep one reference statistic per
channel (ADR-0010) — but it still returns one `Prediction` per input image. No model decides how a part's
three views combine into a sample-level verdict; that policy lives in one place and is applied identically
to every method.

## Anomaly map storage

Anomaly maps are written as **float32 `.npy`** arrays — the source of truth, lossless, directly usable for
recomputing statistics or (later) pixel-level metrics against masks. The API renders **colormapped PNGs on
demand** at `GET /api/images/{image_id}/anomaly-map?experiment_id=…`, caching the rendered PNG. Overlay
opacity is applied in CSS by the UI, never baked into the served image, so the opacity slider is instant and
requires no server round-trip.

## Device policy

Defaults target Apple Silicon: `preferred_device = "mps"` for the DL adapters, `"cpu"` for the classical
baseline (ADR-0008). Device is resolved at job start with a graceful fallback to CPU when MPS is unavailable
or an operator is unimplemented, and the resolved device is recorded in the job log.

---

# Classical baseline (summary)

`classical_circular` is the non-neural reference method. It was originally planned as the vertical slice's
first model, on the grounds that it needs no training infrastructure, no GPU and no external framework. That
ordering has been **superseded**: making the *showcase-specific* method the first one contradicted the
universal goal, so the slice is now proven with a dataset-agnostic method and a dataset-agnostic floor
baseline, and this method is scheduled later as an optional milestone. In outline (ADR-0010): a **circle
fit** on the part boundary with a **prior-based fallback** when the fit is poor; the resulting geometry is
**shared across all channels of a sample**, since the views are near-simultaneous images of the same physical
object; a **polar transform** about the fitted centre turns rotation into translation; **FFT angular
correlation** recovers orientation; a **per-channel median/MAD reference** is built from the training normals;
and scoring is a **percentile of the per-pixel z-score** map. It runs in seconds per sample on CPU. This
method is explicitly `dataset_specific = True` (above) — it is showcase-dataset-specific (circular parts),
exploiting the part's circular geometry, which the deep methods must not. The full algorithm, its parameters
and its failure modes are in **ADR-0010**.

---

