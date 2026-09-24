# Add a model

A new method should cost **one Python module and one registry entry**. If it requires a route, job kind,
evaluation branch, results schema, or method-specific TypeScript, stop and repair the plugin boundary.

## 1. Define the configuration

Create a Pydantic model in `backend/src/anomaly_lab/models/<method>.py`. Every option needs a useful
description because that schema is the UI. Prefer bounded numeric fields and enums over free strings. Defaults
must be safe and defined only in Python.

```python
class ExampleConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    max_reference_images: int = Field(
        default=256,
        ge=1,
        le=4096,
        description="Normals sampled evenly to build the reference.",
    )
```

## 2. Implement `AnomalyModel`

Provide title, summary, `config_model`, `capabilities`, `availability`, `fit`, `predict`, `save`, and `load`.
Heavy libraries such as torch, anomalib, timm, or transformers must be imported inside functions. Registry
description must remain cheap in a torch-free API process.

Read pixels only through `models.preprocessing.load_array` (or a helper built on it). Return one `Prediction`
per `ImageRecord`, with higher score meaning more anomalous. Write maps through the inference context. Never
read labels or masks in the plugin.

```python
class ExampleModel(AnomalyModel):
    title = "Example"
    summary = "A bounded normal-reference method."

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.CPU,
        )
```

## 3. Bound work before it starts

If value is not linear in dataset size, cap the work and sample with `evenly_spaced`. Print the complete plan
before extraction or training: images, patches/vectors, memory, steps, assets, and what will be dropped. A
memory-bank candidate pool must be bounded before allocation, not truncated after the machine swaps.

Seeds must reach every random stream. Test same-seed identity and different-seed difference. Library helpers
may use NumPy or global torch RNG even when your own generator is seeded.

## 4. Emit generic diagnostics

Use declarative `DiagnosticKind` payloads: scalar series, image, map, grid, table, or graph. The UI renders by
kind. Architecture introspection uses the shared forward-hook helper; do not draw a method-specific diagram.
Bound per-image diagnostics and state what was omitted.

## 5. Persist enough to reproduce and resume

Checkpoint format is method-owned and versioned. Save fitted state, resolved configuration, external asset
identity or fingerprint, fitted geometry, and all optimizer/scheduler/RNG state needed if `supports_resume`
is true. Reload on CPU and move to the requested device during execution. Refuse an asset mismatch by name.

## 6. Register it

Add one lazy loader and one key to `models/registry.py`. The method appears in the picker and its form is
generated automatically. No frontend edit is expected.

## 7. Test the claim, not just execution

At minimum:

- configuration/schema and capability tests in the torch-free suite when possible;
- predicting before fit is a named error, not zeros;
- save/load gives the same scores and maps;
- same seed is identical and different seed changes a stochastic fit;
- cancellation lands within a bounded unit;
- resource caps bind and are reported;
- a synthetic detection test asserts ROC-AUC/localization above a comfortable bar;
- every generic diagnostics view consumes the method without a special case.

A torch-dependent file must be named `test_dl_*.py` or CI will run it nowhere.

## 8. Add portable export only when proven

Declare `PortableFormat.ONNX` only after implementing `SupportsOnnxExport`. Export the complete method path,
return an explicit score contract, and implement `portable_reference`. Test Python-versus-ONNX map *and*
score parity. If operator coverage is missing, leave export unsupported; a truthful absence is better than a
button that changes semantics.

## 9. Run the gate

```bash
uv run --directory backend ruff check .
uv run --directory backend ruff format --check .
uv run --directory backend mypy
uv run --directory backend pytest
cd backend && uv run pytest tests/test_dl_*.py
```

Finally update [Supported methods](generated/methods.md) by running `scripts/build-book.py`, the methods
handbook, roadmap/backlog, and a public measurement log. Do not claim quality from the synthetic test.
