# Portable ONNX deployment

The portable unit is a **verified deployment bundle**, not `model.onnx` alone. The graph cannot fully express
source geometry, tensor preparation, score semantics, operating-point provenance, or file integrity.

## Export from the workbench

Train a method whose catalogue advertises ONNX, then choose **Export ONNX**. The generic export job:

1. loads the fitted model and pinned experiment inputs;
2. asks the plugin to write a static batch-one ONNX graph;
3. generates a deterministic, dataset-free prepared tensor;
4. runs both Python and ONNX Runtime reference paths;
5. enforces method-declared map and score tolerances;
6. hashes every payload and atomically publishes the directory.

Failure or cancellation removes staging. No unverified bundle appears in the export list.

## Bundle v2

```text
onnx-YYYYMMDDTHHMMSSZ-job-N/
├── manifest.json
├── model.onnx
└── fixture/
    ├── input.f32
    └── expected-map.f32
```

The manifest records source experiment identity and configuration, static NCHW float32 shapes, input colour
and `[0,1]` range, prepared/source coordinate contract, graph names and opset, anomaly-map direction, score
contract, operating point, pinned region revision, parity tolerances, and SHA-256/size for every payload.

Image score is a discriminated contract:

- `percentile_linear` for `pixel_reference`;
- `max` or `top_k_mean` for `efficientad_custom`;
- `tensor` for PatchCore's graph-produced, memory-bank-reweighted paper score.

This prevents a consumer from reproducing a heatmap while silently changing the accept/reject decision.

## Rust reference runner

```bash
cargo build --release --manifest-path deployment/runner/Cargo.toml
RUNNER=deployment/runner/target/release/anomaly-lab-runner
$RUNNER verify /path/to/bundle
$RUNNER infer /path/to/bundle /path/to/prepared-input.f32 --map-out /tmp/map.f32
```

`verify` parses and validates the manifest, rejects unsafe or symlinked payloads, checks sizes and hashes,
loads the graph through pinned `ort`, runs the bundled fixture, and emits a JSON parity report. `infer`
accepts exactly the already-prepared little-endian tensor described by the manifest and emits JSON containing
score, optional operating-point verdict, latency, and map range.

The reference runner uses ONNX Runtime's CPU provider as a conformance oracle. A hardware integration may
register CoreML, OpenVINO, TensorRT, CUDA, or a vendor provider supported by its ONNX Runtime build, but must
run `verify` on the target and record provider/version. Provider selection is deployment configuration; it
does not alter the bundle.

## Reproducing preparation in Rust

The current runtime boundary begins at prepared pixels. A production Rust application must:

1. decode the source with the declared colour interpretation;
2. reproduce the pinned region manifest's crop/mask/contain-resize/padding into fixed width and height;
3. scale channel values to `[0,1]` and arrange contiguous NCHW float32;
4. run the graph and score contract;
5. compare against the recorded operating point when that policy is appropriate;
6. inverse-project the prepared map with the pinned source transform for overlays or localization.

Do not substitute a convenient resize. A graph parity fixture proves the method path only; integration
parity needs a public source image and the source-to-prepared transform tested across Python and Rust.

## Hardware handoff checklist

- Copy only the runner and complete bundle.
- Disconnect network access if offline operation is required.
- Run `verify` after transfer and after runtime/provider upgrades.
- Record target, provider, ONNX Runtime version, latency distribution, peak memory, and parity error.
- Exercise normal, defect, boundary-sized, grayscale/RGB, and non-square source cases.
- Treat a changed bundle hash or preparation revision as a new deployed model.

The exact internal contract is in the [deployment handbook](../architecture/deployment.md) and its rationale
in [ADR-0034](../adr/0034-portable-models-are-verified-deployment-bundles.md).
