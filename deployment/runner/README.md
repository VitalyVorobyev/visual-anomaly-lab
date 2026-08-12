# anomaly-lab-runner

Reference Rust consumer for a version-one visual-anomaly-lab deployment bundle. It validates the
manifest and every payload hash before opening ONNX Runtime, reproduces the declared linear-percentile
score, applies the recorded operating point, and emits one JSON object to stdout.

```sh
cargo run --release --manifest-path deployment/runner/Cargo.toml -- verify /path/to/onnx-bundle

cargo run --release --manifest-path deployment/runner/Cargo.toml -- infer /path/to/onnx-bundle \
  --input /path/to/prepared-nchw.f32 \
  --map-output /tmp/anomaly-map.f32
```

Inputs and map outputs are contiguous, little-endian `float32`. The tensor shape, NCHW layout, colour
policy, range and prepared-frame dimensions are in `manifest.json`. This reference executable uses the
CPU execution provider. The bundle is runtime-neutral; production builds can configure another ONNX
Runtime execution provider without changing the bundle.
