# Portable deployment

The deployable unit is a **verified bundle**, not a bare model file (ADR-0034). Its ONNX graph computes the
method's prepared-frame anomaly map; `manifest.json` carries the semantics a graph cannot: pixels, tensor
layout, score reduction, operating point, region provenance and parity tolerances.

## Bundle version 1

```
onnx-YYYYMMDDTHHMMSSZ/
├── manifest.json
├── model.onnx
└── fixture/
    ├── input.f32
    └── expected-map.f32
```

Every path in the manifest is POSIX-relative and traversal-safe. Every payload file has a byte count and
SHA-256. Float fixtures are little-endian, contiguous `float32`; their shapes and NCHW layout are in the
manifest. `format_version` is the compatibility boundary.

The first contract is intentionally static-shape, batch-one. An experiment already freezes width, height
and colour, so dynamic dimensions add runtime ambiguity without enabling a current workflow.

## Publication

`POST /api/experiments/{id}/export` enqueues the ordinary `export` job. The handler:

1. loads the experiment, pinned prepared-input build and fitted model;
2. checks `Capabilities.portable_formats` and the `SupportsOnnxExport` protocol agree;
3. writes into a job-specific staging directory;
4. creates a deterministic, dataset-free tensor and runs the Python reference path;
5. runs the ONNX graph with the CPU execution provider and checks map and percentile-score parity;
6. hashes every payload, writes the manifest, then atomically renames staging into `exports/`.

Cancellation or any exception removes staging. A parity failure publishes no bundle. Export is therefore a
claim about numerical behaviour, not merely about whether an ONNX parser accepts the graph.

## Runtime boundary

The runtime receives **prepared pixels**. Source cropping, contain-resize, padding and inverse map projection
belong to the pinned region profile and are identified in the manifest; a production integration must either
materialise that same transform or feed the exact prepared frame. Decode and colour conversion produce the
manifest's `[0,1]` NCHW tensor. Method-specific normalization stays inside the graph.

The graph emits one prepared-frame anomaly map. The host applies the named linear percentile reducer to get
the image score and compares it with the recorded operating point when one was available. Source-frame
projection is a host operation because production systems may own their source geometry independently.
An operating point is resolved from one named subset rather than a mixture: test first, then validation,
then train as an explicit last resort. The chosen subset and rule travel with the value.

`pixel_reference` is the first supported exporter. Other methods truthfully report no portable format until
their graph or auxiliary-tensor representation and parity tolerance have been implemented. The planned Rust
reference runner validates the same manifest and hashes before using pinned `ort` 2.0.0-rc.13 and ONNX
Runtime. `verify` executes the deterministic fixture and enforces both map and score tolerances; `infer`
accepts a prepared little-endian NCHW float32 tensor and emits a JSON score/verdict plus an optional raw map.
The reference binary uses CPU so it is a portable conformance oracle. The bundle itself does not depend on
that Rust crate or execution provider; a production runner may register the target's ONNX Runtime provider
without changing the bundle.

The CI handoff is deliberately cross-language: Python fits and exports the real baseline, then the compiled
Rust binary validates and executes that output. Rust-only tests would not catch schema or percentile drift
between the two implementations.

---

[← the handbook](README.md) · [methods](methods.md) · [why it is shaped this way](../adr/0034-portable-models-are-verified-deployment-bundles.md)
