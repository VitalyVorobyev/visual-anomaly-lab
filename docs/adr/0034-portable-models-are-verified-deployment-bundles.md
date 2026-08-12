# ADR-0034: Portable models are verified deployment bundles

**Status:** Accepted (2026-08-12)

## Context

The workbench trains methods in Python, while the intended production consumer is a Rust process on
dedicated hardware. An ONNX file is only the numerical graph. It does not say how source pixels become a
tensor, how an anomaly map becomes an image score, which operating point was resolved for the run, or how
to project a prepared-frame map back onto the source image. Two consumers can therefore load the same graph
and produce different verdicts without either one reporting an error.

The serious alternatives were to export a bare graph, to serialize each Python implementation wholesale,
or to define one application-specific inference service and keep deployment coupled to Python. A bare graph
is incomplete. Python serialization is unsafe across environments and does not meet the Rust target. A
service preserves semantics but not the required offline, dedicated-hardware deployment boundary.

Not every method is naturally representable as one ONNX graph. A fitted memory bank, a preprocessing
normalization, and a score reducer may belong to different layers of the current implementation. Claiming
universal export before each method has numerical parity would turn a convenient button into a false
contract.

## Decision

**The portable unit is a versioned, checksummed deployment bundle whose graph is ONNX and whose semantics
are verified against the fitted experiment.**

- A bundle contains `manifest.json`, one or more ONNX graphs, immutable auxiliary tensors when the graph
  cannot own them, checksums, and deterministic parity fixtures.
- The manifest freezes source-to-tensor preparation, tensor names, shapes, dtypes and layouts, method-owned
  normalization, output/map semantics, image-score reduction, threshold provenance, region-profile
  requirements, package versions and source experiment identity.
- Export is a declared method capability and a structural protocol beside `AnomalyModel`. The generic job,
  API and UI branch only on that capability; no caller branches on a registry key.
- An exporter must run the Python model and the portable bundle on the same synthetic or public-safe
  fixtures and enforce stated map and score tolerances before publishing the bundle atomically. A failed
  parity check produces no export.
- The reference Rust consumer reads the manifest and validates hashes before loading a graph. Its first
  backend is ONNX Runtime through a pinned `ort` release because execution providers cover the dedicated
  hardware target. The bundle contract remains runtime-neutral; a tract consumer is valid when its operator
  coverage and parity are proven.
- The UI lists the formats a method can actually export. Unsupported methods explain the missing capability;
  they do not show an action that fails later.

## Consequences

- Deployment includes enough information to reproduce the workbench result rather than merely execute a
  graph, and parity fixtures make drift observable in CI and on the target.
- The bundle can be consumed outside this repository and outside Python. Hardware selection remains a
  runtime concern rather than changing the exported experiment.
- Export work is method-by-method. A standard contract does not make an unsupported operator, Python control
  flow or model-specific postprocessing portable by itself.
- ONNX Runtime is a native dependency and the Rust binding is still pre-2.0. The reference runner pins an
  exact version and is not the format specification; deployment owners may replace it without changing the
  bundle.
- Fixed experiment dimensions make the first contract intentionally static-shape and batch-one. Dynamic
  batching can be added as a later manifest version after it has a measured consumer.
