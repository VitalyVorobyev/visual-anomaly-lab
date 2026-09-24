# ADR-0034: Portable models are verified deployment bundles

**Status:** Accepted (2026-08-12)

## Context

The workbench trains methods in Python, while the intended production consumer is a Rust process on
dedicated hardware. An ONNX file is only the numerical graph. It does not say how source pixels
become a tensor, how an anomaly map becomes an image score, which operating point was resolved for
the run, or how to project a prepared-frame map back onto the source image. Two consumers can load
the same graph and produce different verdicts without either reporting an error.

The serious alternatives were to export a bare graph, to serialize each Python implementation
wholesale, or to keep deployment behind a Python inference service. A bare graph is incomplete.
Python serialization is unsafe across environments and does not reach the Rust target. A service
preserves semantics but not the required offline, dedicated-hardware boundary.

Not every method is naturally one ONNX graph. A fitted memory bank, a normalization and a score
reducer may live in different layers of an implementation. Claiming universal export before each
method has numerical parity would turn a convenient button into a false contract.

## Decision

**The portable unit is a versioned, checksummed deployment bundle whose graph is ONNX and whose
semantics are verified against the fitted experiment.**

- A bundle contains `manifest.json`, one or more ONNX graphs, immutable auxiliary tensors when a
  graph cannot own them, checksums, and deterministic parity fixtures.
- The manifest freezes source-to-tensor preparation, tensor names, shapes, dtypes and layouts,
  method-owned normalization, map semantics, image-score reduction, threshold provenance,
  region-profile requirements, package versions and source experiment identity. **Image scoring is
  an explicit, discriminated contract**, so a consumer never has to guess a method's score from its
  display map.
- Export is a declared method capability (`portable_formats`) and a structural protocol beside
  `AnomalyModel`. The job, API and UI branch only on that capability, never on a registry key, and
  the UI offers only formats a method can actually export.
- An exporter runs the Python model and the bundle on the same public-safe fixtures and enforces
  stated map and score tolerances before publishing atomically. A failed parity check produces no
  export.
- The reference Rust consumer validates hashes before loading a graph and runs on ONNX Runtime
  through a pinned `ort` release. The bundle contract is runtime-neutral; another runtime is valid
  once its operator coverage and parity are proven.

## Consequences

- Deployment carries enough to reproduce the workbench result rather than merely execute a graph,
  and parity fixtures make drift observable in CI and on the target.
- The bundle is consumable outside this repository and outside Python; hardware selection stays a
  runtime concern.
- Export is method-by-method. A standard contract does not make an unsupported operator, Python
  control flow or model-specific postprocessing portable, and several methods export nothing yet.
- ONNX Runtime is a native dependency and its Rust binding is pre-2.0, so the reference runner pins
  an exact version. It is not the format specification and may be replaced without changing bundles.
- The contract is static-shape and batch-one. Dynamic batching waits for a measured consumer and a
  new manifest version.
