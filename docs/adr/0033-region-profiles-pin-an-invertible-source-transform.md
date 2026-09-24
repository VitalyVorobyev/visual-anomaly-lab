# ADR-0033: Region profiles pin an invertible source transform

**Status:** Accepted (2026-08-12)

## Context

Cropping a dominant object may remove nuisance background and improve an anomaly detector, or it
may erase useful context and make it worse. That is an experiment variable, not an import-time
correction. Meanwhile, annotations are immutable source-frame evidence (ADR-0032), methods must
receive identical pixels, and a predicted map must return to the source image without the UI
guessing how a crop was made.

The serious alternatives were to rewrite imported images, let each method crop for itself, store
only a mutable current crop per dataset, or treat localisation as best-effort with a silent fallback
to the whole image. Rewriting loses provenance. Per-method cropping destroys comparable input. A
mutable crop makes an old experiment unreproducible. Silent fallback mixes two input populations
under one experiment configuration.

## Decision

**A dataset owns immutable `RegionProfileRevision`s; an experiment pins one revision and its
completed build, and every spatial operation is a persisted, invertible source/prepared transform.**

- A profile revision freezes an extractor key and configuration, prepared size, padding, resampling
  filter, failure policy and seed. A new setting creates a new revision; a completed build is
  immutable, and a rebuild is a new revision.
- Extractors return a region in source pixel-edge coordinates. `identity` returns the full frame and
  is the default; other extractors are explicit opt-ins. **Region failure is explicit** and fails
  that image; it never silently becomes identity.
- One shared bridge expands, clips, crops, contain-resizes and edge-pads, and persists the integer
  resize and padding actually used rather than an ideal floating scale.
- The coordinate conventions are fixed once: integer points name pixel centres, crop bounds are
  half-open pixel edges, masks project nearest-neighbour, and float maps project back bilinearly
  with source pixels outside the crop marked uncovered.
- Source images and masks are never touched. Prepared pixels and per-image transforms are app-owned,
  bounded build artifacts that every method reads through the same preprocessing bridge, and a run
  freezes the digests of what it read.

## Consequences

- A result overlay projects back to the source without reconstructing state from UI settings, and
  paired runs can prove they saw the same prepared pixels.
- A contained resize preserves aspect ratio but introduces padding. Models may learn padding edges;
  edge padding reduces the discontinuity without removing the risk.
- Downsampling a binary mask is not invertible. Points round-trip to floating-point precision, masks
  only within a stated tolerance, and evaluation always uses source-frame truth.
- A failed extractor reduces usable coverage instead of quietly changing semantics: less convenient,
  more honest. Previews must show the failure rate before a build.
- Profiles add dataset-owned storage and lifecycle work. The immutable revision is kept separate
  from mutable preview and build status so operational progress cannot mutate the variable.
- Whether a non-identity profile helps is measured, not assumed; the verdict is in
  `docs/measurements.md`.
