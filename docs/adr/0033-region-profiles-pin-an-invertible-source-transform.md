# ADR-0033: Region profiles pin an invertible source transform

**Status:** Accepted (2026-08-12)

## Context

Cropping a dominant object may remove nuisance background and improve an anomaly detector, or it may erase
useful context and make it worse. That is an experiment variable, not an import-time correction. Meanwhile,
annotations are immutable source-frame evidence (ADR-0032), methods must receive identical pixels, and a
predicted map must return to the source image without the UI guessing how a crop was made.

The serious alternatives were to rewrite imported images, let each method crop for itself, store only a
mutable current crop per dataset, or treat localisation as an optional best-effort operation whose failure
falls back to the whole image. Rewriting loses provenance. Per-method cropping destroys comparable input.
A mutable crop makes an old experiment unreproducible. Silent fallback mixes two input populations under
one experiment configuration.

## Decision

**A dataset owns immutable `RegionProfileRevision`s; an experiment pins one revision, and every spatial
operation is represented by a persisted, invertible source/prepared transform.**

- A profile revision freezes an extractor registry key and JSON configuration, prepared size, padding
  fraction, resampling filter, failure policy and seed. A new setting creates a new revision; it never edits
  one in place.
- Extractors return a region in source pixel-edge coordinates. `identity` returns the full source frame.
  Region failure is explicit and fails that image/profile build; it never silently becomes identity.
- The shared bridge expands a detected region, clips it, crops, contain-resizes and edge-pads. It persists
  the actual integer resize and padding used, not an ideal floating scale.
- Integer point coordinates name pixel centres. Crop bounds are half-open pixel edges. Point transforms use
  Pillow's half-pixel resize convention; masks use nearest-neighbour projection; float anomaly maps use
  bilinear inverse projection and mark source pixels outside the crop as uncovered.
- Source images and source masks remain untouched. Prepared pixels and per-image transforms are app-owned,
  bounded build artifacts. All methods read them through the same preprocessing bridge.

## Consequences

- A result overlay can be projected back without reconstructing state from UI settings, and paired runs can
  prove they saw the same prepared pixels.
- A contained resize preserves aspect ratio but introduces padding. Models may learn padding edges; edge
  padding reduces that discontinuity but does not remove the risk, so the value gate must measure it.
- Downsampling a binary mask is not mathematically invertible. Point transforms round-trip to floating-point
  precision; masks are tested under a stated pixel/IoU tolerance, and evaluation always uses source truth.
- A failed extractor reduces usable coverage instead of quietly changing semantics. That is operationally
  less convenient and scientifically more honest; previews must make the failure rate visible before build.
- Profiles add dataset-owned storage and lifecycle work. The immutable row is intentionally separate from
  mutable preview/build status so operational progress cannot mutate the experiment variable.

## Changelog

- **2026-08-12:** Made the resampling filter explicit on the immutable profile and documented the atomic,
  manifest-backed build that implements the already-decided bounded-artifact consequence.
