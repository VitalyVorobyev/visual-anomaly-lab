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

A second choice sits inside the first: whether the input size belongs to the profile or to the run.
The size a method can read is the method's business — a ViT reads only frames its patch divides, and
each method's measured protocol ran at its own frame — while where to look is the dataset's. Putting
the size on the profile made every method that could not read the profile's size refuse it, and
recovering meant a new revision and a full rebuild for what is a property of the run.

## Decision

**A dataset owns immutable `RegionProfileRevision`s that say where to look; a run pins its own
input size; a build is one revision prepared at one size, and every spatial operation is a
persisted, invertible source/prepared transform.**

- A profile revision freezes an extractor key and configuration, padding, resampling filter and
  sample alignment — never a size. A new setting creates a new revision.
- The run's size is frozen into the experiment: named at creation, or the method's own native size
  for its configuration. A method's native size is the frame its recorded measurement ran at, so a
  run at its defaults reproduces a measured protocol.
- A build is keyed by `(revision, width, height)` and is immutable once published; a rebuild is a new
  revision. The run pins the manifest digest of the build it reads. The pin is set once — by the
  run's first train or infer job, which adopts the build at its size or prepares it — and never
  changes.
- Every dataset has an implicit identity revision, "Full frame". A run that names no profile reads
  it, so no run waits on a profile being defined or built.
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
- Two runs of one profile at different sizes read different builds; runs at the same size share
  one. Preparation cost moves into the first job of each new `(revision, size)`, which says so in
  its log.
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
