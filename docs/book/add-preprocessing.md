# Add preprocessing or localization

First decide which layer owns the change. This is more important than the implementation.

## Ownership test

| Question | Owner |
|---|---|
| Which source pixels depict the object? | Region localizer/profile |
| How are source pixels resized, padded, masked, and projected back? | Shared preparation |
| How does one architecture normalize its prepared tensor? | Model plugin |
| Is this logic valid only for one product geometry? | Dataset-specific plugin, never shared core |

If two methods receive different decoded/resized pixels without an explicit experiment difference, their
comparison is partly invalid.

## Add a region localizer

Implement the localizer protocol in the region package. Input is a source image plus validated configuration;
output is source-frame geometry and diagnostics—not a prepared tensor. Keep asset resolution explicit and
offline-capable. Bound whole-dataset processing and emit per-image failures rather than guessing identity.

Add synthetic fixtures covering non-square images, borders, no object, multiple components, and deterministic
output. For learned segmentation, pin asset revision and checksum and test missing/offline behavior.

Expose configuration through Pydantic JSON Schema. The region-profile UI should render it without a
localizer-name branch. Build creates an immutable manifest with geometry, inverse transform, hashes, timing,
and failure status.

## Add shared preparation

A new shared transform changes the experiment contract and portable input boundary. Specify:

- source and destination coordinate frames;
- interpolation and antialiasing;
- padding values and alignment;
- channel/color conversion;
- forward and inverse map behavior;
- uncovered source pixels;
- deterministic serialization.

Add it to the versioned region/preprocessing schema, not as an untracked callback. Test pixel-center
round-trips and source-map projection. Update the architecture handbook and deployment manifest/version if a
non-Python consumer needs new semantics.

## Add model-owned normalization

Keep normalization inside the plugin's forward/export path. It must run identically in train, predict,
on-demand diagnostics, resume, and ONNX export. Constants or fitted statistics belong in model state or graph
initializers. Test grayscale behavior explicitly rather than relying on broadcasting.

## Prove value

Run a paired public-data gate: same dataset, split, method, seed, and prepared size; identity profile against
the candidate. Predeclare minimum improvement and maximum regressions. Report preparation failures, crop or
mask coverage, missed defect pixels, build time, source-frame pixel metrics, and resource cost. A visually
tight object mask is not a success criterion.

The detailed current contract lives in the [methods](../architecture/methods.md) and
[domain model](../architecture/domain-model.md) handbook pages.
