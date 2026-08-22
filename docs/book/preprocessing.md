# Object regions and preprocessing

The pipeline separates three responsibilities that are often conflated:

1. **Region localization** decides which source pixels represent the inspected object.
2. **Shared preparation** deterministically materializes the model input geometry.
3. **Method normalization** transforms those prepared values for one architecture.

This separation is what makes method comparison meaningful.

## Region profiles

A dataset owns versioned region profiles. A profile can use identity, a classical detector, or MobileSAM to
produce source-frame geometry. The prepared-input build records, per image, the crop, optional mask, resize,
padding, inverse transform, failure state, and hashes. An experiment pins the profile revision and build
manifest.

Identity is the correct starting point. Localization adds value when object pose or background dominates the
signal, but it can remove the very defect being measured. The public paired gate in
[`measurements.md`](../measurements.md) demonstrates the trap: a threshold
crop improved one image-level ROC-AUC while sharply degrading source-frame localization metrics by excluding
defect pixels.

## Available localizers

- **Identity:** full source frame, no localization failure mode.
- **Classical threshold:** deterministic foreground extraction for contrast-separated objects; inexpensive,
  but sensitive to illumination and background.
- **MobileSAM:** promptable segmentation used as a general deep proposal. The asset is local and pinned; a
  model response is converted into reviewed, versioned region geometry.

Dataset-specific geometry belongs only in a dataset-specific plugin. Hough circles or fixed aspect-ratio
rules must not leak into shared preparation merely because they help one showcase.

## Shared preparation

`PreprocessingConfig` freezes width, height, colour mode, and interpolation. `load_array` reads the prepared
artifact into a contiguous `[0,1]` array. Every method must use this bridge. The source map inverse projects
the prepared anomaly map into source coordinates before pixel evaluation or overlay.

## Method-owned normalization

ImageNet mean/std, teacher statistics, branch calibration, or feature normalization belong inside the model
when they are part of that architecture. They must be included in a portable graph or explicitly represented
in its bundle. Moving them into shared preprocessing would push one method's assumptions onto every other
method.

## Classical versus deep localization

Prefer the cheapest stable rule that covers the object and every possible defect. Classical connected
components, colour thresholds, morphology, edges, and Hough transforms are excellent when acquisition is
controlled. MobileSAM is useful when shape varies and a segmentation prior transfers. Neither should be
selected by visual neatness. Compare identity and localized profiles under source-frame metrics and record
build failures, missed defect pixels, preparation time, and crop coverage.
