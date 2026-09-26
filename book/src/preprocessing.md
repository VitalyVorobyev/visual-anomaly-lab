# Object regions and preprocessing

The pipeline separates three responsibilities that are often conflated:

1. **Region localization** decides which source pixels represent the inspected object.
2. **Shared preparation** deterministically materializes the model input geometry.
3. **Method normalization** transforms those prepared values for one architecture.

This separation is what makes method comparison meaningful.

## Region profiles

A dataset owns versioned region profiles. A profile says **where to look**: identity, a classical detector,
or MobileSAM producing source-frame geometry. It has no size — the size is the run's. Every dataset starts
with one, **Full frame** (identity), and a run that names no other reads it, so you never have to define or
build a profile before training.

A build is one profile prepared at one size. It records, per image, the crop, resize, padding, inverse
transform, failure state, and hashes. A run's first **Train** (or, for a zero-shot method, its first
**Score**) prepares the build at the run's size if it does not exist yet — the job's log says so — and pins
its manifest; the run reads exactly those pixels for ever after. Two runs at the same size share one
build.

On **Prepare** you can preview a profile's crops and build it ahead of time. Preview and **Build all** name
a size, 448 × 448 unless you change it; building there only saves the first run at that size the wait.

On a grouped dataset a content-based localizer can find a slightly different box in each channel of one
part, which misregisters any method that fuses channels position by position. A profile's **crop per
sample** setting fixes this: *Shared* gives every image of a sample the union of their crops, so all of a
part's channels are prepared through one transform. The images of a sample must then share one size; a
sample that cannot be united fails as a whole, visibly, instead of falling back to per-image crops.

Identity is the correct starting point. Localization adds value when object pose or background dominates the
signal, but it can remove the very defect being measured. The public paired gate in
[`measurements.md`](https://github.com/VitalyVorobyev/visual-anomaly-lab/blob/main/docs/measurements.md) demonstrates the trap: a threshold
crop improved one image-level ROC-AUC while sharply degrading source-frame localization metrics by excluding
defect pixels.

## Available localizers

- **Identity:** full source frame, no localization failure mode.
- **Classical threshold:** deterministic foreground extraction for contrast-separated objects; inexpensive,
  but sensitive to illumination and background.
- **MobileSAM:** promptable segmentation used as a general deep proposal. The asset is local and pinned; a
  model response is converted into reviewed, versioned region geometry. By default it keeps the largest
  mask, which is often the background; setting a border limit and `union` selection finds the part
  instead, at the cost of sometimes cutting defects off — preview the crops against annotated defects
  before building.

Dataset-specific geometry belongs only in a dataset-specific plugin. Hough circles or fixed aspect-ratio
rules must not leak into shared preparation merely because they help one showcase.

## Shared preparation

`PreprocessingConfig` freezes the run's width, height and colour mode. Leave the size empty on the create
form and the run reads its method's **native size** — the frame the method's recorded gate ran at, which
the form shows beside the empty fields ("448 × 448 · from dino_memory"). A typed size snaps to the
multiple the method reads, the patch of a DINO backbone. `load_array` reads the prepared artifact into a
contiguous `[0,1]` array. Every method must use this bridge. The source map inverse projects
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
