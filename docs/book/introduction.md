# Visual Anomaly Lab

Visual Anomaly Lab is a local desktop workbench for answering a practical question: **which anomaly
detection approach works on these images, under one reproducible protocol?** It imports image datasets,
supports source-frame defect annotation, prepares object regions, trains several method families, scores and
compares them, and exports proven models to a Rust consumer as verified ONNX bundles.

The application is intentionally a research instrument rather than a production reject station. It helps a
researcher inspect failure modes and establish evidence before building line integration around a chosen
model. Source images stay where they are; the catalogue stores paths and derived artifacts locally. There
are no accounts, cloud services, or telemetry.

![Scored samples filtered to mistakes](../images/gallery.jpg)

## The shortest useful loop

1. Register VisA or import an arbitrary image tree.
2. Check labels and masks; annotate missing defects if necessary.
3. Create or adopt a sample-level train/validation/test split.
4. Start with `pixel_reference`, then run one structurally different method.
5. Inspect false positives and false negatives, not only aggregate metrics.
6. Compare runs made on the same dataset and split.
7. Export a supported fitted model and verify it with the Rust runner.

The workbench preserves the decisions that make those results interpretable: source manifest, immutable
split, preprocessing and region-profile revision, method configuration, package and asset identity, raw
scores and maps, evaluation rule, and logs. A result can therefore be reopened and questioned instead of
becoming a number pasted into a notebook.

## What “universal” means here

The core does not assume a product shape, channel count, defect vocabulary, or directory layout. A logical
`Sample` may contain one or many `Image` records; channels are catalogue data. Only the optional
`classical_circular` method may assume circular geometry. Import adapters, deep methods, evaluation,
annotation, region preparation, and the UI remain dataset-agnostic.

“Universal” does not mean every method fits every dataset. It means the workbench makes assumptions and
limits visible, so methods can be compared without quietly changing the pixels or protocol between them.

## Reading this book

- Follow [Quick start](quick-start.md) for a first public-data run.
- Follow [A new dataset, end to end](new-dataset.md) when the data is yours.
- Use [Supported methods](generated/methods.md) and [Choosing a method](model-selection.md) to plan an
  experiment set.
- Read [Portable ONNX deployment](deployment.md) before treating an exported graph as a production model.
- Extension authors should begin with [Add a model](add-model.md) or
  [Add preprocessing or localization](add-preprocessing.md).

The [architecture handbook](../architecture/README.md) is the canonical description of current internals.
Decision records explain why consequential choices were made; this book concentrates on using and extending
the instrument.
