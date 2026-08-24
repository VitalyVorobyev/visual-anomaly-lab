# A new dataset, end to end

This workflow is the recommended order for an unfamiliar inspection problem. It minimizes the risk of
spending hours training a model against a data or evaluation mistake.

## 1. Define the unit of independence

Decide what one `Sample` means before importing. It should be the physical or logical unit that may appear
in exactly one split. If one part has top, side, and infrared images, those are three `Image` records of one
sample—not three samples. Otherwise near-duplicate views can leak between training and test.

Write down:

- how sample identity is recovered from paths or a table;
- which images are channels or views of the same sample;
- normal and defect labels, including unknown/unlabelled cases;
- whether a provider split is authoritative;
- where pixel masks live and which image coordinates they use.

## 2. Import without changing the source

Choose the adapter matching the evidence you have:

- `csv_table` when a CSV explicitly names paths, labels, channels, masks, or subsets;
- `folder_classes` when each image is a sample and folder names carry labels;
- `channel_folders` when filenames connect several channel folders into one sample.

Run scan, inspect warnings and counts, then commit the manifest. Do not “fix” an ambiguous scan by moving
source files until it looks right; configure or extend the adapter so the interpretation is repeatable.

## 3. Verify labels and masks

Browse a balanced selection of normal, defect, and unknown samples. Open several source masks over their
images. Masks offset by a resize or crop are worse than absent because pixel metrics will still produce
plausible numbers. Correct source metadata or create annotation revisions before training.

For unannotated defects, use the canvas polygon or brush workflow. Automatic contour derivation may propose
an edge-following contour from a manually marked region, but the user reviews and commits the resulting
geometry; it is never silent ground truth.

## 4. Establish the split

Adopt an official imported split when one exists. Otherwise generate a seeded sample-level split and inspect
class counts. Keep all images of a sample together. A normal-only training subset is conventional for
one-class methods; validation and test need both classes for image ROC-AUC.

Do not repeatedly redraw the split to improve a number. Create a new named split when the protocol changes.

## 5. Establish the image geometry

Start with the identity region profile. If the object occupies a small or unstable portion of the frame,
create a second profile using a deterministic classical localizer or MobileSAM and review its overlays over
the entire dataset. Measure missed defect pixels, not merely successful crops. Pin a new revision after each
configuration change.

Compare identity and localized profiles as experiments. Localization is useful only if source-frame pixel
metrics or stability improve; a smaller crop is not evidence by itself.

## 6. Run a method ladder

Use a ladder that changes one principle at a time:

1. `pixel_reference`—alignment-sensitive statistical floor;
2. `patchcore_anomalib`—frozen feature memory bank;
3. `efficientad_custom`—compact student–teacher plus reconstruction;
4. `dinomaly_custom`—transformer feature reconstruction when quality justifies longer fitting.

Keep split and prepared pixels fixed. Record resource caps before running. Use at least two public-safe
classes or subsets before promoting a method family; a single easy class can reward the wrong assumption.

## 7. Read failures before tuning

Filter to false positives and false negatives. Check whether failures cluster by channel, acquisition batch,
position, illumination, defect size, or annotation quality. Use branch diagnostics and raw values under the
cursor. Then form one hypothesis and encode it as an experiment field or profile revision.

Avoid an untracked preprocessing notebook between data and model. If a transform changes the pixels every
method should receive, implement it in the shared preparation pipeline. If it is architecture-specific
normalization, keep it inside the model.

## 8. Compare under one protocol

Use the comparison view only for experiments on the same dataset and split. Differences in preprocessing
remain allowed but are called out because they make the result partly a test of geometry. Compare
threshold-independent metrics first, then inspect per-run operating points and disagreement samples.

## 9. Freeze and deploy

Choose a supported exporter, create a verified bundle, and run its fixture through the Rust consumer on the
target hardware. Reproduce source-to-prepared geometry outside Python, verify the bundle after transfer, and
measure provider, latency, memory, and parity on that device. A bare ONNX file is not the deployable unit.

## Completion checklist

- Source provenance and licence are recorded.
- Sample grouping has no cross-split leakage.
- Masks visibly align in source coordinates.
- Split and region-profile revisions are pinned.
- At least one simple and one structurally different method were compared.
- False positives and false negatives were inspected.
- Quality claims name dataset, class, split, seed, configuration, and metric protocol.
- Export verification passes after copying only the bundle and runner to the target.
