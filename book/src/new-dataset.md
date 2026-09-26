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

Browse a balanced selection of normal, defect, and unknown samples. The sample viewer draws each image's
truth over it by default: a class region filled in its class's colour, a box as a rectangle tagged with its
class, an imported defect mask as an outline. The **Truth** switch and its opacity slider sit in the rail's
View section, with a legend of the classes the sample shows. Open several source masks over their
images this way. Masks offset by a resize or crop are worse than absent because pixel metrics will still produce
plausible numbers. Correct source metadata or create annotation revisions before training.

To see what a frozen encoder makes of the data before training anything, switch on **Explore** in the
sample viewer's rail. **Similar** marks every patch the encoder finds like the one clicked (shift-click
one it should not match), coloured over this image's own range, which the rail states; **Clusters**
groups the image's own patches, K from 2 to 12, with edges that follow the features rather than the patch
grid; **PCA** shows the
features' three main directions as false colour; **SAM** asks MobileSAM for masks; **Text** takes a
phrase — "candle", "the cap" — and asks SAM 3 for a mask of every instance of it, each in its own colour
and listed with SAM 3's score. SAM 3 finds objects and parts well and rarely finds a defect by the
defect's name, so ask for the thing rather than the flaw. Its weights are gated: request access at
[facebook/sam3](https://huggingface.co/facebook/sam3), set `HF_TOKEN` for the approved account (or sign
in with `hf auth login`), then accept the SAM License and download the 3.4 GB checkpoint once from the
Text mode itself. The first phrase after SAM 3 was idle loads it, which takes a while; a further phrase
on the same image takes a fraction of a second. It is for intuition:
nothing is stored or scored, and it draws above the truth, which dims while Explore is on. The first click on an image loads and runs the encoder, which takes seconds;
later clicks on it are immediate. **Send to editor** carries the current mask — the thresholded similarity,
the picked cluster, the chosen SAM mask or the picked SAM 3 instance — into the annotation editor as a
suggestion to accept or discard.

For unannotated defects, use the canvas polygon or brush workflow. Automatic contour derivation may propose
an edge-following contour from a manually marked region, but the user reviews and commits the resulting
geometry; it is never silent ground truth.

## 4. Establish the split

Adopt an official imported split when one exists — the **Published** preset on the Splits tab. Otherwise
start from a preset: **Standard · 60/20/20, normals only** for anomaly detection, **1-shot** or **5-shot**
for few-shot segmentation, **70/30 by class** for segmentation and detection. Each card shows the
composition its dry run produced; read the class counts before pressing **Create**. **Custom split** holds
the strategy form for anything else, with the same preview as you change it. A split no experiment ran on
can be deleted; one that holds runs cannot. Keep all images of a sample together. A normal-only training subset is conventional for
one-class methods; validation and test need both classes for image ROC-AUC.

Do not repeatedly redraw the split to improve a number. Create a new named split when the protocol changes.

## 5. Establish the image geometry

Start with the dataset's own **Full frame** profile — identity, created with the dataset, and what a run
reads unless told otherwise. If the object occupies a small or unstable portion of the frame,
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
