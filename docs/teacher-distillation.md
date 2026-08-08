# Distilling the teacher

EfficientAD's teacher is a 2.7M-parameter PDN taught to reproduce the local features of a much
larger network. The paper distils a WideResNet-101 on ImageNet; every published teacher is
somebody's run of that procedure, and **the runs differ**. This page is how to do it here.

## Why this stage exists

Measured on VisA `candle` at a fixed 4000-step budget, changing nothing but which *published*
teacher was loaded:

| Teacher | Sample ROC-AUC | Pixel ROC-AUC | AU-PRO |
|---|---|---|---|
| `anomalib` | 0.769 | 0.891 | 0.539 |
| `nelson1425` | 0.889 | 0.978 | 0.914 |

Three seeds each, and the ranges do not overlap. That is the largest single effect measured
anywhere in this project — larger than 26 000 additional training steps — and it came from a
weight file nobody here produced. Distillation is what turns the teacher from an input we are
handed into one we can measure.

**Inference cost does not change.** The source model is training-only; what ships is the same
PDN. That asymmetry is the whole point of distillation and it is worth stating because
"WideResNet-101" otherwise sounds like a deployment cost.

## What it does

A frozen `wide_resnet101_2` (`IMAGENET1K_V1`) sees an image at 512×512. Its `layer2` (stride 8,
512 channels) and `layer3` (stride 16, 1024 channels) are patch-aggregated — 3×3 neighbourhoods
at stride 1, layer3's *patch grid* resampled up to layer2's, each flattened patch adaptively
pooled to 1024, the two stacked and pooled again to 384 — giving a `(N, 384, 64, 64)` target.
The PDN sees the same image at 256×256 **with padding on**, which is what makes its output
64×64 too, and is trained by MSE against the channel-normalized target. Adam, 1e-4, weight
decay 1e-5.

This is the reference recipe, deliberately. A teacher distilled by a different procedure would
make every comparison against a published teacher partly a measurement of the procedure. The
aggregation is pinned against a transcription of the reference in
`tests/test_efficientad_distill.py`, agreeing to 6e-8 — float32 reduction-order noise, not a
difference in meaning.

Detection later runs the PDN with padding **off**, where the same weights give a 56×56 map that
is padded afterwards. Both are true at once because padding changes a convolution's extent and
not its weights.

## Commands

### 1. Smoke test — a few minutes, nothing to download but the backbone

Imagenette is already on disk; it is the penalty set the student training uses.

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name smoke --steps 200 --batch-size 2 --normalization-images 32 \
  > /tmp/distill-smoke.log 2>&1
```

Stdout is the job event stream in the same JSON-lines format the rest of the application
produces, so redirecting it gives a readable log. The summary goes to stderr.

### 2. A real Imagenette teacher

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name wrn-imagenette --steps 10000 --batch-size 4 \
  > data/jobs/logs/distill-wrn-imagenette.log 2>&1
```

Resumable: rerun the same `--name` and it continues from the last checkpoint. Interrupt it
with Ctrl-C and the checkpoint is still written.

**Expect this to lose to `nelson1425`,** and treat that as the pipeline working. Imagenette is
13 394 images across 10 classes; the reference distils over ImageNet-1K, 1.28M images across
1000. A teacher is a summary of the corpus it saw.

### 3. The full run, over ImageNet-1K

Not a default, and not an overnight job on a laptop — measure the step cost from run 2 and
multiply before starting.

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name wrn-imagenet --corpus directory \
  --corpus-path /path/to/ILSVRC/Data/CLS-LOC/train \
  --steps 60000 --batch-size 16 --normalization-images 10000 \
  > data/jobs/logs/distill-wrn-imagenet.log 2>&1
```

`--batch-size` is the knob that fits the run into unified memory. The patch tensors are pooled
in chunks, so peak memory is flat in batch size rather than linear, but the backbone activations
are not — 4 is comfortable in 24 GB, 16 is the reference.

### 4. Train a student against it, and evaluate

In the application: create an experiment on VisA `candle`, method `efficientad_custom`, set
**Teacher source** to `distilled` and **Distilled teacher** to the name you used. Everything
else is the normal training and evaluation path.

The load checks the teacher's recorded `model_size`, `out_channels` and preprocessing against
the experiment before using it, and refuses by name on a mismatch. A `.pth` of the right shape
loads silently whether or not it is the right teacher, which is precisely why the manifest is
checked instead of the file.

## What is written

`data/model-cache/efficientad-teacher-distilled/<name>/`:

| File | What it is |
|---|---|
| `teacher.pth` | PDN weights, keyed `conv1…convN` — the layout this repository's PDN uses |
| `distillation.json` | The manifest: source, layers, corpus and its size, steps, full config, preprocessing, and the source's per-channel feature mean and standard deviation |
| `checkpoint.pt` | Weights, optimizer, step counter and corpus position, for resume |

The feature-normalization statistics are recorded even though student training refits the
teacher's own statistics per dataset. They are part of what the teacher *is*: two teachers
distilled against differently-normalized targets are different teachers, and a manifest that
omitted them could not tell you so.

## Extending it to DINOv2

`FeatureSource` is the whole seam — `name`, `input_size`, `out_channels`, `features(batch)`,
`close()`. A frozen DINOv2-S is a second implementation of it and one branch in `build_source`;
the loop, the checkpoint, the manifest and the student side do not move. Nothing is implemented
yet, on purpose: the immediate objective was a correct WideResNet baseline, and an abstraction
with one implementation is a guess until it has two.

The one thing to watch when it happens: DINOv2-S is a ViT with patch 14, so its native grid is
not 64×64 for any round input size. Whatever resampling makes it 64×64 is a decision that
belongs in the source, not in the loop — which is what `features()` returning a fixed grid is
there to enforce.

---

[← the handbook](architecture/README.md) · [the measurements](measurements-efficientad.md)
