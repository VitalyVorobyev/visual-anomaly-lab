# Distilling an EfficientAD teacher

EfficientAD's teacher is a 2.7M-parameter PDN taught to reproduce the local features of a much larger
network. The paper distils a WideResNet-101 on ImageNet; every published teacher is somebody's run of that
procedure, and **the runs differ** — swapping one published teacher for another is the largest single effect
measured in this project ([measurements](https://github.com/VitalyVorobyev/visual-anomaly-lab/blob/main/docs/measurements.md),
ADR-0031). Distillation turns the teacher from an input you are handed into one you can measure.

**Inference cost does not change.** The source model is training-only; what ships is the same PDN.

Distillation produces a model *asset*, not an experiment, and it has no screen: it runs from the command
line.

## The recipe

A frozen `wide_resnet101_2` (`IMAGENET1K_V1`) sees an image at 512×512. Its `layer2` (stride 8, 512
channels) and `layer3` (stride 16, 1024 channels) are patch-aggregated — 3×3 neighbourhoods at stride 1,
layer3's patch grid resampled up to layer2's, each flattened patch adaptively pooled to 1024, the two
stacked and pooled again to 384 — giving a `(N, 384, 64, 64)` target. The PDN sees the same image at
256×256 **with padding on**, so its output is 64×64 too, and is trained by MSE against the
channel-normalized target (Adam, 1e-4, weight decay 1e-5).

This is the reference recipe on purpose: a different procedure would make every comparison against a
published teacher partly a measurement of the procedure. `tests/test_dl_efficientad_distill.py` pins the
aggregation against a transcription of the reference.

Detection runs the PDN with padding **off**, where the same weights give a 56×56 map that is padded
afterwards; padding changes a convolution's extent, not its weights.

## 1. Smoke test

A few minutes on the Imagenette corpus the student training already uses as its penalty set:

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name smoke --steps 200 --batch-size 2 --normalization-images 32 \
  > /tmp/distill-smoke.log 2>&1
```

Stdout is the job event stream in the application's JSON-lines format; the summary goes to stderr. A
student trained for a few dozen steps against a smoke teacher scores near or below chance — that proves the
chain holds together (corpus, source statistics, PDN grid, manifest, student load, evaluation), not
accuracy.

## 2. An Imagenette teacher

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name wrn-imagenette --steps 10000 --batch-size 4 \
  > data/jobs/logs/distill-wrn-imagenette.log 2>&1
```

Resumable: rerun the same `--name` to continue from the last checkpoint (`--no-resume` refuses an existing
name). Ctrl-C still writes the checkpoint.

Expect it to lose to the best published teacher: Imagenette is 13 394 images across 10 classes, the
reference 1.28M across 1000, and a teacher is a summary of the corpus it saw.

## 3. A full run over a large corpus

Not a laptop overnight job — measure the step rate from run 2 and multiply before starting; a full
reference-recipe distillation on Apple Silicon takes days.

| Corpus | Images | Size | Account? | Note |
|---|---|---|---|---|
| Imagenette | 13 394 | 1.5 GB | no | 10 classes, so narrow |
| COCO `train2017` | 118 287 | 19 GB | no | full resolution, natural scenes |
| ImageNet-1K `val` | 50 000 | 6.7 GB | yes | 1000 classes, the right *distribution* at 4 % of the size |
| ImageNet-1K `train` | 1 281 167 | ~150 GB | yes | the paper's corpus |

The teacher learns generic local features, so **breadth of scene content matters more than label
coverage**: COCO is a credible corpus with no account, ImageNet-1K `val` the cheapest route to the paper's
distribution. The corpus is recorded in the manifest, so two teachers are never confused for one.

```bash
uv run --directory backend python -m anomaly_lab.cli distill \
  --name wrn-imagenet --corpus directory \
  --corpus-path /path/to/ILSVRC/Data/CLS-LOC/train \
  --steps 60000 --batch-size 16 --normalization-images 10000 \
  > data/jobs/logs/distill-wrn-imagenet.log 2>&1
```

`--batch-size` fits the run into unified memory. Patch tensors are pooled in chunks, so their memory is
flat in batch size, but backbone activations are not — 4 is comfortable in 24 GB, 16 is the reference.

## 4. Train a student against it

Create an experiment with method `efficientad_custom`, set **Teacher source** to `distilled` and
**Distilled teacher** to the name you used; everything else is the normal training and evaluation path.

The load checks the teacher's recorded `model_size`, `out_channels` and preprocessing against the
experiment and refuses by name on a mismatch — a `.pth` of the right shape loads silently whether or not it
is the right teacher, so the manifest is checked, not the file.

## What is written

`data/model-cache/efficientad-teacher-distilled/<name>/`:

| File | What it is |
|---|---|
| `teacher.pth` | PDN weights, keyed `conv1…convN` |
| `distillation.json` | Source, layers, corpus and size, steps, full config, preprocessing, and the source's per-channel feature mean and standard deviation |
| `checkpoint.pt` | Weights, optimizer, step counter and corpus position, for resume |

The feature-normalization statistics are recorded although student training refits the teacher's own
statistics per dataset: two teachers distilled against differently normalized targets are different
teachers.

## Another source model

`FeatureSource` is the seam — `name`, `input_size`, `out_channels`, `features(batch)`, `close()`. A frozen
DINOv2 would be a second implementation and one branch in `build_source`; the loop, checkpoint, manifest and
student side do not move. DINOv2 uses 14-pixel patches, so its native grid is not 64×64 for a round input
size; the resampling to 64×64 belongs in the source, which is what `features()` returning a fixed grid
enforces.
