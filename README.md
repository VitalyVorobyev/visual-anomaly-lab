# Visual Anomaly Lab

**A local desktop workbench for visual anomaly detection.** Bring an image dataset, annotate defects,
prepare object regions, train several methods, inspect their mistakes, compare them under one protocol, and
export a proven model to ONNX for a Rust consumer.

Everything runs on your machine. There are no accounts, cloud services, or telemetry. Source images are
indexed where they already live and are never copied into the application workspace.

![Scored samples filtered to the model's mistakes](book/src/images/gallery.jpg)

## Five-minute start

You need [uv](https://docs.astral.sh/uv/), [Bun](https://bun.sh/), Rust, and—on macOS—Xcode command-line
tools.

```bash
git clone https://github.com/VitalyVorobyev/visual-anomaly-lab.git
cd visual-anomaly-lab
uv sync --directory backend --extra dl
(cd frontend && bun install)
./scripts/dev-app.sh
```

Omit `--extra dl` for a smaller install with the statistical baseline and every non-deep workflow.

In the app:

1. Open **Datasets** and register a local VisA pack, or import your own image tree.
2. Review labels and masks; annotate missing defects when needed.
3. Adopt the provider's split or create a seeded sample-level split.
4. Start a `pixel_reference` experiment, train it, then score the test subset.
5. Open **Samples**, filter to **Mistakes**, and inspect what the model actually found.
6. Add PatchCore or another method and compare runs from the same dataset workspace.

The complete walkthrough is in the **[Visual Anomaly Lab book](book/src/introduction.md)**, beginning with
[Quick start](book/src/quick-start.md) and [A new dataset, end to end](book/src/new-dataset.md).

## What the workbench gives you

- **Dataset-centered workflow.** Browse images, annotations, splits, region profiles, experiments, and
  experiment history from one dataset workspace. Filter history by method, status, split, or search text.
- **Truth that fits the task.** A sample's normal/defect label is anomaly truth; classes live in
  annotations. A dataset of classes — FSS-1000's few-shot panel, PKU-Market-PCB's boxes — is one dataset,
  counted, filtered and trained by class, and never dressed up as an anomaly dataset.
- **Source-frame annotation editor.** Polygon and region editing, undo/redo, automatic contour proposals,
  1:1 and fit views, double-click fit/restore, and direct drag-to-pan without a separate pan mode.
- **Object-region experiments.** Identity, classical localization, and MobileSAM-backed region profiles are
  versioned and pinned to experiments so preprocessing remains reproducible.
- **Several method families.** A transparent statistical floor, EfficientAD implementations, a bounded
  PatchCore memory bank, transformer reconstruction with Dinomaly, experimental learned synthesis, and a
  frozen DINO patch memory whose scoring rule — global, per position, or per-position Gaussian — is a field.
- **Evidence, not only scores.** Image and pixel metrics, source-frame maps, threshold exploration,
  false-positive/false-negative galleries, intermediate diagnostics, resource plans, and full job logs.
- **Honest comparison.** Runs share dataset and split; threshold-independent metrics compare directly, while
  each run resolves its own operating point because raw score units are not interchangeable.
- **Verified deployment.** Supported fitted methods export as checksummed ONNX bundles with deterministic
  parity fixtures and an independent Rust/ONNX Runtime reference consumer.

![One sample with prediction and ground truth](book/src/images/sample.jpg)

## Included methods

| Family | Methods | Current role |
|---|---|---|
| Statistical reference | `pixel_reference` | Fast CPU floor; ONNX export |
| Student–teacher + reconstruction | `efficientad_custom` | Compact deep reference; exports to ONNX |
| Feature memory bank | `patchcore_anomalib` | Short bounded fit; bank and paper score export to ONNX |
| Transformer reconstruction | `dinomaly_custom` | Ours; a choice of frozen encoder and a configurable decoder depth. Reached VisA parity with the anomalib wrapper it was measured against, which has since retired; does not export yet |
| Learned anomaly synthesis | `glass_anomalib` | Experimental; public gate did not promote it; ONNX export |
| Frozen-backbone patch memory | `dino_memory` | Ours; a DINOv2/DINOv3 patch memory scored globally, per position, or as a per-position Gaussian. Nothing is trained; no public gate run yet |

See the generated [method catalogue](book/src/generated/methods.md),
[selection guide](book/src/model-selection.md), and checked
[public benchmark report](book/src/generated/benchmarks.md) for capabilities, limitations, plots, and the
evidence behind those roles.

## Public reference data

Datasets are not bundled. Place downloads under the gitignored top-level `datasets/` directory and the app
offers a one-click registration when it recognizes a complete pack. Registration reads files in place.

- [VisA](https://github.com/amazon-science/spot-diff)—12 industrial object classes with official one-class
  splits and pixel masks, CC BY 4.0.
- [GKN Blade Surface Defect Dataset](https://doi.org/10.17632/3bh998k78g.1)—good, nick, and scratch images,
  CC BY 4.0.
- [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)—15 object and texture classes with
  pixel masks, CC BY-NC-SA 4.0, so **non-commercial use only**. MVTec distributes it behind a form; the
  mirror this repository's fetch script uses is
  [`TheoM55/mvtec_anomaly_detection`](https://huggingface.co/datasets/TheoM55/mvtec_anomaly_detection).
- [FSS-1000](https://github.com/HKUSTCV/FSS-1000)—1 000 object classes of ten images, each with a
  foreground mask (Li et al., CVPR 2020), registered as one dataset of a twenty-class few-shot panel whose
  masks are class annotations. The upstream
  repository publishes no licence and asks that users cite the paper; unzip its download so the classes
  sit under `datasets/FSS-1000/fewshot_data/`.
- [PKU-Market-PCB](https://robotics.pkusz.edu.cn/resources/datasetENG/)—693 printed-circuit-board images
  with six kinds of synthesised defect, every defect a Pascal VOC box of its kind (Huang and Wei,
  [arXiv:1901.08204](https://arxiv.org/abs/1901.08204)), registered as one detection dataset. The Open Lab
  on Human Robot Interaction at Peking University publishes it for detection, classification and
  registration research with no licence; cite the paper. Unzip `PCB_DATASET.zip` so that
  `datasets/PKU-PCB/PCB_DATASET/images/` and `…/Annotations/` exist.

Use the imported split to compare with a provider protocol. Any other tree can use the configurable CSV,
folder-class, or multi-channel adapters.

## Browser mode

The desktop shell is thin. For browser-based development or use:

```bash
./scripts/dev-backend.sh
./scripts/dev-frontend.sh
```

Open <http://localhost:5173>. Interactive API documentation is at <http://127.0.0.1:8000/docs>.

## Documentation

- **[The book](book/src/introduction.md):** user workflows, full pipelines, method choice, benchmarks,
  ONNX/Rust deployment, and extension guides.
- **[Architecture handbook](docs/architecture/README.md):** the canonical description of current internals.
- **[Development](book/src/development.md):** contributor setup, checks, safety, and evidence workflows.
- **[Roadmap](docs/roadmap.md):** shipped milestones and remaining exit criteria.

Build the local HTML book with `mdbook build book` and open `site/index.html`.

## Scope

Visual Anomaly Lab is a research workbench, not a production-line reject controller. It assumes one trusted
local user and trusted model code. Production integration begins from a verified export and must separately
validate source-image preparation, target execution provider, latency, memory, and hardware parity.

## Credits

Reference datasets remain under their authors' licences. Screenshots in this README use VisA imagery
(CC BY 4.0). Model implementations build on the cited papers and, where named, Intel's
[anomalib](https://github.com/open-edge-platform/anomalib).

`dino_memory` can optionally use Meta's DINOv3 encoder weights, which are distributed under the
[DINOv3 licence](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m): the download is gated,
access must be requested from Meta for your own Hugging Face account, and nothing is fetched unless you
select one of those two backbones. The method's default is an ungated Apache-2.0 DINOv2 encoder that
needs no account at all.

Explore's Text mode uses Meta's [SAM 3](https://github.com/facebookresearch/sam3) through Hugging Face
`transformers`. Its weights are distributed under the
[SAM License](https://huggingface.co/facebook/sam3): the download is gated, access must be requested for
your own Hugging Face account, and the checkpoint is fetched — at a pinned revision, verified file by file
— only when you accept that licence in the application.
