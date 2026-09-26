# Quick start

## Requirements

The primary target is an Apple Silicon Mac. Install:

- [uv](https://docs.astral.sh/uv/) for Python and backend dependencies;
- [Bun](https://bun.sh/) for the React frontend;
- the Rust toolchain from [rustup](https://rustup.rs/) for Tauri and the deployment runner;
- Xcode command-line tools: `xcode-select --install`.

Clone and install:

```bash
git clone https://github.com/VitalyVorobyev/visual-anomaly-lab.git
cd visual-anomaly-lab
uv sync --directory backend --extra dl
(cd frontend && bun install)
./scripts/dev-app.sh
```

Omit `--extra dl` for a small, torch-free installation. `pixel_reference`, import, annotation, evaluation,
comparison, and the full UI still work; deep methods appear unavailable with a reason.

For faster frontend iteration, run the same application in a browser:

```bash
./scripts/dev-backend.sh
./scripts/dev-frontend.sh
```

Open `http://localhost:5173`. The desktop shell adds native folder selection and Finder integration; core
workflows are otherwise the same.

## Get a public dataset

Download [VisA](https://github.com/amazon-science/spot-diff) into `datasets/VisA/` so that
`datasets/VisA/split_csv/1cls.csv` exists. Datasets under the repository's top-level `datasets/` directory
are ignored for size and must never be committed.

Open **Datasets**. A complete local VisA pack appears automatically. Choose an object class and select
**Register**. Registration reads the official CSV and masks in place; it does not copy or edit the source.
When it finishes, each dataset it added has **Start a run** beside **Browse**.

If you use your own tree instead, continue with [Import and registration](import.md); its finished screen
offers the same **Start a run**.

## Start a run

**Start a run** — on the dataset's page, the import's finished screen, or a catalogue card's corner — opens
the guided run: five steps, each one decision made in front of the dataset's own images, each with its
default already chosen. Nothing has to be prepared or split in advance.

1. **Goal.** What the run should do, suggested from the dataset's truth: anomaly detection for a dataset
   labelled normal and defect, detection for one annotated in boxes, few-shot segmentation for one
   annotated in regions. Each choice shows the samples it learns from — the defects, a class's examples.
2. **Look.** Where the run looks: **Full frame** unless a region profile says otherwise. One image is shown
   beside the frame the method will actually read, at the method's own size; ←/→ step through others.
   **Adjust on Prepare** tunes a profile and comes back with it.
3. **Split.** Which samples train and which are scored. The task's first preset is chosen, with a bar
   of what each subset holds and a few of its samples; for VisA you may prefer **Published**, the
   provider's one-class protocol. A preset is only drawn when the run starts.
4. **Method.** The registry's recommendation for the task first, the floor last — methods this
   installation cannot run are said so, and the first one it can run is chosen. The options in front are
   the ones a person decides; the rest are folded under **Advanced**.
5. **Run.** A summary of every choice, a name already filled in, and **Start run**: it draws the split,
   creates the experiment and queues **Train & score**, then opens the run page, which follows the job.

Pressing **Next** through every step gives a sensible run; **Review & run** jumps to the last step from any
other, so a first run is three presses from the dataset page. With only `pixel_reference` installed that
first run is the baseline: it prepares the full frame at 256 × 256, builds a robust per-pixel reference from
the normal training images, and scores the test subset. When it is scored, open **Samples**, filter to
**Mistakes**, and inspect the hottest false positives and false negatives.

`pixel_reference` is deliberately simple. If it performs well, alignment and position may dominate your
task. If it fails, its maps often reveal whether object localization, illumination normalization, or a more
semantic method is the next useful experiment.

**New experiment**, beside **Start a run**, is the full form for when you know what you want: channels,
colour, evaluation options and an explicit input size. Its **Create & run** does what **Start run** does;
**Create only** leaves the run as a draft.

## Add a second method

For a first contrast choose `patchcore_anomalib` in a second guided run: it does not optimize weights, but
builds a bounded memory bank of normal patch features. On the Split step choose the split the first run
drew — it is listed as existing — so both runs are measured on the same samples. Its native size is
448 × 448 rather than 256 × 256; to have both runs see identical pixels, use **New experiment** and type
256 × 256 into its size fields. Compare the runs from the dataset's experiment history.

Never compare raw score values across methods. A score only has meaning inside its run. The comparison view
uses threshold-independent metrics directly and resolves threshold-dependent outputs independently under one
named rule per run.

## Verify an export

Every currently registered method declares a parity-tested ONNX export. From a trained experiment choose
**Export ONNX**, then build and run the reference consumer:

```bash
cargo build --release --manifest-path deployment/runner/Cargo.toml
deployment/runner/target/release/anomaly-lab-runner \
  verify /path/to/experiment/exports/onnx-.../
```

`verify` checks every payload hash, runs the synthetic fixture, and enforces map and score parity. See
[Portable ONNX deployment](deployment.md) before preparing real source pixels outside the workbench.

## Resetting local state

Application metadata and derived artifacts live under the configured app data directory; in a development
checkout this is `data/`. Source datasets are not stored there. Stop the app and remove that directory only
when you deliberately want a fresh catalogue and no longer need its experiments or annotations.
