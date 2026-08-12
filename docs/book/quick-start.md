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
Create a split using the **Imported** strategy so the run follows the provider's one-class protocol.

If you use your own tree instead, continue with [Import and registration](import.md).

## Run the baseline

From the dataset workspace:

1. Open **Experiments** and choose **New experiment**.
2. Select the split and `pixel_reference`.
3. Leave the first configuration at its defaults and create the experiment.
4. Run **Train**. It builds a robust per-pixel reference from normal training images.
5. Run **Score & evaluate** on the test subset.
6. Open **Samples**, filter to **Mistakes**, and inspect the hottest false positives and false negatives.

`pixel_reference` is deliberately simple. If it performs well, alignment and position may dominate your
task. If it fails, its maps often reveal whether object localization, illumination normalization, or a more
semantic method is the next useful experiment.

## Add a second method

For a first contrast choose `patchcore_anomalib`: it does not optimize weights, but builds a bounded memory
bank of normal patch features. Keep the same dataset, split, preprocessing, and region profile. Compare the
runs from the dataset's experiment history.

Never compare raw score values across methods. A score only has meaning inside its run. The comparison view
uses threshold-independent metrics directly and resolves threshold-dependent outputs independently under one
named rule per run.

## Verify an export

`pixel_reference`, `efficientad_custom`, and `patchcore_anomalib` currently declare ONNX export. From a
trained experiment choose **Export ONNX**, then build and run the reference consumer:

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
