#!/usr/bin/env -S uv run --project backend --extra dl python
"""Sweep `dinomaly_custom`'s encoder and decoder depth on identical public pixels.

The question the encoder field exists to ask: is Dinomaly's recorded result about the method,
or about DINOv2? And is the published decoder depth of 8 the right one for this data? The
protocol and its decision rule are predeclared in docs/measurements.md.

    ./scripts/dinomaly-encoder-sweep.py --data-dir /tmp/dinomaly-sweep \\
        --datasets-dir /path/to/datasets --weights-cache /path/to/model-cache

Every arm runs through the application's own train and infer jobs, one child process per
cell so peak RSS is comparable, on one identity prepared-input build per class at 448 x 448
— the one size both patch sizes divide, so the /14 and /16 encoders see the same pixels.
Each arm costs a full fit, so nothing can share a forward pass; the harness is the gate's,
not a campaign's (ADR-0038). Only the verdict ships.

`--weights-cache` copies already-downloaded encoder snapshots (a model-cache directory's
`huggingface/hub/models--timm--*`) into the isolated catalogue, read-only at the source, so a
sweep does not download what the machine already holds. Every finished cell is appended to
`cells.jsonl` as a self-describing row; `result.json` holds the whole report and the verdict.
A cell's checkpoint and maps are removed once its row is written, and a class's prepared pixels
once its cells are done, unless `--keep-artifacts`: the rows hold every number.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import m11_public_gate as gate

from anomaly_lab.config import Settings
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORIES = ("candle", "pcb1")
PREPARED_SIZE = 448
SEED = gate.SEED
METRICS = ("image_roc_auc", "pixel_roc_auc", "au_pro")

IMAGE_MARGIN = 0.01
"""A variant must beat the default's image ROC-AUC by at least this on *every* class."""

PIXEL_TOLERANCE = 0.01
"""...while losing no more than this on the two-class mean pixel ROC-AUC and AU-PRO."""


@dataclass(frozen=True)
class Arm:
    """One configuration. Everything not named here is the shipped default."""

    key: str
    encoder: DinoBackbone
    decoder_depth: int

    def config(self, steps: int | None) -> dict[str, Any]:
        values: dict[str, Any] = {
            "encoder": self.encoder.value,
            "decoder_depth": self.decoder_depth,
            "allow_downloads": True,
            "seed": SEED,
        }
        if steps is not None:
            values["max_steps"] = steps
        return values


DEFAULT_ARM = "default"
ARMS: dict[str, Arm] = {
    arm.key: arm
    for arm in (
        Arm(DEFAULT_ARM, DinoBackbone.DINOV2_VIT_S14_REG4, 8),
        Arm("dinov2_vit_b14", DinoBackbone.DINOV2_VIT_B14, 8),
        Arm("dinov3_vit_s16", DinoBackbone.DINOV3_VIT_S16, 8),
        Arm("depth_4", DinoBackbone.DINOV2_VIT_S14_REG4, 4),
    )
}


def _settings(data_dir: Path, datasets_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir.resolve(),
        reference_datasets_dir=datasets_dir.resolve(),
        dev_cors=False,
    )


def _seed_weights(source: Path, settings: Settings, arms: list[Arm]) -> list[str]:
    """Copy the arms' cached encoder snapshots into the isolated catalogue's cache."""
    hub = settings.model_cache_dir / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    copied = []
    for encoder in sorted({arm.encoder for arm in arms}):
        name = "models--timm--" + BACKBONES[encoder].timm_name
        found = source / "huggingface" / "hub" / name
        if found.is_dir() and not (hub / name).exists():
            shutil.copytree(found, hub / name, symlinks=True)
            copied.append(name)
    return copied


def _execute(data_dir: Path, datasets_dir: Path, experiment_id: int, log: Any) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(data_dir),
        "--datasets-dir",
        str(datasets_dir),
        "--_experiment-id",
        str(experiment_id),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write(completed.stderr)
    log.flush()
    if completed.returncode:
        raise RuntimeError(
            f"experiment {experiment_id} failed ({completed.returncode}); see gate.log"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"experiment {experiment_id} emitted an invalid child report")
    return dict(json.loads(lines[0]))


def _child(data_dir: Path, datasets_dir: Path, experiment_id: int) -> int:
    # The gate's child resolves its settings through a module-level helper; point it at the
    # datasets directory this sweep was given, then run the gate's own train-and-infer leg.
    gate._settings = lambda path: _settings(path, datasets_dir)  # type: ignore[assignment]
    return gate._child(data_dir, experiment_id)


def _cell_row(category: str, arm: Arm, run: dict[str, Any], steps: int | None) -> dict[str, Any]:
    infer = run.get("infer", {})
    images = int(infer.get("images") or 0)
    seconds = float(infer.get("inference_seconds") or 0.0)
    return {
        "category": category,
        "arm": arm.key,
        "encoder": arm.encoder.value,
        "decoder_depth": arm.decoder_depth,
        "max_steps": steps if steps is not None else 5_000,
        "prepared_size": PREPARED_SIZE,
        "seed": SEED,
        **{name: gate._metric(run, name) for name in METRICS},
        "train_seconds": run.get("train_seconds"),
        "test_images": images,
        "ms_per_image": 1000.0 * seconds / images if images else None,
        "peak_rss_bytes": run.get("peak_rss_bytes"),
        "artifact_bytes": run.get("artifact_bytes"),
        "experiment_id": run.get("experiment_id"),
    }


def _delta(cells: list[dict[str, Any]], arm: str, category: str, name: str) -> float | None:
    def value(key: str) -> float | None:
        row = next((c for c in cells if c["arm"] == key and c["category"] == category), None)
        return None if row is None else row[name]

    ours, base = value(arm), value(DEFAULT_ARM)
    return None if ours is None or base is None else ours - base


def _decision(cells: list[dict[str, Any]], categories: tuple[str, ...]) -> dict[str, Any]:
    """The predeclared rule, applied per variant against the default arm."""
    verdicts: dict[str, Any] = {}
    for key, arm in ARMS.items():
        if key == DEFAULT_ARM or not any(c["arm"] == key for c in cells):
            continue
        image = {c: _delta(cells, key, c, "image_roc_auc") for c in categories}
        mean_delta = {}
        for name in ("pixel_roc_auc", "au_pro"):
            values = [_delta(cells, key, c, name) for c in categories]
            known = [v for v in values if v is not None]
            mean_delta[name] = sum(known) / len(known) if len(known) == len(values) else None
        beats = all(v is not None and v >= IMAGE_MARGIN for v in image.values())
        keeps = all(v is not None and v >= -PIXEL_TOLERANCE for v in mean_delta.values())
        passes = beats and keeps
        verdicts[key] = {
            "image_roc_auc_delta": image,
            "mean_pixel_delta": mean_delta,
            "passes_rule": passes,
            # A gated encoder is never the default: an untouched run must not need a licence.
            "eligible_as_default": passes and not BACKBONES[arm.encoder].gated,
        }
    eligible = [key for key, v in verdicts.items() if v["eligible_as_default"]]
    winner = max(
        eligible,
        key=lambda key: sum(verdicts[key]["image_roc_auc_delta"].values()),
        default=DEFAULT_ARM,
    )
    return {
        "rule": {
            "image_margin_every_class": IMAGE_MARGIN,
            "pixel_tolerance_on_mean": PIXEL_TOLERANCE,
            "gated_encoders_never_default": True,
        },
        "variants": verdicts,
        "default_after_sweep": winner,
    }


def _run(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir
    gate._empty_destination(data_dir)
    settings = _settings(data_dir, args.datasets_dir)
    settings.ensure_directories()
    apply_schema(settings.db_path)
    arms = [ARMS[key] for key in (args.arms or tuple(ARMS))]
    categories = tuple(args.categories or DEFAULT_CATEGORIES)
    copied = _seed_weights(args.weights_cache, settings, arms) if args.weights_cache else []
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "VisA official 1cls split",
        "method": "dinomaly_custom",
        "prepared_size": PREPARED_SIZE,
        "seed": SEED,
        "smoke_steps": args.steps,
        "arms": {arm.key: asdict(arm) | {"config": arm.config(args.steps)} for arm in arms},
        "weights_copied": copied,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "packages": gate._packages(),
        "cells": [],
    }
    cells: list[dict[str, Any]] = report["cells"]
    rows = data_dir / "cells.jsonl"
    with (data_dir / "gate.log").open("a", encoding="utf-8") as log:
        for job_id, category in enumerate(categories, start=1):
            dataset_id, split_id = gate._official_dataset(settings, category, log)
            build = gate._identity_profile(
                settings,
                category=category,
                dataset_id=dataset_id,
                prepared_size=PREPARED_SIZE,
                job_id=job_id,
                log=log,
            )
            for arm in arms:
                spec = gate.CandidateSpec(
                    key="dinomaly_custom",
                    label=f"Dinomaly encoder sweep · {arm.key}",
                    family="transformer-reconstruction",
                    prepared_size=PREPARED_SIZE,
                    config=arm.config(args.steps),
                )
                experiment_id = gate._create_experiment(
                    settings,
                    category=category,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    method="dinomaly_custom",
                    method_config=spec.config,
                    build=build,
                    candidate=spec,
                )
                print(
                    f"Running {category} / {arm.key} as experiment {experiment_id}...",
                    file=sys.stderr,
                )
                run = _execute(data_dir, args.datasets_dir, experiment_id, log)
                row = _cell_row(category, arm, run, args.steps)
                cells.append(row)
                with rows.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
                if not args.keep_artifacts:
                    # The row holds every number; the checkpoint and the float maps are
                    # gigabytes per cell that nothing downstream reads.
                    shutil.rmtree(settings.experiment_dir(experiment_id), ignore_errors=True)
            if not args.keep_artifacts:
                shutil.rmtree(settings.region_profile_dir(build.profile.id), ignore_errors=True)
    report["decision"] = _decision(cells, categories)
    output = data_dir / "result.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"Full evidence: {output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--datasets-dir", type=Path, default=REPOSITORY / "datasets")
    parser.add_argument(
        "--weights-cache",
        type=Path,
        help="An existing model-cache directory to copy cached encoder snapshots from.",
    )
    parser.add_argument("--arm", action="append", choices=tuple(ARMS), dest="arms")
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the shipped 5000 only for a smoke run; the recorded sweep uses 5000.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep checkpoints, maps and prepared pixels (removed by default).",
    )
    parser.add_argument("--_experiment-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._experiment_id is not None:
        return _child(args.data_dir, args.datasets_dir, args._experiment_id)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
