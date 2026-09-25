#!/usr/bin/env -S uv run --project backend --extra dl python
"""Sweep `dino_memory`'s `layers` on DINOv3 ViT-S/16 against the recorded DINOv2 leg.

The question the recorded DINOv3 row leaves open: at the shared 448 px gate size DINOv3
ViT-S/16 trailed DINOv2 ViT-S/14-reg4 on every metric with `last_two`. Is that the encoder,
or the layers the recipe reads from it? The protocol and its decision rule are predeclared in
docs/measurements.md.

    ./scripts/dino-memory-layer-sweep.py --data-dir /tmp/dino-memory-layers \\
        --datasets-dir /path/to/datasets --weights-cache /path/to/model-cache

Every arm runs through the application's own train and infer jobs, one child process per
cell so peak RSS is comparable, on one identity prepared-input build per class at 448 x 448
— the one size both patch sizes divide, so the /14 and /16 encoders see the same pixels.
Each arm's configuration is the recorded gate candidate's (`dino_memory_v3`, or `dino_memory`
for the DINOv2 reference) with only `layers` changed, so the control arms re-run the recorded
rows. Nothing is trained: a fit is one encoder pass over the bank images and a coreset
selection, so there is no step budget to shorten.

The DINOv3 weights are licence-gated: an approved HF_TOKEN must be in the environment even
when the snapshot is already cached, because the hub checks access before it reads a cache.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import m11_public_gate as gate

from anomaly_lab.config import Settings
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone, FeatureLayers

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORIES = ("candle", "pcb1")
PREPARED_SIZE = 448
SEED = gate.SEED
METRICS = ("image_roc_auc", "pixel_roc_auc", "au_pro")

IMAGE_MARGIN = 0.01
"""A layer arm must beat the DINOv3 control's two-class mean image ROC-AUC by at least this..."""

CLASS_TOLERANCE = 0.01
"""...while losing no more than this image ROC-AUC on *any one* class..."""

PIXEL_TOLERANCE = 0.01
"""...and no more than this on the two-class mean pixel ROC-AUC and AU-PRO."""

GAP_TOLERANCE = 0.01
"""A DINOv3 arm within this of the DINOv2 reference on every two-class mean closes the gap."""

RECORDED = {
    "reference": {"image_roc_auc": 0.9000, "pixel_roc_auc": 0.9921, "au_pro": 0.9315},
    "control": {"image_roc_auc": 0.8147, "pixel_roc_auc": 0.9850, "au_pro": 0.8588},
}
"""The recorded two-class means the re-run arms are checked against — a check of construction,
not part of the rule."""


@dataclass(frozen=True)
class Arm:
    """One configuration: the recorded gate candidate with `layers` set."""

    key: str
    candidate: str
    backbone: DinoBackbone
    layers: FeatureLayers

    def config(self) -> dict[str, Any]:
        values = dict(gate.CANDIDATES[self.candidate].config)
        values["layers"] = self.layers.value
        # The recorded candidate names its backbone; the arm must agree with it.
        if values["backbone"] != self.backbone.value:
            msg = f"arm {self.key} names {self.backbone.value}, its candidate {values['backbone']}"
            raise ValueError(msg)
        return values


REFERENCE_ARM = "v2_last_two"
CONTROL_ARM = "v3_last_two"
ARMS: dict[str, Arm] = {
    arm.key: arm
    for arm in (
        Arm(REFERENCE_ARM, "dino_memory", DinoBackbone.DINOV2_VIT_S14_REG4, FeatureLayers.LAST_TWO),
        # Every value the field offers, read from the enum rather than listed by hand.
        *(
            Arm(f"v3_{layers.value}", "dino_memory_v3", DinoBackbone.DINOV3_VIT_S16, layers)
            for layers in FeatureLayers
        ),
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
    for backbone in sorted({arm.backbone for arm in arms}):
        name = "models--timm--" + BACKBONES[backbone].timm_name
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


def _cell_row(category: str, arm: Arm, run: dict[str, Any]) -> dict[str, Any]:
    infer = run.get("infer", {})
    images = int(infer.get("images") or 0)
    seconds = float(infer.get("inference_seconds") or 0.0)
    return {
        "category": category,
        "arm": arm.key,
        "backbone": arm.backbone.value,
        "layers": arm.layers.value,
        "layer_indices": list(arm.layers.indices),
        "config": arm.config(),
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


def _value(cells: list[dict[str, Any]], arm: str, category: str, name: str) -> float | None:
    row = next((c for c in cells if c["arm"] == arm and c["category"] == category), None)
    return None if row is None or row[name] is None else float(row[name])


def _mean(cells: list[dict[str, Any]], arm: str, categories: tuple[str, ...], name: str) -> Any:
    values = [_value(cells, arm, category, name) for category in categories]
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known and len(known) == len(values) else None


def _minus(ours: float | None, base: float | None) -> float | None:
    return None if ours is None or base is None else ours - base


def _decision(cells: list[dict[str, Any]], categories: tuple[str, ...]) -> dict[str, Any]:
    """The predeclared rule: which DINOv3 layers are recommended, and whose deficit it is."""
    means = {
        key: {name: _mean(cells, key, categories, name) for name in METRICS}
        for key in ARMS
        if any(c["arm"] == key for c in cells)
    }
    v3_arms = [key for key in means if ARMS[key].backbone is DinoBackbone.DINOV3_VIT_S16]

    # Question 1: does a layer choice replace `last_two` as DINOv3's recommended setting?
    variants: dict[str, Any] = {}
    if CONTROL_ARM in means:
        for key in v3_arms:
            if key == CONTROL_ARM:
                continue
            per_class = {
                category: _minus(
                    _value(cells, key, category, "image_roc_auc"),
                    _value(cells, CONTROL_ARM, category, "image_roc_auc"),
                )
                for category in categories
            }
            mean_delta = {
                name: _minus(means[key][name], means[CONTROL_ARM][name]) for name in METRICS
            }
            gains = (image := mean_delta["image_roc_auc"]) is not None and image >= IMAGE_MARGIN
            spares_each_class = all(
                v is not None and v >= -CLASS_TOLERANCE for v in per_class.values()
            )
            keeps_pixels = all(
                (v := mean_delta[name]) is not None and v >= -PIXEL_TOLERANCE
                for name in ("pixel_roc_auc", "au_pro")
            )
            variants[key] = {
                "image_roc_auc_delta_per_class": per_class,
                "mean_delta": mean_delta,
                "passes_rule": gains and spares_each_class and keeps_pixels,
            }
    passing = [key for key, v in variants.items() if v["passes_rule"]]
    recommended = max(
        passing,
        key=lambda key: variants[key]["mean_delta"]["image_roc_auc"],
        default=CONTROL_ARM if CONTROL_ARM in means else None,
    )

    # Question 2: is the recorded deficit the recipe's or the encoder's?
    gaps: dict[str, Any] = {}
    if REFERENCE_ARM in means:
        for key in v3_arms:
            gaps[key] = {
                name: _minus(means[key][name], means[REFERENCE_ARM][name]) for name in METRICS
            }

    def closes(key: str) -> bool:
        return all(v is not None and v >= -GAP_TOLERANCE for v in gaps[key].values())

    # A layer choice that reaches DINOv2 by trading one class for the other is not the recipe
    # fixing the encoder, so a closing arm must also pass question 1's rule.
    closing = [key for key in passing if key in gaps and closes(key)]
    if CONTROL_ARM not in gaps:
        verdict = "incomplete"
    elif closes(CONTROL_ARM):
        verdict = "gap_not_reproduced"
    elif closing:
        verdict = "recipe"
    elif len(gaps) < len(FeatureLayers):
        verdict = "incomplete"
    else:
        verdict = "encoder"

    reproduction = {
        role: {name: _minus(means.get(key, {}).get(name), RECORDED[role][name]) for name in METRICS}
        for role, key in (("reference", REFERENCE_ARM), ("control", CONTROL_ARM))
    }
    return {
        "rule": {
            "image_margin_on_mean": IMAGE_MARGIN,
            "image_tolerance_every_class": CLASS_TOLERANCE,
            "pixel_tolerance_on_mean": PIXEL_TOLERANCE,
            "gap_tolerance_on_every_mean": GAP_TOLERANCE,
        },
        "means": means,
        "variants": variants,
        "dinov3_recommended_layers": (
            None if recommended is None else ARMS[recommended].layers.value
        ),
        "gap_to_reference": gaps,
        "closing_arms": closing,
        "deficit_belongs_to": verdict,
        "reproduction_against_record": reproduction,
    }


def _run(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir
    gate._empty_destination(data_dir)
    settings = _settings(data_dir, args.datasets_dir)
    settings.ensure_directories()
    apply_migrations(settings.db_path)
    arms = [ARMS[key] for key in (args.arms or tuple(ARMS))]
    categories = tuple(args.categories or DEFAULT_CATEGORIES)
    copied = _seed_weights(args.weights_cache, settings, arms) if args.weights_cache else []
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "VisA official 1cls split",
        "method": "dino_memory",
        "prepared_size": PREPARED_SIZE,
        "seed": SEED,
        "arms": {
            arm.key: {
                "backbone": arm.backbone.value,
                "layers": arm.layers.value,
                "layer_indices": list(arm.layers.indices),
                "recorded_candidate": arm.candidate,
                "config": arm.config(),
            }
            for arm in arms
        },
        "recorded": RECORDED,
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
                    key="dino_memory",
                    label=f"DINO memory layer sweep · {arm.key}",
                    family="frozen-backbone patch memory",
                    prepared_size=PREPARED_SIZE,
                    config=arm.config(),
                    step_field=None,
                )
                experiment_id = gate._create_experiment(
                    settings,
                    category=category,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    method="dino_memory",
                    method_config=spec.config,
                    build=build,
                    candidate=spec,
                )
                print(
                    f"Running {category} / {arm.key} as experiment {experiment_id}...",
                    file=sys.stderr,
                )
                run = _execute(data_dir, args.datasets_dir, experiment_id, log)
                row = _cell_row(category, arm, run)
                cells.append(row)
                with rows.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), file=sys.stderr)
                if not args.keep_artifacts:
                    # The row holds every number; the bank and the float maps are what
                    # nothing downstream reads.
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
        "--keep-artifacts",
        action="store_true",
        help="Keep memory banks, maps and prepared pixels (removed by default).",
    )
    parser.add_argument("--_experiment-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._experiment_id is not None:
        return _child(args.data_dir, args.datasets_dir, args._experiment_id)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
