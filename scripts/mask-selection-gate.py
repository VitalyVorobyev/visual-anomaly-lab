#!/usr/bin/env -S uv run --project backend --extra dl python
"""Validate the MobileSAM border rule on public classes it was not designed on.

Protocol and decision rule: `docs/measurements.md`, "MobileSAM mask selection". The rule
(`max_border_fraction` and `selection`) was frozen from the *training normals* of the design
classes, which `--design` reproduces; the validation classes below never took part in it.

Every image is passed through MobileSAM once and its candidate masks are handed to
`select_region` under each configuration, so the rules differ only in how they choose.
A selection is scored against the image's ground-truth defect mask: the unpadded box must
keep at least `MIN_RETAINED` of the defect pixels and cover at most `MAX_BOX_FRACTION` of
the frame. An extraction failure retains nothing and is never correct.

    ./scripts/mask-selection-gate.py --out /tmp/mask-selection/result.json
    ./scripts/mask-selection-gate.py --design --out /tmp/mask-selection/design.json

Read-only over `/datasets`; writes one JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections.abc import Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.models.base import evenly_spaced
from anomaly_lab.regions.base import RegionExtractionError
from anomaly_lab.regions.mobile_sam import (
    ASSET_KEY,
    MobileSamRegionConfig,
    MobileSamRegionExtractor,
    border_fraction,
    select_region,
)

REPOSITORY = Path(__file__).resolve().parent.parent
VISA = "VisA_20220922"
MVTEC = "MVTec-AD"

DESIGN_CLASSES = ("candle", "capsules", "cashew", "chewinggum", "fryum", "pcb1")
DESIGN_PER_CLASS = 12
VISA_VALIDATION = ("macaroni1", "macaroni2", "pcb2", "pcb3", "pcb4", "pipe_fryum")
MVTEC_OBJECTS = (
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
)
MVTEC_TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
PER_CLASS = 24

MIN_RETAINED = 0.99
MAX_BOX_FRACTION = 0.90
CRITERION = {
    "minimum_correct_rate_gain": 0.20,
    "minimum_pooled_retention": 0.98,
    "maximum_failure_rate": 0.05,
}
RULES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "candidate": {"max_border_fraction": 0.33, "selection": "union"},
    # Reported, not decided on: why the candidate unites rather than picks one mask.
    "largest_interior": {"max_border_fraction": 0.33, "selection": "largest"},
}


def _checkpoint(model_cache: Path) -> Path:
    spec = get_spec(ASSET_KEY)
    path = model_cache / "assets" / ASSET_KEY / spec.filename
    if not path.is_file() or path.stat().st_size != spec.expected_size:
        raise SystemExit(
            f"MobileSAM checkpoint is not cached at {path}; this gate downloads nothing"
        )
    return path


def _visa(
    datasets: Path, classes: tuple[str, ...], split: str, label: str
) -> Iterator[tuple[str, Path, Path | None]]:
    root = datasets / VISA
    rows = list(csv.DictReader((root / "split_csv" / "1cls.csv").open()))
    for name in classes:
        chosen = sorted(
            (
                row
                for row in rows
                if row["object"] == name and row["split"] == split and row["label"] == label
            ),
            key=lambda row: row["image"],
        )
        limit = DESIGN_PER_CLASS if split == "train" else PER_CLASS
        for index in evenly_spaced(len(chosen), limit):
            row = chosen[index]
            yield name, root / row["image"], (root / row["mask"]) if row["mask"] else None


def _mvtec(datasets: Path, classes: tuple[str, ...]) -> Iterator[tuple[str, Path, Path | None]]:
    root = datasets / MVTEC
    rows = list(csv.DictReader((root / "metadata.csv").open()))
    for name in classes:
        chosen = sorted(
            (
                row
                for row in rows
                if row["object"] == name and row["split"] == "test" and row["mask_path"]
            ),
            key=lambda row: row["path"],
        )
        for index in evenly_spaced(len(chosen), PER_CLASS):
            row = chosen[index]
            yield name, root / "images" / row["path"], root / "masks" / row["mask_path"]


def _score(
    records: list[dict[str, Any]], defect: np.ndarray, config: MobileSamRegionConfig
) -> dict[str, Any]:
    height, width = defect.shape
    total = int(defect.sum())
    try:
        extraction = select_region(records, height, width, config)
    except RegionExtractionError:
        return {
            "failed": True,
            "retained": 0,
            "defect": total,
            "box_fraction": None,
            "correct": False,
        }
    bounds = extraction.bounds
    left, top = int(np.floor(bounds.left)), int(np.floor(bounds.top))
    right, bottom = int(np.ceil(bounds.right)), int(np.ceil(bounds.bottom))
    retained = int(defect[top:bottom, left:right].sum())
    box_fraction = (bounds.right - bounds.left) * (bounds.bottom - bounds.top) / (height * width)
    correct = retained >= MIN_RETAINED * total and box_fraction <= MAX_BOX_FRACTION
    return {
        "failed": False,
        "retained": retained,
        "defect": total,
        "box_fraction": box_fraction,
        "correct": bool(correct),
    }


def _pool(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {"images": 0}
    boxes = [entry["box_fraction"] for entry in entries if entry["box_fraction"] is not None]
    defect = sum(entry["defect"] for entry in entries)
    return {
        "images": len(entries),
        "correct_rate": sum(entry["correct"] for entry in entries) / len(entries),
        "pooled_retention": sum(entry["retained"] for entry in entries) / defect
        if defect
        else None,
        "failure_rate": sum(entry["failed"] for entry in entries) / len(entries),
        "mean_box_fraction": float(np.mean(boxes)) if boxes else None,
    }


def _design(generator: Any, datasets: Path, out: Path) -> None:
    """The evidence the rule was frozen from: training normals of the design classes only."""
    rows = []
    for name, image_path, _mask in _visa(datasets, DESIGN_CLASSES, "train", "normal"):
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        records = generator.generate(rgb)
        height, width = rgb.shape[:2]
        baseline = select_region(records, height, width, MobileSamRegionConfig())
        chosen_border = max(
            border_fraction(record["segmentation"])
            for record in records
            if int(record["area"]) == baseline.metadata["area_pixels"]
        )
        rows.append(
            {
                "class": name,
                "baseline_border": chosen_border,
                "baseline_box_fraction": (baseline.bounds.right - baseline.bounds.left)
                * (baseline.bounds.bottom - baseline.bounds.top)
                / (height * width),
                "borders": sorted(border_fraction(record["segmentation"]) for record in records),
            }
        )
        print(name, image_path.name, f"baseline border {chosen_border:.2f}", flush=True)
    every = [value for row in rows for value in row["borders"]]
    summary = {
        "images": len(rows),
        "baseline_min_border": min(row["baseline_border"] for row in rows),
        "baseline_min_box_fraction": min(row["baseline_box_fraction"] for row in rows),
        "largest_border_below_half": max((value for value in every if value < 0.5), default=None),
        "smallest_border_above_half": min((value for value in every if value >= 0.5), default=None),
    }
    out.write_text(json.dumps({"summary": summary, "images": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, default=REPOSITORY / "data" / "model-cache")
    parser.add_argument("--datasets", type=Path, default=REPOSITORY / "datasets")
    parser.add_argument(
        "--design", action="store_true", help="reproduce the design evidence instead"
    )
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    extractor = MobileSamRegionExtractor(
        MobileSamRegionConfig(), assets={ASSET_KEY: _checkpoint(args.model_cache.resolve())}
    )
    # MobileSAM's prompt grid is float64, which MPS rejects; the extractor falls back to CPU on
    # the first image in the application, and this gate starts there.
    generator = extractor._load_generator(force_cpu=True)
    started = time.perf_counter()
    if args.design:
        _design(generator, args.datasets, args.out)
        return 0

    groups = {
        "visa": VISA_VALIDATION,
        "mvtec_objects": MVTEC_OBJECTS,
        "mvtec_textures": MVTEC_TEXTURES,
    }
    sources = [
        ("visa", _visa(args.datasets, VISA_VALIDATION, "test", "anomaly")),
        ("mvtec_objects", _mvtec(args.datasets, MVTEC_OBJECTS)),
        ("mvtec_textures", _mvtec(args.datasets, MVTEC_TEXTURES)),
    ]
    entries: list[dict[str, Any]] = []
    for group, items in sources:
        for name, image_path, mask_path in items:
            assert mask_path is not None
            rgb = np.asarray(Image.open(image_path).convert("RGB"))
            defect = np.asarray(Image.open(mask_path).convert("L")) > 0
            if defect.shape != rgb.shape[:2]:
                raise SystemExit(f"mask and image disagree in size: {image_path}")
            tick = time.perf_counter()
            records = generator.generate(rgb)
            seconds = time.perf_counter() - tick
            row: dict[str, Any] = {
                "group": group,
                "class": name,
                "image": image_path.name,
                "seconds": seconds,
            }
            for rule, overrides in RULES.items():
                row[rule] = _score(records, defect, MobileSamRegionConfig(**overrides))
            entries.append(row)
            print(
                group,
                name,
                image_path.name,
                f"{seconds:.2f}s",
                " ".join(f"{rule}={'ok' if row[rule]['correct'] else 'no'}" for rule in RULES),
                flush=True,
            )

    per_class = {
        f"{group}/{name}": {
            rule: _pool(
                [row[rule] for row in entries if row["group"] == group and row["class"] == name]
            )
            for rule in RULES
        }
        for group, names in groups.items()
        for name in names
    }
    decided = [row for row in entries if row["group"] != "mvtec_textures"]
    pooled = {
        "objects": {rule: _pool([row[rule] for row in decided]) for rule in RULES},
        "textures": {
            rule: _pool([row[rule] for row in entries if row["group"] == "mvtec_textures"])
            for rule in RULES
        },
    }
    base, cand = pooled["objects"]["baseline"], pooled["objects"]["candidate"]
    checks = {
        "correct_rate_gain": cand["correct_rate"] - base["correct_rate"]
        >= CRITERION["minimum_correct_rate_gain"],
        "pooled_retention": (cand["pooled_retention"] or 0.0)
        >= CRITERION["minimum_pooled_retention"],
        "failure_rate": cand["failure_rate"] <= CRITERION["maximum_failure_rate"],
    }
    result = {
        "criterion": CRITERION,
        "correct_definition": {"min_retained": MIN_RETAINED, "max_box_fraction": MAX_BOX_FRACTION},
        "rules": RULES,
        "pooled": pooled,
        "per_class": per_class,
        "checks": checks,
        "promote": all(checks.values()),
        "seconds": time.perf_counter() - started,
        "mean_generate_seconds": float(np.mean([row["seconds"] for row in entries])),
        "device": "cpu",
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "mobile-sam": version("mobile-sam"),
        },
        "images": entries,
    }
    args.out.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {key: result[key] for key in ("pooled", "checks", "promote", "seconds")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
