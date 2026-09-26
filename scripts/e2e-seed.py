"""Seed a throwaway backend with a small synthetic workbench for the screenshot suite.

    uv run --directory backend python ../scripts/e2e-seed.py <tree-dir> <api-base-url>

Writes twelve 256x256 synthetic "discs" (eight good, four with a dark blemish) under
<tree-dir>, then drives the real API: import them with `folder_classes`, split them, and
train and score a `pixel_reference` experiment. Every
image is generated here (ADR-0022); nothing is read from a dataset.

`frontend/e2e/global-setup.ts` runs it once per data directory and reuses the result, so
a baseline and the run compared against it see the same rows, ids and timestamps.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def disc(seed: int, *, blemish: bool) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:256, 0:256]
    r = np.hypot(x - 128, y - 128)
    img = 150 + 60 * np.cos(r / 9.0) * (r < 110) + rng.normal(0, 4, (256, 256))
    if blemish:
        img -= 90 * np.exp(-((x - 70 - seed % 40) ** 2 + (y - 90) ** 2) / 60.0)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")


def write_tree(root: Path) -> None:
    for index in range(8):
        path = root / "good" / f"g{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        disc(index, blemish=False).save(path)
    for index in range(4):
        path = root / "defect" / f"d{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        disc(100 + index, blemish=True).save(path)


def call(base: str, method: str, path: str, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{base}{path}", data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {path} -> {error.code}: {error.read().decode()}") from None


def wait(base: str, job: dict[str, Any]) -> dict[str, Any]:
    for _ in range(600):
        job = call(base, "GET", f"/api/jobs/{job['id']}")
        if job["status"] == "succeeded":
            return job
        if job["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"job {job['id']} ({job['kind']}) {job['status']}: {job['error']}")
        time.sleep(0.5)
    raise TimeoutError(f"job {job['id']} did not finish")


def main(tree: Path, base: str) -> None:
    write_tree(tree)
    scan = wait(
        base,
        call(base, "POST", "/api/import/scan", {
            "root_path": str(tree),
            "dataset_name": "Synthetic discs",
            "adapter": "folder_classes",
            "options": {"normal_dirs": ["good"], "defect_dirs": ["defect"]},
        }),
    )
    manifest = call(base, "GET", f"/api/import/manifests/{scan['result']['manifest_id']}")
    dataset = call(base, "POST", "/api/import/commit", {"manifest": manifest})["dataset_id"]
    split = call(base, "POST", "/api/splits", {"dataset_id": dataset, "name": "default", "seed": 7})
    # The import gives the dataset its full-frame region profile, and training prepares the
    # input it needs, so nothing is built in advance.
    experiment = call(base, "POST", "/api/experiments", {
        "name": "pixel floor",
        "dataset_id": dataset,
        "split_id": split["id"],
        "model_type": "pixel_reference",
    })
    wait(base, call(base, "POST", f"/api/experiments/{experiment['id']}/train", None))
    wait(base, call(base, "POST", f"/api/experiments/{experiment['id']}/infer", None))
    print(f"seeded dataset {dataset}, experiment {experiment['id']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2].rstrip("/"))
