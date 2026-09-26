"""Seed a scratch lab with one public VisA class and two scored runs.

Talks only to the HTTP API of a backend started on a *scratch* data dir, so the user's
own catalogue is never written. Idempotent: an existing dataset, split or run
of the same name is reused. Prints one JSON object of the ids the screenshot pass needs.

    python seed.py --api http://127.0.0.1:8010 --visa-root /abs/path/datasets/VisA_20220922 \
        [--category candle] > ids.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8010")
    parser.add_argument("--visa-root", required=True, type=Path)
    parser.add_argument("--category", default="candle")
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Also draw 5 `defect` references and score a `color_prototype` run (ADR-0040).",
    )
    args = parser.parse_args()
    api: str = args.api.rstrip("/")

    def req(method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            api + path, method=method, data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            sys.exit(f"{method} {path} -> {error.code}: {error.read()[:400]!r}")

    def wait(job: dict[str, Any]) -> dict[str, Any]:
        while True:
            current = req("GET", f"/api/jobs/{job['id']}")
            if current["status"] in ("succeeded", "failed", "cancelled"):
                if current["status"] != "succeeded":
                    sys.exit(f"{current['kind']} job {current['status']}: {current.get('error')}")
                log(f"{current['kind']} succeeded")
                return current
            time.sleep(2)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    visa = args.visa_root.resolve()
    category = args.category

    dataset = next((d for d in req("GET", "/api/datasets") if d["name"] == category), None)
    if dataset is None:
        scan = wait(
            req(
                "POST",
                "/api/import/scan",
                {
                    "root_path": str(visa),
                    "dataset_root": str(visa / category),
                    "dataset_name": category,
                    "adapter": "csv_table",
                    "options": {
                        "csv_path": "split_csv/1cls.csv",
                        "filter_column": "object",
                        "filter_value": category,
                    },
                },
            )
        )
        manifest = req("GET", f"/api/import/manifests/{scan['result']['manifest_id']}")
        dataset_id = req("POST", "/api/import/commit", {"manifest": manifest})["dataset_id"]
    else:
        dataset_id = dataset["id"]

    splits = req("GET", f"/api/splits?dataset_id={dataset_id}")
    split = (
        splits[0]
        if splits
        else req("POST", "/api/splits", {"dataset_id": dataset_id, "name": "generated", "seed": 7})
    )

    # Every dataset has an implicit "Full frame" profile, and a run's train job prepares it at
    # the run's size, so nothing is built here.
    experiments = req("GET", f"/api/experiments?dataset_id={dataset_id}")
    experiments = (
        experiments.get("items", experiments) if isinstance(experiments, dict) else experiments
    )
    run_ids = []
    for name in ("pixel reference A", "pixel reference B"):
        run = next((e for e in experiments if e["name"] == name), None)
        if run is None:
            run = req(
                "POST",
                "/api/experiments",
                {
                    "name": name,
                    "dataset_id": dataset_id,
                    "split_id": split["id"],
                    "model_type": "pixel_reference",
                },
            )
        if run["status"] != "trained":
            wait(req("POST", f"/api/experiments/{run['id']}/train"))
        detail = req("GET", f"/api/experiments/{run['id']}")
        if not detail.get("scored_subsets"):
            wait(req("POST", f"/api/experiments/{run['id']}/infer"))
        run_ids.append(run["id"])

    few_shot_run = None
    if args.few_shot:
        splits = req("GET", f"/api/splits?dataset_id={dataset_id}")
        references = next((s for s in splits if s["name"] == "five shots"), None) or req(
            "POST",
            "/api/splits",
            {
                "dataset_id": dataset_id,
                "name": "five shots",
                "seed": 0,
                "params": {"strategy": "few_shot", "label_key": "defect", "shots": 5},
            },
        )
        run = next((e for e in experiments if e["name"] == "colour floor"), None)
        if run is None:
            run = req(
                "POST",
                "/api/experiments",
                {
                    "name": "colour floor",
                    "dataset_id": dataset_id,
                    "split_id": references["id"],
                    "model_type": "color_prototype",
                    "task": "few_shot_segmentation",
                    "target_label": "defect",
                },
            )
        if run["status"] != "trained":
            wait(req("POST", f"/api/experiments/{run['id']}/train"))
        if not req("GET", f"/api/experiments/{run['id']}").get("scored_subsets"):
            wait(req("POST", f"/api/experiments/{run['id']}/infer"))
        few_shot_run = run["id"]

    results = req("GET", f"/api/experiments/{run_ids[0]}/results?subset=test")
    defect = next(s for s in results["samples"] if s["label"] == "defect")
    sample = req("GET", f"/api/datasets/{dataset_id}/samples/{defect['sample_id']}")
    print(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "experiment_ids": run_ids,
                "defect_sample_id": defect["sample_id"],
                "defect_image_id": sample["images"][0]["id"],
                **({"few_shot_run": few_shot_run} if few_shot_run is not None else {}),
            }
        )
    )


if __name__ == "__main__":
    main()
