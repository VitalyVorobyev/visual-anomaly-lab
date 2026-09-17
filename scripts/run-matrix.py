#!/usr/bin/env python3
"""Drive a declared experiment matrix through the running sidecar's public HTTP API.

A comparison campaign is a table of configurations, and hand-entering twelve of them
through the create form is where comparison errors are born: one field differing that
nobody intended, discoverable only by diffing frozen configs after the fact. This script
takes the whole matrix as one JSON file, creates each experiment through the same
validated endpoint the form uses, queues train and infer, waits, and collects every
run's metrics into one report.

Deliberately unlike the public-gate scripts, which build isolated data directories from
`/datasets`: here the datasets, splits, annotations and region profiles already live in
the app database, and going through the API means the real queue, the real resident-worker
eviction, and no second code path.

**The matrix file names datasets and runs; this script names nothing.** Keep matrix files
and reports in the gitignored `results/` directory.

Matrix file shape:

    {
      "runs": [
        {
          "name": "...",                # experiment name, also the report key
          "dataset_id": 21,
          "split_id": 4,
          "region_profile_id": 15,
          "model_type": "dino_memory",
          "config": {...},              # validated against the method's own schema
          "preprocessing": {"color": "grayscale"},
          "evaluation": {...},
          "channels": [],               # empty = every channel
          "notes": "..."
        }
      ],
      "reevaluate": [25, 26]            # existing experiment ids to re-evaluate only
    }

Usage, with the sidecar already running:

    uv run --directory backend python ../scripts/run-matrix.py ../results/matrix.json \
        --report ../results/matrix-report.json

Runs execute strictly in order (the job queue is serial anyway); a failed job stops the
matrix unless --keep-going is set. Exit 0 only when every requested run succeeded.
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

TERMINAL = {"succeeded", "failed", "cancelled"}


def call(base: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    request = urllib.request.Request(
        f"{base}{path}",
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{request.method} {path} -> {exc.code}: {detail}") from exc


def wait_for_job(base: str, job_id: int, label: str, poll_seconds: float) -> dict[str, Any]:
    last_message = ""
    while True:
        job = call(base, f"/api/jobs/{job_id}")
        status = job["status"]
        message = job.get("message") or ""
        if message and message != last_message:
            print(f"    [{label}] {message}", flush=True)
            last_message = message
        if status in TERMINAL:
            return job
        time.sleep(poll_seconds)


def run_one(base: str, spec: dict[str, Any], poll_seconds: float) -> dict[str, Any]:
    body = {
        "name": spec["name"],
        "dataset_id": spec["dataset_id"],
        "split_id": spec["split_id"],
        "region_profile_id": spec["region_profile_id"],
        "model_type": spec["model_type"],
        "config": spec.get("config", {}),
        "preprocessing": spec.get("preprocessing", {}),
        "evaluation": spec.get("evaluation", {}),
        "channels": spec.get("channels", []),
        "notes": spec.get("notes"),
    }
    detail = call(base, "/api/experiments", body)
    experiment_id = detail["id"]
    print(f"  created experiment {experiment_id}: {spec['name']}", flush=True)

    for phase, path in (("train", "train"), ("infer", "infer")):
        started = time.monotonic()
        body = {"experiment_id": experiment_id}
        job = call(base, f"/api/experiments/{experiment_id}/{path}", body)
        job = wait_for_job(base, job["id"], f"{spec['name']}:{phase}", poll_seconds)
        elapsed = time.monotonic() - started
        print(f"    {phase}: {job['status']} in {elapsed:.0f}s", flush=True)
        if job["status"] != "succeeded":
            return {
                "experiment_id": experiment_id,
                "name": spec["name"],
                "status": f"{phase}_{job['status']}",
                "error": job.get("error"),
            }

    detail = call(base, f"/api/experiments/{experiment_id}")
    return {
        "experiment_id": experiment_id,
        "name": spec["name"],
        "status": "succeeded",
        "model_type": detail["model_type"],
        "model_config": detail.get("model_config"),
        "region_profile_id": detail.get("region_profile_id"),
        "metrics": detail.get("metrics", []),
    }


def reevaluate_one(base: str, experiment_id: int) -> dict[str, Any]:
    metrics = call(base, f"/api/experiments/{experiment_id}/reevaluate", {})
    detail = call(base, f"/api/experiments/{experiment_id}")
    return {
        "experiment_id": experiment_id,
        "name": detail.get("name"),
        "status": "reevaluated",
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("matrix", type=Path, help="JSON file declaring the runs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--report", type=Path, help="Write the collected results here.")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with the next run after a failure instead of stopping.",
    )
    args = parser.parse_args(argv)

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    runs = matrix.get("runs", [])
    reevaluations = matrix.get("reevaluate", [])
    print(f"{len(runs)} run(s), {len(reevaluations)} re-evaluation(s) against {args.base_url}")

    results: list[dict[str, Any]] = []
    failed = False
    for experiment_id in reevaluations:
        print(f"re-evaluating experiment {experiment_id}", flush=True)
        try:
            results.append(reevaluate_one(args.base_url, experiment_id))
        except RuntimeError as exc:
            failed = True
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            results.append({"experiment_id": experiment_id, "status": "error", "error": str(exc)})
            if not args.keep_going:
                break

    if not failed or args.keep_going:
        for spec in runs:
            print(f"run: {spec['name']}", flush=True)
            try:
                outcome = run_one(args.base_url, spec, args.poll_seconds)
            except RuntimeError as exc:
                outcome = {"name": spec.get("name"), "status": "error", "error": str(exc)}
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            results.append(outcome)
            if outcome["status"] != "succeeded" and not args.keep_going:
                failed = True
                break
            if outcome["status"] != "succeeded":
                failed = True

    report = {"base_url": args.base_url, "matrix": str(args.matrix), "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.report}")

    bad = [r for r in results if r["status"] not in {"succeeded", "reevaluated"}]
    print(f"done: {len(results) - len(bad)} ok, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
