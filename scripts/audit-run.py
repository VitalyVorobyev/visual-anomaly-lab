#!/usr/bin/env python3
"""Audit a scored run: export every per-image score, and compare map aggregations.

Two questions this answers that the results screen does not:

  * **Show me the raw evidence.** One CSV row per image — label, subset, stored score,
    inference time, and the paths to the anomaly map and the ground-truth mask. That is
    what makes a reported ROC-AUC checkable by somebody who does not trust it.
  * **Does the aggregation matter?** The stored anomaly map is reduced to one number per
    image, and `max` is the published choice. This recomputes the image-level ROC-AUC under
    `max`, top-k means, high quantiles and the plain mean, from the maps already on disk.

**Read-only, numpy only, no torch, no GPU.** It reads what a run already wrote, so it costs
nothing and can be pointed at a run from months ago.

    uv run --directory backend python ../scripts/audit-run.py 5
    uv run --directory backend python ../scripts/audit-run.py 5 --subset val --csv out.csv

Measured twice on `candle`, and the two answers disagree in a way worth knowing:

    aggregation      anomalib teacher    nelson1425 teacher
    max              0.7509              0.8857
    top64_mean       0.7584              0.8882
    p99              0.7652              0.7981
    p95              0.7695              0.7101
    mean             0.7559              0.7613

Under the weaker teacher **every reducer scored the same** — the plain mean did as well as
the max — and the honest reading was that the aggregation did not matter. It was not: nothing
was localized, so every summary of the map was a summary of the same noise. Under the better
teacher the max is worth 0.12 over the mean and 0.17 over p95, because there is now a peak to
find. A negative result measured on a broken configuration is a measurement of the breakage.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPOSITORY / "data" / "app.sqlite3"


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Rank-based ROC-AUC with ties averaged, or None when one class is absent.

    None rather than 0.5 or 0.0: a subset with no defects has no ROC-AUC, and the rest of
    this application renders that as a dash rather than inventing a number.
    """
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ordered = scores[order]
    index = 0
    while index < len(scores):
        stop = index
        while stop + 1 < len(scores) and ordered[stop + 1] == ordered[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aggregations(array: np.ndarray) -> dict[str, float]:
    """Every way of turning one map into one number that is worth comparing."""
    flat = np.sort(array.ravel())[::-1]
    out: dict[str, float] = {"max": float(flat[0])}
    for k in (16, 64, 256, 1024):
        out[f"top{k}_mean"] = float(flat[: min(k, flat.size)].mean())
    for q in (99.9, 99.0, 95.0):
        out[f"p{q:g}"] = float(np.percentile(array, q))
    out["mean"] = float(array.mean())
    return out


def rows_for(conn: sqlite3.Connection, experiment_id: int, subset: str | None) -> list[dict]:
    clause = "AND sa.subset = ?" if subset else ""
    parameters: tuple = (experiment_id, experiment_id)
    if subset:
        parameters = (experiment_id, experiment_id, subset)
    return [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT ir.image_id, im.path AS image_path, sm.label, sa.subset,
                   ir.score, ir.inference_ms, ir.map_path,
                   (SELECT mk.path FROM mask mk WHERE mk.image_id = im.id LIMIT 1) AS mask_path
              FROM image_result ir
              JOIN image im ON im.id = ir.image_id
              JOIN sample sm ON sm.id = im.sample_id
              JOIN experiment e ON e.id = ir.experiment_id
              JOIN split_assignment sa
                   ON sa.sample_id = sm.id AND sa.split_id = e.split_id
             WHERE ir.experiment_id = ? AND e.id = ? {clause}
             ORDER BY ir.score DESC
            """,
            parameters,
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment_id", type=int)
    parser.add_argument("--subset", default="test", help="Subset to audit; 'all' for every one.")
    parser.add_argument("--csv", type=Path, help="Write the per-image table here.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    header = conn.execute(
        "SELECT name, model_type, model_config FROM experiment WHERE id = ?",
        (args.experiment_id,),
    ).fetchone()
    if header is None:
        print(f"no experiment {args.experiment_id} in {args.db}", file=sys.stderr)
        return 2

    subset = None if args.subset == "all" else args.subset
    rows = rows_for(conn, args.experiment_id, subset)
    conn.close()
    if not rows:
        print(f"experiment {args.experiment_id} has no scored images in {args.subset}")
        return 1

    print(f"experiment {args.experiment_id}: {header['name']} ({header['model_type']})")
    print(f"  config {header['model_config']}")
    labelled = [r for r in rows if r["label"] in {"normal", "defect"}]
    labels = np.array([1 if r["label"] == "defect" else 0 for r in labelled])
    print(
        f"  {args.subset}: {len(rows)} scored, {len(labelled)} labelled "
        f"({int(labels.sum())} defect / {int((1 - labels).sum())} normal)"
    )

    computed: dict[str, list[float]] = {}
    missing = 0
    for row in labelled:
        if not row["map_path"] or not Path(row["map_path"]).is_file():
            missing += 1
            continue
        array = np.squeeze(np.load(row["map_path"]))
        for name, value in aggregations(array).items():
            computed.setdefault(name, []).append(value)
        row["_computed"] = True

    stored = np.array([r["score"] for r in labelled])
    print("\n  image ROC-AUC by aggregation:")
    baseline = roc_auc(labels, stored)
    print(f"    {'stored score':<14} {baseline if baseline is None else f'{baseline:.4f}'}")
    if missing:
        # A silent omission would read as "these are all the images there were".
        print(f"    ({missing} of {len(labelled)} maps are missing and were left out)")
    kept = labels[[bool(r.get("_computed")) for r in labelled]]
    for name, values in computed.items():
        score = roc_auc(kept, np.array(values))
        print(f"    {name:<14} {score if score is None else f'{score:.4f}'}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "image_id", "image_path", "label", "subset", "score", "inference_ms",
            "map_path", "mask_path",
        ]
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  wrote {len(rows)} rows to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
