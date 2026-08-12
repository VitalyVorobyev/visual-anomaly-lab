"""Results repository — scores, aggregated scores, and metric sets.

Every read here joins the score back to the sample's label and subset, because a score
without them cannot be evaluated or displayed, and doing the join in SQL keeps the
evaluation layer from having to hold three parallel dictionaries in step.

Writes replace rather than merge. An inference run is the unit of truth: a run that
scored 200 images and a previous run that scored 400 must not leave 200 stale rows behind
pretending to be part of the same result.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anomaly_lab.domain.entities import (
    Aggregation,
    ImageResult,
    Label,
    MetricSet,
    SampleResult,
    Subset,
)


@dataclass(frozen=True)
class ScoredImage:
    """One image's score with everything needed to evaluate or draw it."""

    image_id: int
    sample_id: int
    channel: str | None
    path: str
    width: int
    height: int
    score: float
    map_path: str | None
    inference_ms: float
    label: Label
    subset: Subset | None


@dataclass(frozen=True)
class ScoredSample:
    """One part's aggregated score, with its identity and its label."""

    sample_id: int
    group_key: str
    external_id: str
    label: Label
    notes: str | None
    agg_score: float
    aggregation: Aggregation
    subset: Subset | None


def replace_image_results(
    conn: sqlite3.Connection,
    experiment_id: int,
    rows: Sequence[ImageResult],
) -> int:
    """Replace every image result for this experiment, atomically."""
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM image_result WHERE experiment_id = ?", (experiment_id,))
        conn.executemany(
            """
            INSERT INTO image_result (experiment_id, image_id, score, map_path, inference_ms)
                 VALUES (?, ?, ?, ?, ?)
            """,
            [
                (experiment_id, row.image_id, row.score, row.map_path, row.inference_ms)
                for row in rows
            ],
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return len(rows)


def replace_sample_results(
    conn: sqlite3.Connection,
    experiment_id: int,
    rows: Sequence[SampleResult],
) -> int:
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM sample_result WHERE experiment_id = ?", (experiment_id,))
        conn.executemany(
            """
            INSERT INTO sample_result (experiment_id, sample_id, agg_score, aggregation)
                 VALUES (?, ?, ?, ?)
            """,
            [(experiment_id, row.sample_id, row.agg_score, row.aggregation.value) for row in rows],
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return len(rows)


def replace_metric_sets(
    conn: sqlite3.Connection,
    experiment_id: int,
    metrics_by_subset: Mapping[Subset, Mapping[str, Any]],
    *,
    ground_truth_digests: Mapping[Subset, str] | None = None,
) -> int:
    """Replace this experiment's metric sets. Subsets with no metrics are simply absent."""
    conn.execute("BEGIN")
    try:
        conn.execute("DELETE FROM metric_set WHERE experiment_id = ?", (experiment_id,))
        conn.executemany(
            """
            INSERT INTO metric_set (experiment_id, subset, metrics, ground_truth_digest)
                 VALUES (?, ?, ?, ?)
            """,
            [
                (
                    experiment_id,
                    subset.value,
                    json.dumps(dict(metrics), sort_keys=True),
                    ground_truth_digests.get(subset) if ground_truth_digests else None,
                )
                for subset, metrics in metrics_by_subset.items()
            ],
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return len(metrics_by_subset)


def list_metric_sets(conn: sqlite3.Connection, experiment_id: int) -> list[MetricSet]:
    rows = conn.execute(
        "SELECT * FROM metric_set WHERE experiment_id = ? ORDER BY subset",
        (experiment_id,),
    ).fetchall()
    return [MetricSet.model_validate(dict(row)) for row in rows]


def _subset_of(value: object) -> Subset | None:
    return Subset(value) if isinstance(value, str) and value else None


def list_scored_images(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    subset: Subset | None = None,
) -> list[ScoredImage]:
    """Every scored image, joined to its sample's label and its subset in this split.

    The split is taken from the experiment rather than from a parameter, so a subset
    filter can never accidentally read assignments from a different split.
    """
    clauses = ["image_result.experiment_id = ?"]
    params: list[object] = [experiment_id]
    if subset is not None:
        clauses.append("split_assignment.subset = ?")
        params.append(subset.value)

    rows = conn.execute(
        f"""
        SELECT image_result.image_id      AS image_id,
               image.sample_id            AS sample_id,
               channel.name               AS channel,
               image.path                 AS path,
               image.width                AS width,
               image.height               AS height,
               image_result.score         AS score,
               image_result.map_path      AS map_path,
               image_result.inference_ms  AS inference_ms,
               sample.label               AS label,
               split_assignment.subset    AS subset
          FROM image_result
          JOIN image      ON image.id = image_result.image_id
          JOIN sample     ON sample.id = image.sample_id
          JOIN experiment ON experiment.id = image_result.experiment_id
          LEFT JOIN channel ON channel.id = image.channel_id
          LEFT JOIN split_assignment
                 ON split_assignment.sample_id = sample.id
                AND split_assignment.split_id = experiment.split_id
         WHERE {" AND ".join(clauses)}
         ORDER BY image_result.image_id
        """,
        params,
    ).fetchall()

    return [
        ScoredImage(
            image_id=int(row["image_id"]),
            sample_id=int(row["sample_id"]),
            channel=row["channel"],
            path=str(row["path"]),
            width=int(row["width"]),
            height=int(row["height"]),
            score=float(row["score"]),
            map_path=row["map_path"],
            inference_ms=float(row["inference_ms"]),
            label=Label(row["label"]),
            subset=_subset_of(row["subset"]),
        )
        for row in rows
    ]


def list_scored_samples(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    subset: Subset | None = None,
) -> list[ScoredSample]:
    """Aggregated results, ordered most anomalous first — the ranked list, for free."""
    clauses = ["sample_result.experiment_id = ?"]
    params: list[object] = [experiment_id]
    if subset is not None:
        clauses.append("split_assignment.subset = ?")
        params.append(subset.value)

    rows = conn.execute(
        f"""
        SELECT sample_result.sample_id    AS sample_id,
               sample.group_key           AS group_key,
               sample.external_id         AS external_id,
               sample.label               AS label,
               sample.notes               AS notes,
               sample_result.agg_score    AS agg_score,
               sample_result.aggregation  AS aggregation,
               split_assignment.subset    AS subset
          FROM sample_result
          JOIN sample     ON sample.id = sample_result.sample_id
          JOIN experiment ON experiment.id = sample_result.experiment_id
          LEFT JOIN split_assignment
                 ON split_assignment.sample_id = sample.id
                AND split_assignment.split_id = experiment.split_id
         WHERE {" AND ".join(clauses)}
         ORDER BY sample_result.agg_score DESC, sample_result.sample_id
        """,
        params,
    ).fetchall()

    return [
        ScoredSample(
            sample_id=int(row["sample_id"]),
            group_key=str(row["group_key"]),
            external_id=str(row["external_id"]),
            label=Label(row["label"]),
            notes=row["notes"],
            agg_score=float(row["agg_score"]),
            aggregation=Aggregation(row["aggregation"]),
            subset=_subset_of(row["subset"]),
        )
        for row in rows
    ]


def get_image_result(
    conn: sqlite3.Connection,
    experiment_id: int,
    image_id: int,
) -> ImageResult | None:
    row = conn.execute(
        "SELECT * FROM image_result WHERE experiment_id = ? AND image_id = ?",
        (experiment_id, image_id),
    ).fetchone()
    return ImageResult.model_validate(dict(row)) if row is not None else None


def scored_subsets(conn: sqlite3.Connection, experiment_id: int) -> list[Subset]:
    """Which subsets this experiment actually has results for, in canonical order."""
    rows = conn.execute(
        """
        SELECT DISTINCT split_assignment.subset AS subset
          FROM sample_result
          JOIN experiment ON experiment.id = sample_result.experiment_id
          JOIN split_assignment
            ON split_assignment.sample_id = sample_result.sample_id
           AND split_assignment.split_id = experiment.split_id
         WHERE sample_result.experiment_id = ?
        """,
        (experiment_id,),
    ).fetchall()
    found = {Subset(row["subset"]) for row in rows if row["subset"]}
    return [subset for subset in Subset if subset in found]
