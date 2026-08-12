"""Comparing several runs under one protocol (ADR-0028).

Torch-free throughout: both columns are `pixel_reference` runs differing only in
configuration, which is enough to exercise every rule here — the point of the module is
that no score crosses from one run to another, and that is as true of two configurations of
one method as of two methods.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.domain.entities import Aggregation, JobKind, JobStatus, Label, Subset
from anomaly_lab.eval.compare import OperatingPoint, agreement, resolve_threshold
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.regions.preparation import run_region_prepare_job
from tests.conftest import Fixture, create_experiment, run_handler, seed_synthetic_split


def sample(sample_id: int, score: float, label: Label) -> ScoredSample:
    return ScoredSample(
        sample_id=sample_id,
        group_key="all",
        external_id=str(sample_id),
        label=label,
        notes=None,
        agg_score=score,
        aggregation=Aggregation.MAX,
        subset=Subset.TEST,
    )


# ------------------------------------------------------------------ the operating point


def test_f1_is_the_same_rule_the_results_screen_opens_at() -> None:
    """Delegated, not reimplemented — a comparison that disagrees with the screen it was
    reached from is worse than a slow one."""
    samples = [
        sample(1, 0.1, Label.NORMAL),
        sample(2, 0.2, Label.NORMAL),
        sample(3, 0.9, Label.DEFECT),
    ]
    resolved = resolve_threshold(samples, OperatingPoint.F1)

    assert resolved.value == pytest.approx(0.9)
    assert "F1" in resolved.rationale


def test_recall_takes_the_highest_cut_that_still_reaches_the_target() -> None:
    """Recall only falls as the threshold rises, so every lower cut reaches the target too;
    the highest one is the only choice that does not buy false alarms it did not need."""
    samples = [
        sample(1, 0.9, Label.DEFECT),
        sample(2, 0.6, Label.DEFECT),
        sample(3, 0.3, Label.DEFECT),
        sample(4, 0.4, Label.NORMAL),
    ]

    every = resolve_threshold(samples, OperatingPoint.RECALL, recall_target=1.0)
    assert every.value == pytest.approx(0.3)

    two_thirds = resolve_threshold(samples, OperatingPoint.RECALL, recall_target=0.66)
    assert two_thirds.value == pytest.approx(0.6)
    assert "2 of 3" in two_thirds.rationale


def test_a_target_no_threshold_reaches_has_no_operating_point() -> None:
    """The nearest achievable point would be a different operating point wearing this
    one's label, so there is none."""
    resolved = resolve_threshold(
        [sample(1, 0.1, Label.NORMAL)], OperatingPoint.RECALL, recall_target=0.9
    )
    assert resolved.value is None
    assert "no defects" in resolved.rationale


# ------------------------------------------------------------------------- the agreement


def test_agreement_is_read_from_the_predictions_not_the_outcomes() -> None:
    """An unlabeled sample is tagged `unlabeled` by every run whatever it predicted, so
    agreement read off the outcome would call every unlabeled row unanimous — and those are
    exactly the rows where a disagreement is worth a human's attention."""
    left = [sample(1, 0.9, Label.UNLABELED)]
    right = [sample(1, 0.1, Label.UNLABELED)]

    rows = agreement([left, right], [0.5, 0.5])

    assert len(rows) == 1
    assert rows[0].outcomes == ["unlabeled", "unlabeled"]
    assert rows[0].predicted == [True, False]
    assert rows[0].agree is False


def test_a_sample_one_run_never_scored_gets_no_verdict_rather_than_a_guess() -> None:
    rows = agreement([[sample(1, 0.9, Label.DEFECT)], []], [0.5, 0.5])

    assert rows[0].scores == [pytest.approx(0.9), None]
    assert rows[0].outcomes == ["tp", None]
    # One judgement is not a disagreement.
    assert rows[0].agree is True


def test_a_run_with_no_operating_point_judges_nothing() -> None:
    rows = agreement([[sample(1, 0.9, Label.DEFECT)]] * 2, [0.5, None])

    assert rows[0].predicted == [True, None]
    assert rows[0].outcomes == ["tp", None]


# ------------------------------------------------------------------------------ the route


def compare(client: TestClient, ids: list[int], **query: Any) -> dict[str, Any]:
    response = client.get("/api/compare", params={"ids": ids, **query})
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


@pytest.fixture
def two_runs(client: TestClient, settings: Settings, seeded: Fixture) -> list[int]:
    """Two scored runs on one split, differing only in a configuration value."""
    ids: list[int] = []
    for index, sigma in enumerate((1.0, 3.0)):
        experiment = create_experiment(
            client, seeded, name=f"run-{index}", config={"smoothing_sigma": sigma}
        )
        run_handler(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
        run_handler(
            settings, JobKind.INFER, {"experiment_id": experiment["id"], "subsets": ["test"]}
        )
        ids.append(int(experiment["id"]))
    return ids


def test_two_runs_compare_on_the_subset_they_both_scored(
    client: TestClient, two_runs: list[int]
) -> None:
    payload = compare(client, two_runs)

    assert payload["subset"] == "test"
    assert [run["id"] for run in payload["runs"]] == two_runs
    assert all(run["scored"] for run in payload["runs"])
    # The stored metric set, verbatim — nothing here recomputes a metric (ADR-0011).
    for run in payload["runs"]:
        assert "sample_roc_auc" in run["metrics"]


def test_each_run_carries_its_own_display_range(client: TestClient, two_runs: list[int]) -> None:
    """Never reconciled with another run's. The A/B map view draws each map on its own
    range and prints both, because what transfers across runs is a fraction of the range
    and not a value (ADR-0028)."""
    payload = compare(client, two_runs)

    ranges = [run["map_range"] for run in payload["runs"]]
    assert all(found is not None for found in ranges)
    assert all(found["high"] > found["low"] for found in ranges)


def test_every_run_carries_its_own_threshold_and_the_sentence_behind_it(
    client: TestClient, two_runs: list[int]
) -> None:
    """The whole point: N thresholds in N units, each printed with its rationale. A
    confusion matrix with no threshold beside it is a claim about an operating point the
    reader cannot name."""
    payload = compare(client, two_runs)

    for run in payload["runs"]:
        assert run["threshold"] is not None
        assert run["threshold_rationale"] != ""
        assert run["confusion"]["true_positive"] + run["confusion"]["false_negative"] == 3


def test_the_columns_agree_with_the_single_run_screen(
    client: TestClient, two_runs: list[int]
) -> None:
    """`f1` is `suggest_threshold`, so the comparison must open where the results screen
    does. A number that drifts from the screen it was reached from is the bug this shares
    an implementation to avoid."""
    payload = compare(client, two_runs)

    for run in payload["runs"]:
        results = client.get(
            f"/api/experiments/{run['id']}/results", params={"subset": "test"}
        ).json()
        assert run["threshold"] == pytest.approx(results["suggested_threshold"])
        assert run["threshold_rationale"] == results["threshold_rationale"]


def test_the_recall_rule_moves_every_threshold(client: TestClient, two_runs: list[int]) -> None:
    at_f1 = compare(client, two_runs)
    at_recall = compare(client, two_runs, at="recall", recall_target=1.0)

    assert at_recall["operating_point"] == "recall"
    for run in at_recall["runs"]:
        assert run["confusion"]["false_negative"] == 0
    # Perfect recall is a lower cut than the F1 optimum on this fixture, in both runs'
    # own units — which is the shape of the trade the rule exists to show.
    for lax, tight in zip(at_recall["runs"], at_f1["runs"], strict=True):
        assert lax["threshold"] <= tight["threshold"]


def test_every_sample_appears_once_with_a_verdict_per_run(
    client: TestClient, two_runs: list[int]
) -> None:
    payload = compare(client, two_runs)

    assert len(payload["samples"]) == 6
    for row in payload["samples"]:
        assert len(row["outcomes"]) == len(two_runs)
        assert len(row["scores"]) == len(two_runs)


# ------------------------------------------------------------------------- what it refuses


def test_a_comparison_of_one_run_is_refused(client: TestClient, two_runs: list[int]) -> None:
    response = client.get("/api/compare", params={"ids": two_runs[:1]})
    assert response.status_code == 422
    assert "at least two" in response.json()["detail"]


def test_the_same_run_twice_is_refused(client: TestClient, two_runs: list[int]) -> None:
    response = client.get("/api/compare", params={"ids": [two_runs[0], two_runs[0]]})
    assert response.status_code == 422


def test_an_unknown_run_is_a_404(client: TestClient, two_runs: list[int]) -> None:
    response = client.get("/api/compare", params={"ids": [two_runs[0], 9999]})
    assert response.status_code == 404


def test_runs_on_different_datasets_are_refused(
    client: TestClient, settings: Settings, seeded: Fixture, tmp_path: Path, two_runs: list[int]
) -> None:
    """A different split is a different question, and putting the numbers in adjacent
    columns is the error the column layout invites — so it is a refusal, not a caveat."""
    with connection(settings.db_path) as conn:
        other = seed_synthetic_split(conn, tmp_path / "other", name="synthetic-2")
    elsewhere = create_experiment(client, other, name="elsewhere")

    response = client.get("/api/compare", params={"ids": [two_runs[0], elsewhere["id"]]})

    assert response.status_code == 422
    assert "different dataset" in response.json()["detail"]


def test_runs_on_different_splits_of_one_dataset_are_refused(
    client: TestClient, settings: Settings, seeded: Fixture, two_runs: list[int]
) -> None:
    with connection(settings.db_path) as conn:
        second = _duplicate_split(conn, seeded)
    other = create_experiment(client, seeded, name="other-split", split_id=second)

    response = client.get("/api/compare", params={"ids": [two_runs[0], other["id"]]})

    assert response.status_code == 422
    assert "different split" in response.json()["detail"]


def _duplicate_split(conn: sqlite3.Connection, seeded: Fixture) -> int:
    """The same assignments under a second split id — identical data, different question.

    Identical on purpose: the refusal is about the *split*, not about the samples landing
    in it, and a second split that also happened to differ would prove the weaker thing.
    """
    assignments = {
        sample_id: subset
        for subset in Subset
        for sample_id in splits_repo.list_sample_ids(conn, seeded.split_id, subset)
    }
    return splits_repo.create_split(
        conn,
        seeded.dataset_id,
        name="second",
        strategy="imported",
        seed=1,
        params={"strategy": "imported"},
        assignments=assignments,
    ).id


# ---------------------------------------------------------------------- what it warns about


def test_different_spatial_input_is_a_warning_and_not_a_refusal(
    client: TestClient, settings: Settings, seeded: Fixture, two_runs: list[int]
) -> None:
    """A paired region experiment is legitimate, but its metrics need a visible caveat."""
    with connection(settings.db_path) as conn:
        coarse_profile = profiles_repo.create_revision(
            conn,
            dataset_id=seeded.dataset_id,
            name="coarse full frame",
            extractor_type="identity",
            extractor_config={},
            prepared_width=8,
            prepared_height=8,
            padding_fraction=0.0,
            seed=17,
        )
    run_region_prepare_job(
        JobContext(
            job_id=98,
            kind=JobKind.REGION_PREPARE,
            params={
                "dataset_id": seeded.dataset_id,
                "profile_id": coarse_profile.id,
                "mode": "build",
            },
            settings=settings,
        )
    )
    coarse = create_experiment(client, seeded, name="coarse", region_profile_id=coarse_profile.id)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": coarse["id"]})
    run_handler(settings, JobKind.INFER, {"experiment_id": coarse["id"], "subsets": ["test"]})

    payload = compare(client, [two_runs[0], int(coarse["id"])])

    assert any("different prepared-region" in note for note in payload["warnings"])
    assert all(run["scored"] for run in payload["runs"])


def test_a_run_with_nothing_scored_gets_an_empty_column_and_a_reason(
    client: TestClient, seeded: Fixture, two_runs: list[int]
) -> None:
    idle = create_experiment(client, seeded, name="never-run")

    payload = compare(client, [two_runs[0], int(idle["id"])])

    assert payload["runs"][1]["scored"] is False
    assert payload["runs"][1]["confusion"] is None
    assert any("no results" in note for note in payload["warnings"])


def test_a_run_trained_after_it_was_scored_says_so(
    client: TestClient, settings: Settings, two_runs: list[int]
) -> None:
    """The exact state a checkpoint reaches when it is continued and not re-scored: the
    numbers on screen describe an older model, and only the timestamps know."""
    with connection(settings.db_path) as conn:
        job = jobs_repo.create_job(conn, kind=JobKind.TRAIN, experiment_id=two_runs[0])
        jobs_repo.finish_job(conn, job.id, status=JobStatus.SUCCEEDED)

    payload = compare(client, two_runs)

    assert any("older checkpoint" in note for note in payload["warnings"])


def test_changed_ground_truth_marks_every_affected_comparison_column(
    client: TestClient,
    settings: Settings,
    seeded: Fixture,
    two_runs: list[int],
) -> None:
    with connection(settings.db_path) as conn:
        image = conn.execute(
            "SELECT sample_id FROM image WHERE id = ?", (seeded.defect_image_ids[0],)
        ).fetchone()
        assert image is not None
        samples_repo.set_label(conn, int(image["sample_id"]), Label.NORMAL)

    payload = compare(client, two_runs)

    assert all(run["ground_truth_stale"] for run in payload["runs"])
    assert any("Ground truth changed" in note for note in payload["warnings"])


def test_an_evaluation_difference_is_named(
    client: TestClient, settings: Settings, seeded: Fixture, two_runs: list[int]
) -> None:
    """The stored scores are the same; how they were read into these numbers is not."""
    averaged = create_experiment(
        client, seeded, name="mean-aggregated", evaluation={"aggregation": "mean"}
    )
    run_handler(settings, JobKind.TRAIN, {"experiment_id": averaged["id"]})
    run_handler(settings, JobKind.INFER, {"experiment_id": averaged["id"], "subsets": ["test"]})

    payload = compare(client, [two_runs[0], int(averaged["id"])])

    assert any("aggregation" in note for note in payload["warnings"])
