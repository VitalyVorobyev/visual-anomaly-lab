"""The catalogue schema: one script, one version, and a refusal for anything else."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from anomaly_lab import serve
from anomaly_lab.config import Settings
from anomaly_lab.db import migrate
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import (
    SCHEMA_VERSION,
    SchemaError,
    SchemaVersionError,
    apply_schema,
    apply_schema_to,
    contains_transaction_control,
    current_schema_version,
    schema_sql,
)
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind

# The canonical domain entities of ADR-0005, one table each, and the annotation and
# region-profile tables beside them.
EXPECTED_TABLES = {
    "annotation_draft",
    "annotation_label",
    "annotation_revision",
    "annotation_sample_draft",
    "dataset",
    "channel",
    "sample",
    "image",
    "mask",
    "split",
    "split_assignment",
    "experiment",
    "job",
    "image_result",
    "sample_result",
    "metric_set",
    "region_profile_revision",
}


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _columns(conn: sqlite3.Connection, table: str) -> dict[str, sqlite3.Row]:
    return {str(row["name"]): row for row in conn.execute(f"PRAGMA table_info({table})")}


def test_a_fresh_database_gets_the_schema_and_its_version(settings: Settings) -> None:
    version = apply_schema(settings.db_path)

    assert version == SCHEMA_VERSION
    with connect(settings.db_path) as conn:
        assert current_schema_version(conn) == SCHEMA_VERSION
        assert _table_names(conn) == EXPECTED_TABLES


def test_reopening_a_current_catalogue_is_a_no_op(settings: Settings) -> None:
    apply_schema(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('kept', '/kept')")
        conn.commit()

    assert apply_schema(settings.db_path) == SCHEMA_VERSION
    with connect(settings.db_path) as conn:
        assert current_schema_version(conn) == SCHEMA_VERSION
        assert conn.execute("SELECT name FROM dataset").fetchone()[0] == "kept"


def test_creates_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deeper" / "app.sqlite3"
    apply_schema(db_path)
    assert db_path.exists()


@pytest.mark.parametrize("found", [1, 26, SCHEMA_VERSION - 1])
def test_a_catalogue_from_an_older_schema_is_refused_and_left_alone(
    settings: Settings, found: int
) -> None:
    apply_schema(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute(f"PRAGMA user_version = {found:d}")

    with pytest.raises(SchemaVersionError) as refused:
        apply_schema(settings.db_path)

    message = str(refused.value)
    assert message == (
        "This catalogue was created by an older version of anomaly lab and cannot be opened. "
        f"Delete {settings.db_path} (and the app data directory beside it) to start fresh."
    )
    with connect(settings.db_path) as conn:
        # Nothing was migrated, dropped or re-stamped.
        assert current_schema_version(conn) == found
        assert _table_names(conn) == EXPECTED_TABLES


def test_a_catalogue_from_a_newer_schema_is_refused(settings: Settings) -> None:
    apply_schema(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1:d}")

    with pytest.raises(SchemaVersionError, match="newer version") as refused:
        apply_schema(settings.db_path)
    assert str(settings.db_path) in str(refused.value)


def test_an_unstamped_database_that_holds_tables_is_refused(settings: Settings) -> None:
    """Version 0 with tables is not empty: applying the script over it would half-work."""
    with connect(settings.db_path) as conn:
        conn.execute("CREATE TABLE dataset (id INTEGER PRIMARY KEY)")

    with pytest.raises(SchemaVersionError, match="older version"):
        apply_schema(settings.db_path)


def test_serve_explains_a_refused_catalogue_on_both_streams(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The desktop shell shows the `error` event's message as its startup-failure headline."""
    apply_schema(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute("PRAGMA user_version = 3")
    monkeypatch.setattr(serve, "get_settings", lambda: settings)

    assert serve.main() == 1

    out, err = capsys.readouterr()
    event = json.loads(out.strip())
    assert event["ev"] == "error"
    assert event["message"].startswith("This catalogue was created by an older version")
    assert str(settings.db_path) in event["message"]
    assert err.strip() == event["message"]
    assert "Traceback" not in err


def test_a_failed_script_leaves_nothing_behind(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner wraps the script in a transaction, so a failure is all-or-nothing."""
    monkeypatch.setattr(
        migrate,
        "schema_sql",
        lambda: "CREATE TABLE good (id INTEGER);\nCREATE TABLE bad (this is not sql);",
    )

    with connect(settings.db_path) as conn:
        with pytest.raises(SchemaError):
            apply_schema_to(conn)

        assert current_schema_version(conn) == 0
        assert "good" not in _table_names(conn)


def test_the_script_does_not_manage_its_own_transaction() -> None:
    """An inner COMMIT would end the runner's transaction and leave a partial schema."""
    assert not contains_transaction_control(schema_sql())
    # A trigger body's BEGIN ... END is not transaction control.
    assert contains_transaction_control("CREATE TABLE t (id INTEGER);\nCOMMIT;")
    assert not contains_transaction_control(
        "CREATE TRIGGER t BEFORE UPDATE ON x\nBEGIN\n    SELECT 1;\nEND;"
    )


def test_foreign_keys_and_wal_are_enabled(migrated_db: sqlite3.Connection) -> None:
    assert migrated_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert str(migrated_db.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


def test_foreign_keys_are_enforced(migrated_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            "INSERT INTO channel (dataset_id, name, position) VALUES (?, ?, ?)",
            (999, "bright", 0),
        )
    assert migrated_db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sample_identity_is_unique_per_dataset(migrated_db: sqlite3.Connection) -> None:
    """`external_id` collides across capture groups, so identity is the triple."""
    migrated_db.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
    for group in ("set1", "set2"):
        migrated_db.execute(
            "INSERT INTO sample (dataset_id, group_key, external_id) VALUES (1, ?, '17')",
            (group,),
        )

    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            "INSERT INTO sample (dataset_id, group_key, external_id) VALUES (1, 'set1', '17')"
        )


def test_image_channel_is_optional(migrated_db: sqlite3.Connection) -> None:
    """Single-view datasets need no synthetic channel (ADR-0005)."""
    migrated_db.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
    migrated_db.execute(
        "INSERT INTO sample (dataset_id, group_key, external_id) VALUES (1, 'g', '1')"
    )
    migrated_db.execute(
        """
        INSERT INTO image (sample_id, channel_id, path, width, height, bit_depth, file_size, sha256)
        VALUES (1, NULL, '/tmp/a.png', 4, 4, 8, 100, 'abc')
        """
    )
    assert migrated_db.execute("SELECT channel_id FROM image").fetchone()[0] is None


def test_labels_are_constrained(migrated_db: sqlite3.Connection) -> None:
    migrated_db.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            "INSERT INTO sample (dataset_id, group_key, external_id, label) "
            "VALUES (1, 'g', '1', 'maybe')"
        )


def test_region_profile_revisions_are_dataset_owned_and_immutable(
    migrated_db: sqlite3.Connection,
) -> None:
    migrated_db.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
    migrated_db.execute(
        """
        INSERT INTO region_profile_revision (
            dataset_id, name, revision_no, extractor_type, extractor_config,
            prepared_width, prepared_height
        ) VALUES (1, 'dominant object', 1, 'identity', '{}', 256, 256)
        """
    )
    row = migrated_db.execute("SELECT * FROM region_profile_revision").fetchone()
    assert (row["resample"], row["sample_alignment"]) == ("bilinear", "per_image")

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        migrated_db.execute(
            "UPDATE region_profile_revision SET extractor_type = 'other' WHERE id = 1"
        )
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            """
            INSERT INTO region_profile_revision (
                dataset_id, name, revision_no, extractor_type, extractor_config,
                prepared_width, prepared_height, resample
            ) VALUES (1, 'bad resample', 1, 'identity', '{}', 256, 256, 'cubic-ish')
            """
        )

    migrated_db.execute("DELETE FROM dataset WHERE id = 1")
    assert migrated_db.execute("SELECT COUNT(*) FROM region_profile_revision").fetchone()[0] == 0


def test_a_job_kind_is_validated_by_jobkind_not_by_the_schema(
    migrated_db: sqlite3.Connection,
) -> None:
    """The schema keeps no list of kinds; `JobKind` is the only one.

    The schema accepting any text is the point — a new kind costs no schema change — so the
    refusal of a nonsense kind has to happen on the way in, in the repository.
    """
    migrated_db.execute("INSERT INTO job (kind, params) VALUES ('distill', '{}')")
    migrated_db.execute("INSERT INTO job (kind, params) VALUES ('a_future_kind', '{}')")

    with pytest.raises(ValueError, match="ponder"):
        jobs_repo.create_job(migrated_db, kind="ponder")  # type: ignore[arg-type]
    created = jobs_repo.create_job(migrated_db, kind=JobKind.DISTILL)
    assert created.kind is JobKind.DISTILL


def test_results_carry_open_vocabularies_and_closed_shapes(
    migrated_db: sqlite3.Connection,
) -> None:
    """Vocabularies are validated in Python; lifecycles, ranges and flags are CHECKed here."""
    conn = migrated_db
    conn.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/d')")
    conn.execute("INSERT INTO sample (dataset_id, group_key, external_id) VALUES (1, 'g', '1')")
    conn.execute(
        "INSERT INTO image (sample_id, path, width, height, bit_depth, file_size, sha256) "
        "VALUES (1, '/d/1.png', 8, 8, 24, 64, 'h')"
    )
    conn.execute(
        "INSERT INTO split (dataset_id, name, strategy, seed, params) "
        "VALUES (1, 's', 'imported', 0, '{}')"
    )
    conn.execute(
        "INSERT INTO region_profile_revision (dataset_id, name, revision_no, extractor_type, "
        "prepared_width, prepared_height) VALUES (1, 'full frame', 1, 'identity', 8, 8)"
    )
    conn.execute(
        "INSERT INTO experiment (name, dataset_id, split_id, region_profile_id, "
        "region_manifest_sha256, model_type, artifact_dir) "
        "VALUES ('e', 1, 1, 1, 'sha', 'pixel_reference', '/artifacts/1')"
    )
    experiment = conn.execute("SELECT task, channels, classes FROM experiment").fetchone()
    assert tuple(experiment) == ("anomaly", "[]", "[]")
    conn.execute(
        "INSERT INTO image_result (experiment_id, image_id, score, inference_ms) "
        "VALUES (1, 1, 0.5, 1.0)"
    )
    conn.execute(
        "INSERT INTO sample_result (experiment_id, sample_id, agg_score, aggregation, "
        "normalization) VALUES (1, 1, 0.5, 'max', 'none')"
    )
    conn.execute("INSERT INTO job (kind, experiment_id) VALUES ('train', 1)")

    # A result row that does not say how it was reduced, or a metric set that does not say
    # what truth it measured, is not a row.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sample_result SET normalization = NULL")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO metric_set (experiment_id, subset) VALUES (1, 'test')")

    # The vocabularies are open ...
    conn.execute("UPDATE sample_result SET aggregation = 'a_future_mode'")
    conn.execute("UPDATE experiment SET task = 'a_future_task'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE experiment SET task = NULL")
    # ... and the shapes are not.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE job SET status = 'paused'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE job SET progress = 2.0")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE image_result SET localized = 2")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE sample_result SET localized = -1")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE experiment SET classes = 'not json'")

    # NULL is "not checked", and the flags are three-valued.
    row = conn.execute("SELECT peak_x, peak_y, localized FROM image_result").fetchone()
    assert tuple(row) == (None, None, None)

    # Cascades reach every result table and the run's jobs.
    conn.execute("DELETE FROM experiment WHERE id = 1")
    for table in ("job", "image_result", "sample_result"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_dropped_columns_stay_dropped(migrated_db: sqlite3.Connection) -> None:
    profile = _columns(migrated_db, "region_profile_revision")
    assert "seed" not in profile
    assert "failure_policy" not in profile
    assert _columns(migrated_db, "sample_result")["normalization"]["notnull"] == 1
    assert _columns(migrated_db, "metric_set")["ground_truth_digest"]["notnull"] == 1
