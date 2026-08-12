"""Migration runner and schema v1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.db import migrate
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import (
    Migration,
    MigrationError,
    apply_migrations,
    apply_migrations_to,
    contains_transaction_control,
    current_schema_version,
    discover_migrations,
)

# The canonical domain entities of ADR-0005, one table each.
EXPECTED_TABLES = {
    "annotation_draft",
    "annotation_label",
    "annotation_revision",
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


def test_discovers_migrations_in_order() -> None:
    migrations = discover_migrations()
    assert migrations, "no migration files were found in package data"
    assert [m.number for m in migrations] == sorted(m.number for m in migrations)
    assert migrations[0].number == 1


def test_applies_schema_v1(settings: Settings) -> None:
    version = apply_migrations(settings.db_path)

    assert version == len(discover_migrations())
    with connect(settings.db_path) as conn:
        assert current_schema_version(conn) == version
        assert _table_names(conn) >= EXPECTED_TABLES
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(metric_set)")}
        assert "ground_truth_digest" in columns


def test_is_idempotent(settings: Settings) -> None:
    first = apply_migrations(settings.db_path)
    second = apply_migrations(settings.db_path)

    assert first == second
    with connect(settings.db_path) as conn:
        assert current_schema_version(conn) == second


def test_creates_parent_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deeper" / "app.sqlite3"
    apply_migrations(db_path)
    assert db_path.exists()


def test_rejects_a_version_ahead_of_the_files(settings: Settings) -> None:
    """A database from a newer checkout must not be silently re-migrated."""
    apply_migrations(settings.db_path)
    with connect(settings.db_path) as conn:
        conn.execute("PRAGMA user_version = 99")

    with connect(settings.db_path) as conn:
        # Every file is already below the recorded version, so nothing is applied.
        assert apply_migrations_to(conn) == 99


def test_reports_a_gap_in_the_sequence(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing file must fail loudly rather than apply the next one out of order."""
    orphan = Migration(number=7, name="007_orphan.sql", sql="CREATE TABLE t (id INTEGER);")
    monkeypatch.setattr(migrate, "discover_migrations", lambda: [orphan])

    with connect(settings.db_path) as conn, pytest.raises(MigrationError, match="gap"):
        apply_migrations_to(conn)


def test_foreign_keys_and_wal_are_enabled(migrated_db: sqlite3.Connection) -> None:
    assert migrated_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert str(migrated_db.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


def test_foreign_keys_are_enforced(migrated_db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            "INSERT INTO channel (dataset_id, name, position) VALUES (?, ?, ?)",
            (999, "bright", 0),
        )


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


def test_annotation_migration_backfills_a_default_taxonomy(settings: Settings) -> None:
    with connect(settings.db_path) as conn:
        first = next(m for m in discover_migrations() if m.number == 1)
        conn.executescript(f"BEGIN;\n{first.sql}\nPRAGMA user_version = 1;\nCOMMIT;")
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('kept', '/tmp/kept')")

        assert apply_migrations_to(conn) >= 5
        row = conn.execute("SELECT key, name FROM annotation_label WHERE dataset_id = 1").fetchone()
        assert tuple(row) == ("defect", "Defect")


def test_region_profile_revisions_are_dataset_owned_and_immutable(
    migrated_db: sqlite3.Connection,
) -> None:
    migrated_db.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
    migrated_db.execute(
        """
        INSERT INTO region_profile_revision (
            dataset_id, name, revision_no, extractor_type, extractor_config,
            prepared_width, prepared_height, seed
        ) VALUES (1, 'dominant object', 1, 'identity', '{}', 256, 256, 17)
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        migrated_db.execute(
            "UPDATE region_profile_revision SET extractor_type = 'other' WHERE id = 1"
        )
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(
            """
            INSERT INTO region_profile_revision (
                dataset_id, name, revision_no, extractor_type, extractor_config,
                prepared_width, prepared_height, failure_policy, seed
            ) VALUES (1, 'bad fallback', 1, 'identity', '{}', 256, 256, 'identity', 17)
            """
        )

    migrated_db.execute("DELETE FROM dataset WHERE id = 1")
    assert migrated_db.execute("SELECT COUNT(*) FROM region_profile_revision").fetchone()[0] == 0


def test_a_failed_migration_leaves_nothing_behind(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner wraps each file in a transaction, so a failure is all-or-nothing."""
    broken = Migration(
        number=1,
        name="001_broken.sql",
        sql="CREATE TABLE good (id INTEGER);\nCREATE TABLE bad (this is not sql);",
    )
    monkeypatch.setattr(migrate, "discover_migrations", lambda: [broken])

    with connect(settings.db_path) as conn:
        with pytest.raises(MigrationError):
            apply_migrations_to(conn)

        assert current_schema_version(conn) == 0
        assert "good" not in _table_names(conn)


def test_a_migration_file_may_not_manage_its_own_transaction() -> None:
    """An inner COMMIT would end the runner's transaction and leave a partial schema."""
    for migration in discover_migrations():
        assert not contains_transaction_control(migration.sql), migration.name


def test_a_distill_job_is_accepted_and_a_nonsense_kind_is_not(
    migrated_db: sqlite3.Connection,
) -> None:
    """Migration 002 widened the `kind` CHECK, and the CHECK is what makes it a real list."""
    migrated_db.execute("INSERT INTO job (kind, params) VALUES ('distill', '{}')")
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute("INSERT INTO job (kind, params) VALUES ('ponder', '{}')")


def test_region_preparation_job_kind_and_profile_resample_are_migrated(
    migrated_db: sqlite3.Connection,
) -> None:
    migrated_db.execute("INSERT INTO job (kind, params) VALUES ('region_prepare', '{}')")
    columns = {
        str(row["name"])
        for row in migrated_db.execute("PRAGMA table_info(region_profile_revision)")
    }
    assert "resample" in columns


def test_rebuilding_the_job_table_kept_its_rows_and_its_foreign_key(
    settings: Settings,
) -> None:
    """002 drops and recreates `job`. A rebuild that loses rows is the classic way to do it.

    Written against a database at version 1 rather than a migrated one, because the thing
    under test is the *copy*, and a table with no rows in it copies perfectly.
    """
    with connect(settings.db_path) as conn:
        monkeyed = [m for m in discover_migrations() if m.number == 1]
        for migration in monkeyed:
            conn.executescript(f"BEGIN;\n{migration.sql}\nPRAGMA user_version = 1;\nCOMMIT;")
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
        conn.execute(
            "INSERT INTO job (kind, params, status, message) "
            "VALUES ('import', '{\"a\": 1}', 'succeeded', 'kept')"
        )
        conn.commit()

        assert apply_migrations_to(conn) >= 2
        row = conn.execute("SELECT kind, params, status, message FROM job").fetchone()
        assert tuple(row) == ("import", '{"a": 1}', "succeeded", "kept")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        # The index the rebuild recreates, without which every queue read is a table scan.
        # PRAGMA index_list rows are (seq, name, unique, origin, partial).
        names = {r[1] for r in conn.execute("PRAGMA index_list('job')")}
        assert {"idx_job_experiment", "idx_job_status"} <= names


def test_the_teacher_backfill_only_touches_records_that_predate_the_field(
    settings: Settings,
) -> None:
    """003 writes history down; it must not rewrite a run that already recorded its teacher.

    The point of the backfill is that a configuration is the record of what was run. An
    absent field takes the current default, so when the default moved to `nelson1425` every
    earlier run would have started claiming a teacher it never saw.
    """
    with connect(settings.db_path) as conn:
        first = next(m for m in discover_migrations() if m.number == 1)
        conn.executescript(f"BEGIN;\n{first.sql}\nPRAGMA user_version = 1;\nCOMMIT;")
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/tmp/d')")
        conn.execute("INSERT INTO split (dataset_id, name, strategy, seed) VALUES (1, 's', 'x', 0)")
        for name, model_type, config in (
            ("old", "efficientad_custom", '{"max_steps": 4000}'),
            ("new", "efficientad_custom", '{"teacher_source": "nelson1425"}'),
            ("other", "pixel_reference", '{"score_percentile": 99.0}'),
        ):
            conn.execute(
                "INSERT INTO experiment (name, dataset_id, split_id, model_type, "
                "model_config, artifact_dir) VALUES (?, 1, 1, ?, ?, '/tmp/a')",
                (name, model_type, config),
            )
        conn.commit()

        for migration in discover_migrations():
            if 1 < migration.number <= 3:
                conn.executescript(
                    f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.number};\nCOMMIT;"
                )
        found = dict(
            conn.execute(
                "SELECT name, json_extract(model_config, '$.teacher_source') FROM experiment"
            )
        )
        assert found["old"] == "anomalib"
        assert found["new"] == "nelson1425"
        # A method with no such field must not grow one, or its config stops validating.
        assert found["other"] is None

        # The current M10 migration deliberately discards every pre-region experiment:
        # none of these rows can truthfully identify the prepared pixels it saw.
        apply_migrations_to(conn)
        assert conn.execute("SELECT COUNT(*) FROM experiment").fetchone()[0] == 0
