"""Migration runner and schema v1."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.db import migrate
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import (
    _TRANSACTION_CONTROL,
    Migration,
    MigrationError,
    apply_migrations,
    apply_migrations_to,
    current_schema_version,
    discover_migrations,
)

# The canonical domain entities of ADR-0005, one table each.
EXPECTED_TABLES = {
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
        assert not _TRANSACTION_CONTROL.search(migration.sql), migration.name
