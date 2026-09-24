"""`db.connection.transaction`: the one place a write transaction begins and ends."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from anomaly_lab.db.connection import connect, transaction


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "tx.sqlite3")
    connection.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
    try:
        yield connection
    finally:
        connection.close()


def _names(conn: sqlite3.Connection) -> list[str]:
    return [str(row["name"]) for row in conn.execute("SELECT name FROM item ORDER BY id")]


def test_commits_on_success(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.execute("INSERT INTO item (name) VALUES ('a')")
        conn.execute("INSERT INTO item (name) VALUES ('b')")
        assert conn.in_transaction

    assert not conn.in_transaction
    assert _names(conn) == ["a", "b"]


def test_rolls_back_every_statement_on_an_exception(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError, match="boom"), transaction(conn):
        conn.execute("INSERT INTO item (name) VALUES ('a')")
        raise RuntimeError("boom")

    assert not conn.in_transaction
    assert _names(conn) == []


def test_rolls_back_on_a_failing_statement(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError), transaction(conn):
        conn.execute("INSERT INTO item (name) VALUES ('a')")
        conn.execute("INSERT INTO item (name) VALUES ('a')")

    assert _names(conn) == []


def test_rolls_back_on_a_base_exception(conn: sqlite3.Connection) -> None:
    """A cancelled request must not leave the connection holding an open transaction."""
    with pytest.raises(KeyboardInterrupt), transaction(conn):
        conn.execute("INSERT INTO item (name) VALUES ('a')")
        raise KeyboardInterrupt

    assert not conn.in_transaction
    assert _names(conn) == []


def test_immediate_takes_the_write_lock_up_front(tmp_path: Path) -> None:
    """`BEGIN IMMEDIATE` is what serialises a read-check-write against another writer."""
    path = tmp_path / "lock.sqlite3"
    first = connect(path)
    second = sqlite3.connect(path, isolation_level=None, timeout=0)
    try:
        first.execute("CREATE TABLE item (id INTEGER PRIMARY KEY)")
        with (
            transaction(first, immediate=True),
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            second.execute("BEGIN IMMEDIATE")
        second.execute("BEGIN IMMEDIATE")
        second.execute("ROLLBACK")
    finally:
        first.close()
        second.close()


def test_rollback_runs_the_cleanups_newest_first(conn: sqlite3.Connection, tmp_path: Path) -> None:
    written = tmp_path / "revision-1.png"
    order: list[str] = []

    with pytest.raises(RuntimeError), transaction(conn) as tx:
        written.write_bytes(b"png")
        tx.remove_on_rollback(written)
        tx.on_rollback(lambda: order.append("first"))
        tx.on_rollback(lambda: order.append("second"))
        raise RuntimeError("after the file was written")

    assert not written.exists()
    assert order == ["second", "first"]


def test_commit_never_runs_the_cleanups(conn: sqlite3.Connection, tmp_path: Path) -> None:
    written = tmp_path / "revision-1.png"
    called: list[str] = []

    with transaction(conn) as tx:
        written.write_bytes(b"png")
        tx.remove_on_rollback(written)
        tx.on_rollback(lambda: called.append("cleanup"))

    assert written.exists()
    assert called == []


def test_a_failing_cleanup_neither_masks_the_error_nor_skips_the_rest(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    written = tmp_path / "kept-only-if-skipped.png"
    written.write_bytes(b"png")

    def broken() -> None:
        raise OSError("cleanup failed")

    with pytest.raises(ValueError, match="the real failure"), transaction(conn) as tx:
        tx.remove_on_rollback(written)
        tx.on_rollback(broken)
        raise ValueError("the real failure")

    assert not written.exists()


def test_a_missing_file_is_not_a_cleanup_failure(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """A file registered before it was written — the render failed — is simply absent."""
    with pytest.raises(RuntimeError), transaction(conn) as tx:
        tx.remove_on_rollback(tmp_path / "never-written.png")
        raise RuntimeError("render failed")
