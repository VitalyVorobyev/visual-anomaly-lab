"""SQLite connection factory.

`foreign_keys` and WAL journalling are enabled on every connection (§4). Connections are
created per unit of work and never shared between threads — `sqlite3` objects are not
thread-portable, and FastAPI runs synchronous route handlers in a threadpool.

Connections are opened in autocommit mode (`isolation_level=None`) so that transaction
boundaries are explicit rather than inferred by the driver. They are written in one place,
`transaction`, which every multi-statement write goes through.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# ADR-0004 accepts `database is locked` contention as a cost of SQLite's single writer.
# A busy timeout is the standard mitigation: block briefly instead of failing instantly.
BUSY_TIMEOUT_MS = 5000


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas every connection in this app must have."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS:d}")
    return conn


@contextmanager
def connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection and guarantee it is closed."""
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


class Transaction:
    """An open write transaction, and the undo work that goes with it.

    The filesystem cannot join a database transaction, so a file written while one is open
    has to be taken back by hand if the rows that would have referenced it are not
    committed. `on_rollback` registers that work; `transaction` runs it, newest first,
    exactly when it rolls back.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._undo: list[Callable[[], object]] = []

    def on_rollback(self, callback: Callable[[], object]) -> None:
        """Run `callback` if this transaction rolls back. Never run on commit."""
        self._undo.append(callback)

    def remove_on_rollback(self, path: Path) -> None:
        """Delete `path` if this transaction rolls back — for a file it is about to write."""
        self.on_rollback(lambda: path.unlink(missing_ok=True))

    def _run_undo(self) -> None:
        # Newest first, and every callback runs: one cleanup failing must neither mask the
        # exception that caused the rollback nor leave the remaining files behind.
        for callback in reversed(self._undo):
            try:
                callback()
            except Exception:
                logger.exception("A rollback cleanup callback failed")
        self._undo.clear()


@contextmanager
def transaction(conn: sqlite3.Connection, *, immediate: bool = False) -> Iterator[Transaction]:
    """Commit on success, roll back on any exception, then run the rollback callbacks.

    `immediate=True` issues `BEGIN IMMEDIATE`, taking the write lock up front. Use it for
    read-check-write sequences — a draft's version check, a deletion's blocker check — so
    no other writer can change what was checked before the write commits. A plain `BEGIN`
    is enough for a batch of writes that reads nothing it depends on.

    A failed `COMMIT` rolls back like any other failure. `BaseException` rather than
    `Exception`, so a cancelled request cannot leave a transaction open on a connection.
    """
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    tx = Transaction(conn)
    try:
        yield tx
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        tx._run_undo()
        raise
