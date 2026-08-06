"""SQLite connection factory.

`foreign_keys` and WAL journalling are enabled on every connection (§4). Connections are
created per unit of work and never shared between threads — `sqlite3` objects are not
thread-portable, and FastAPI runs synchronous route handlers in a threadpool.

Connections are opened in autocommit mode (`isolation_level=None`) so that transaction
boundaries are written explicitly in SQL rather than inferred by the driver.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
