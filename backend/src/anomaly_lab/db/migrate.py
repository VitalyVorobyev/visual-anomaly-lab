"""The catalogue schema: one script, stamped into `PRAGMA user_version` (ADR-0004).

There is exactly one schema script, `migrations/001_initial.sql`, and one constant,
`SCHEMA_VERSION`, that names it. Opening a catalogue does one of three things:

- an empty database (version 0, no tables) gets the script, and the version is stamped;
- a database already at `SCHEMA_VERSION` is left alone;
- anything else is refused with `SchemaVersionError`. Nothing is ever migrated: a catalogue
  from another version of the schema is deleted and started fresh.

Changing the script means bumping `SCHEMA_VERSION`, so that every catalogue written by the
previous script is refused rather than opened against tables it does not have. A value is
never reused.

`apply_schema` is idempotent, so it can be called both from the serve entrypoint (before the
port is announced) and from the application lifespan (so the `uv run uvicorn` development
path gets the schema too).
"""

from __future__ import annotations

import re
import sqlite3
from importlib.resources import files
from pathlib import Path

from anomaly_lab.db.connection import connection

#: The version the schema script stamps. Bump it whenever the script changes.
SCHEMA_VERSION = 27

SCHEMA_SCRIPT = "001_initial.sql"

# Transaction control is added by `apply_schema_to`, not by the script.
_TRANSACTION_CONTROL = re.compile(
    r"^\s*(BEGIN|COMMIT|ROLLBACK|END)\b", re.IGNORECASE | re.MULTILINE
)
_TRIGGER_BLOCK = re.compile(
    r"\bCREATE\s+TRIGGER\b.*?^\s*END\s*;", re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def contains_transaction_control(sql: str) -> bool:
    """Reject transaction statements, without mistaking a trigger body for one."""
    return _TRANSACTION_CONTROL.search(_TRIGGER_BLOCK.sub("", sql)) is not None


class SchemaError(RuntimeError):
    """The schema script is malformed or failed to apply."""


class SchemaVersionError(RuntimeError):
    """The catalogue was written by a different schema and cannot be opened.

    The message is meant for a person: it is what the desktop shell shows in place of the
    workbench, so it names the file to delete.
    """

    def __init__(self, db_path: str, found: int) -> None:
        self.db_path = db_path
        self.found = found
        if found > SCHEMA_VERSION:
            message = (
                "This catalogue was created by a newer version of anomaly lab and cannot be "
                f"opened. Update the app, or delete {db_path} (and the app data directory "
                "beside it) to start fresh."
            )
        else:
            message = (
                "This catalogue was created by an older version of anomaly lab and cannot be "
                f"opened. Delete {db_path} (and the app data directory beside it) to start "
                "fresh."
            )
        super().__init__(message)


def schema_sql() -> str:
    """The schema script from package data."""
    sql = files("anomaly_lab.db").joinpath("migrations", SCHEMA_SCRIPT).read_text("utf-8")
    if contains_transaction_control(sql):
        msg = (
            f"{SCHEMA_SCRIPT} contains transaction control. The runner wraps the script in "
            "BEGIN/COMMIT; an inner COMMIT would leave a partial schema applied."
        )
        raise SchemaError(msg)
    return sql


def current_schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _database_path(conn: sqlite3.Connection) -> str:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main":
            return str(row[2]) or ":memory:"
    return ":memory:"  # pragma: no cover - every connection has a main database


def apply_schema_to(conn: sqlite3.Connection) -> int:
    """Create the schema on an empty database, or confirm it is current. Returns the version.

    Raises `SchemaVersionError` for a database stamped with any other version, including a
    version-0 database that already holds tables.
    """
    version = current_schema_version(conn)
    if version == SCHEMA_VERSION:
        return version

    has_tables = conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone() is not None
    if version != 0 or has_tables:
        raise SchemaVersionError(_database_path(conn), version)

    # `executescript` implicitly COMMITs any pending transaction before it runs, so an
    # outer BEGIN issued via `execute` would be discarded. The transaction has to live
    # inside the script itself for the schema to be applied atomically.
    #
    # `PRAGMA user_version = ?` cannot be parameterized; the value is an int constant.
    script = f"BEGIN;\n{schema_sql()}\nPRAGMA user_version = {SCHEMA_VERSION:d};\nCOMMIT;"
    try:
        conn.executescript(script)
    except sqlite3.Error as exc:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        msg = f"The schema script {SCHEMA_SCRIPT} failed: {exc}"
        raise SchemaError(msg) from exc
    return SCHEMA_VERSION


def apply_schema(db_path: Path) -> int:
    """Open `db_path`, creating its parent directory, and make sure it holds the schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connection(db_path) as conn:
        return apply_schema_to(conn)
