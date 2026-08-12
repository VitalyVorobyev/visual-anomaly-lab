"""Forward-only SQL migrations tracked by `PRAGMA user_version` (ADR-0004).

Migrations are plain SQL files named `NNN_description.sql`, applied in numeric order.
There are no down-migrations and no checksums; ADR-0004 accepts that cost explicitly.

`apply_migrations` is idempotent, so it can be called both from the serve entrypoint
(before the port is announced) and from the application lifespan (so the `uv run uvicorn`
development path migrates too).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from anomaly_lab.db.connection import connection

_MIGRATION_FILENAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

# Transaction control is added by the runner, not by the migration file.
_TRANSACTION_CONTROL = re.compile(
    r"^\s*(BEGIN|COMMIT|ROLLBACK|END)\b", re.IGNORECASE | re.MULTILINE
)
_TRIGGER_BLOCK = re.compile(
    r"\bCREATE\s+TRIGGER\b.*?^\s*END\s*;", re.IGNORECASE | re.MULTILINE | re.DOTALL
)


def contains_transaction_control(sql: str) -> bool:
    """Reject transaction statements, without mistaking a trigger body for one."""
    return _TRANSACTION_CONTROL.search(_TRIGGER_BLOCK.sub("", sql)) is not None


class MigrationError(RuntimeError):
    """A migration file is malformed, out of sequence, or failed to apply."""


@dataclass(frozen=True)
class Migration:
    number: int
    name: str
    sql: str


def discover_migrations() -> list[Migration]:
    """Load migration files from package data, sorted by number."""
    root = files("anomaly_lab.db").joinpath("migrations")
    found: dict[int, Migration] = {}

    for entry in root.iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _MIGRATION_FILENAME.match(entry.name)
        if match is None:
            msg = f"Migration {entry.name!r} does not match the NNN_description.sql convention"
            raise MigrationError(msg)

        number = int(match.group(1))
        if number in found:
            msg = f"Duplicate migration number {number:03d}: {found[number].name} and {entry.name}"
            raise MigrationError(msg)

        sql = entry.read_text(encoding="utf-8")
        if contains_transaction_control(sql):
            msg = (
                f"Migration {entry.name!r} contains transaction control. The runner wraps each "
                "file in BEGIN/COMMIT; an inner COMMIT would leave a partial migration applied."
            )
            raise MigrationError(msg)

        found[number] = Migration(number=number, name=entry.name, sql=sql)

    return [found[number] for number in sorted(found)]


def current_schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations_to(conn: sqlite3.Connection) -> int:
    """Apply every pending migration in order. Returns the resulting schema version."""
    version = current_schema_version(conn)

    for migration in discover_migrations():
        if migration.number <= version:
            continue
        if migration.number != version + 1:
            msg = (
                f"Migration sequence has a gap: database is at version {version}, "
                f"next file on disk is {migration.name}"
            )
            raise MigrationError(msg)

        # `executescript` implicitly COMMITs any pending transaction before it runs, so an
        # outer BEGIN issued via `execute` would be discarded. The transaction has to live
        # inside the script itself for the migration to be atomic.
        #
        # `PRAGMA user_version = ?` cannot be parameterized; the value is an int parsed from
        # a \d{3} filename match, so the interpolation below cannot carry anything else.
        script = f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.number:d};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            msg = f"Migration {migration.name} failed: {exc}"
            raise MigrationError(msg) from exc

        version = migration.number

    return version


def apply_migrations(db_path: Path) -> int:
    """Open `db_path`, creating its parent directory, and bring it up to date."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connection(db_path) as conn:
        return apply_migrations_to(conn)
