# ADR-0004: Persistence with SQLite and filesystem artifacts

**Status:** Accepted (2026-08-06)

## Context

The workbench persists datasets, labels, splits, experiment configurations, job state, per-image
scores and metrics, and must reopen a past experiment and show its results. It is single-user and
local; a database server or a cloud service would be pure overhead.

The data has two shapes. One is small, relational and queried by predicate — "all defect samples in
the test split of dataset X, ranked by score". The other is large, opaque and always fetched whole —
an anomaly map, a memory bank, a training log. Storing the second kind in a database buys nothing
and makes the database unwieldy.

The live alternatives were an ORM (SQLAlchemy, SQLModel) over SQLite, an embedded document store,
and blobs in the database.

## Decision

**SQLite for metadata, the filesystem for artifacts, both under the data directory** (repo-local
`data/` by default, see ADR-0002).

- **The database** holds every entity of the domain model (see ADR-0005). It stores scores and
  *paths*, never pixels.
- **The filesystem** holds thumbnails, anomaly maps as **float32 `.npy`**, checkpoints, manifests
  and job logs, with artifacts namespaced per experiment. Source images are referenced in place,
  never copied (see ADR-0022).
- **Migrations are plain, forward-only SQL files**, numbered and applied in order, tracked with
  `PRAGMA user_version`. The initial schema is frozen: every change is a new numbered migration,
  never an edit to an applied one.
- **A thin repository layer** of intention-revealing functions over parameterized SQL, returning
  plain dataclasses or pydantic models.
- **No ORM.**

Being able to open the database in any SQLite browser, load an `.npy` directly, and delete the
whole directory to reset is a feature of a research tool.

## Consequences

No database administration, a schema readable in one sitting, and every query visible as SQL.
Artifacts stay out of the database, so it stays small. Maps load into NumPy at full precision, so
colormap and threshold decisions stay in the view rather than being baked in at write time.

- **Hand-written SQL is hand-maintained.** A schema change touches the migration, the repository
  functions and the dataclasses, with nothing verifying they agree.
- **No relationship loading.** A sample with its images and results is an explicit join; N+1
  patterns are easy to write by accident.
- **Migrations are forward-only and unchecksummed.** Editing an applied migration silently
  desynchronizes machines.
- **Two stores can disagree.** No transaction spans the database and the filesystem, so a deleted
  row can orphan a directory and a crashed job can leave an `.npy` with no row; startup has to
  reconcile what it can.
- **One writer at a time.** A job writing results while the UI writes labels can hit
  `database is locked`. WAL mode and short transactions mitigate it; this is one reason jobs are
  serialized (see ADR-0009).
- **Raw float maps are storage-hungry**, at several megabytes per full-resolution map. Retention
  will need managing.
