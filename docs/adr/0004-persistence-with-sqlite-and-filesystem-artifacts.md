# ADR-0004: Persistence with SQLite and filesystem artifacts

**Status:** Accepted (2026-09-26)

## Context

The workbench persists datasets, labels, splits, experiment configurations, job state, per-image
scores and metrics, and must reopen a past experiment and show its results. It is single-user and
local; a database server or a cloud service would be pure overhead.

The data has two shapes. One is small, relational and queried by predicate — "all defect samples in
the test split of dataset X, ranked by score". The other is large, opaque and always fetched whole —
an anomaly map, a memory bank, a training log. Storing the second kind in a database buys nothing
and makes the database unwieldy.

The schema still moves with every task and method the workbench learns, and no catalogue it holds
is yet worth more than re-importing its datasets and re-running its experiments.

## Decision

**SQLite for metadata, the filesystem for artifacts, both under the data directory** (repo-local
`data/` by default, relocatable with `ANOMALY_LAB_DATA_DIR`).

- **The database** holds every entity of the domain model (see ADR-0041). It stores scores and
  *paths*, never pixels.
- **The filesystem** holds thumbnails, anomaly maps as raw float32 arrays, checkpoints, manifests
  and job logs, with artifacts namespaced per experiment. Source images are referenced in place,
  never copied (see ADR-0022).
- **A thin repository layer** of intention-revealing functions over parameterized SQL, returning
  plain dataclasses or pydantic models. **No ORM.**
- **Until a catalogue is worth keeping, the schema is one initial script, rewritten in place.** Its
  version is recorded with `PRAGMA user_version`. A catalogue written under any other version is
  **refused at startup with a message telling the user to delete it**; it is never migrated.
- **Numbered, forward-only migrations begin when the owner declares a catalogue worth keeping.**
  From then on the initial script is frozen and every schema change is a new numbered file.

Being able to open the database in any SQLite browser, load a map with `numpy` directly, and delete
the whole directory to reset is a feature of a research tool.

## Alternatives considered

- **An ORM** (SQLAlchemy, SQLModel). Relationship loading and generated migrations, at the cost of
  a layer between the reader and the SQL, and of a schema that is no longer readable in one sitting.
- **An embedded document store, or blobs in the database.** Arrays in the database make it large and
  put multi-megabyte writes inside transactions, for data that is always read whole.
- **Numbered migrations from the start.** They preserve every catalogue, which is the right trade
  once a catalogue holds something that cannot be recreated. Before that, every schema change
  becomes permanent history, and the history of a schema still being designed is mostly noise a
  reader must replay to learn what a table is.

## Consequences

No database administration, a schema readable in one sitting, and every query visible as SQL.
Artifacts stay out of the database, so it stays small. Maps load at full precision, so colormap and
threshold decisions stay in the view rather than being baked in at write time.

- **A schema change discards every catalogue.** A user rebuilds from the source datasets and re-runs
  what they need. That is accepted while nothing in a catalogue is irreplaceable, and it is the
  signal to start numbering once something is.
- **Hand-written SQL is hand-maintained.** A schema change touches the script, the repository
  functions and the dataclasses, with nothing verifying they agree.
- **No relationship loading.** A sample with its images and results is an explicit join; N+1
  patterns are easy to write by accident.
- **Two stores can disagree.** No transaction spans the database and the filesystem, so a deleted
  row can orphan a directory and a crashed job can leave an array with no row; startup reconciles
  what it can.
- **One writer at a time.** A job writing results while the UI writes labels can hit
  `database is locked`. WAL mode and short transactions mitigate it; this is one reason jobs are
  serialized (see ADR-0009).
- **Raw float maps are storage-hungry**, several megabytes per full-resolution map. Retention will
  need managing.
