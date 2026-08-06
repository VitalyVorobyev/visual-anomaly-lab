# ADR-0004: Persistence with SQLite and filesystem artifacts

**Status:** Accepted (2026-08-06)

## Context

The workbench must persist datasets, labels, split assignments, experiment configurations, job
state, per-image scores, and evaluation metrics, and must be able to reopen a past experiment and
show its results. The brief prescribes "a small local persistence layer such as SQLite" plus
"filesystem storage for datasets, model checkpoints, and generated artifacts", and rules out
multi-user access and cloud services.

The data has two distinct shapes. One is small, relational, and queried by predicate — "all defect
samples in the test split of dataset X, ranked by score". The other is large, opaque, and always
fetched whole — a 1280x1024 anomaly map, a PatchCore memory bank, a training log. Storing the
second kind in a database buys nothing and makes the database unwieldy.

## Decision

**SQLite for metadata, the filesystem for artifacts, both under repo-local `data/`.**

- **The database** (`data/app.sqlite3`) holds every entity from the domain model (see ADR-0005):
  datasets, channels, samples, images, splits, split assignments, experiments, jobs, image results,
  sample results, metric sets. It stores scores and *paths*, never pixels.
- **The filesystem** holds source images (referenced in place, never copied — see ADR-0001),
  thumbnails, anomaly maps as **float32 `.npy`**, model checkpoints, and job logs, organized under
  `data/{thumbnails,artifacts,manifests,exports}/` with artifacts namespaced per experiment.
- **Migrations are plain SQL files** named `NNN_description.sql`, applied in order and tracked with
  SQLite's `PRAGMA user_version`. A handful of lines of Python applies any file whose number exceeds
  the current version.
- **A thin repository layer** wraps the DB: a module per aggregate exposing intention-revealing
  functions (`create_experiment`, `list_samples_for_split`, `bulk_insert_image_results`) over
  parameterized SQL, returning plain dataclasses or pydantic models.
- **No ORM.** Not SQLAlchemy, not SQLModel, not Peewee.

The `data/` root is repo-local by default and relocatable via `ANOMALY_LAB_DATA_DIR` (ADR-0002).
Being able to open `app.sqlite3` in any SQLite browser, look at an `.npy` on disk, and delete the
whole directory to reset is a feature of a research tool, not an oversight.

## Consequences

Zero database administration: no server, no connection pooling, no separate lifecycle. The schema is
readable in one sitting, and every query in the codebase is visible as SQL rather than assembled by
a query builder at runtime. Large artifacts stay out of the DB, so the database file remains small
and fast to back up or inspect. `.npy` maps load directly into NumPy at full float precision, so
colormapping and thresholding decisions stay in the UI layer rather than being baked in at write
time (see ADR-0007).

Negative consequences, accepted honestly:

- **Hand-written SQL is hand-maintained.** Every schema change means touching migrations *and* the
  repository functions *and* the dataclasses, with nothing verifying they agree. Column renames are
  a manual, error-prone sweep. An ORM would have caught some of this.
- **No automatic relationship loading.** Fetching a sample with its images and results is an
  explicit join or several queries; N+1 patterns are easy to write by accident.
- **Migrations are forward-only and untested by default.** `PRAGMA user_version` gives no
  down-migrations and no checksum of applied files; editing an already-applied migration silently
  desynchronizes developer machines.
- **Two stores can disagree.** A deleted experiment row can orphan a directory of checkpoints; a
  crashed job can leave a `.npy` with no result row. There is no transaction spanning both, so a
  reconciliation/cleanup operation will eventually be needed.
- **Concurrency limits.** SQLite tolerates one writer; a background job writing results while the UI
  writes labels can hit `database is locked`. WAL mode and short transactions mitigate but do not
  eliminate this. This is one reason job execution is serialized (see ADR-0009).
- **Anomaly maps are storage-hungry.** A float32 1280x1024 map is ~5 MB; a full inference run over
  the reference dataset produces gigabytes. Retention will eventually need managing.
