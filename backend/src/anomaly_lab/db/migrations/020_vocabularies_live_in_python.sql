-- `job.kind` and `sample_result.aggregation` stop being closed lists in the schema.
--
-- Both were CHECK constraints naming every allowed value, and SQLite cannot alter a CHECK:
-- widening one means rebuilding the whole table. `job.kind` has paid that five times
-- (migrations 002, 004, 007, 009, 011), which made "a new job kind costs one entry in
-- `jobs/handlers.py` and one handler" true of the runtime and false of the schema
-- The vocabularies already live in Python, where every
-- write goes through them: `JobKind` (`create_job` coerces through it, so an unknown kind
-- is a `ValueError` before any SQL runs) and `Aggregation` (a field of the pydantic
-- `SampleResult` every row is written from). Two lists that must agree is one list too
-- many, and the one that cost a migration to extend is the one that goes.
--
-- The CHECKs that describe *shape* rather than an extensible vocabulary stay: a job's
-- `status` lifecycle, `progress` in [0, 1], and `localized` as a three-valued flag.
--
-- Both tables are unreferenced by any foreign key, so each is rebuilt as in 011: create,
-- copy every row, drop, rename, recreate the indexes.

CREATE TABLE job_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- One of `JobKind`. Validated in Python; see the header.
    kind          TEXT    NOT NULL,
    experiment_id INTEGER REFERENCES experiment (id) ON DELETE CASCADE,
    status        TEXT    NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    progress      REAL    NOT NULL DEFAULT 0.0 CHECK (progress BETWEEN 0.0 AND 1.0),
    message       TEXT,
    log_path      TEXT,
    params        TEXT    NOT NULL DEFAULT '{}',
    result        TEXT    NOT NULL DEFAULT '{}',
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT
);

INSERT INTO job_new (id, kind, experiment_id, status, progress, message, log_path,
                     params, result, started_at, finished_at, error)
SELECT id, kind, experiment_id, status, progress, message, log_path,
       params, result, started_at, finished_at, error
FROM job;

-- AUTOINCREMENT promises an id is never reused, and a job's id names its log file. The
-- copy above only advances the new table's counter to the largest id still present, so a
-- deleted job with a higher id would have its number handed out again. Carry the old
-- counter across the rebuild, which the earlier rebuilds did not: deleting an experiment or a
-- dataset deletes its jobs, so the newest id is not always still present.
CREATE TEMP TABLE job_sequence AS SELECT seq FROM sqlite_sequence WHERE name = 'job';

DROP TABLE job;
ALTER TABLE job_new RENAME TO job;

INSERT INTO sqlite_sequence (name, seq)
SELECT 'job', seq FROM temp.job_sequence
 WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'job');
UPDATE sqlite_sequence
   SET seq = (SELECT seq FROM temp.job_sequence)
 WHERE name = 'job' AND seq < (SELECT seq FROM temp.job_sequence);
DROP TABLE temp.job_sequence;

CREATE INDEX idx_job_experiment ON job (experiment_id);
CREATE INDEX idx_job_status ON job (status, id);

CREATE TABLE sample_result_new (
    experiment_id INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    sample_id     INTEGER NOT NULL REFERENCES sample (id) ON DELETE CASCADE,
    agg_score     REAL    NOT NULL,
    -- One of `Aggregation`, recorded per row so a stored result is self-describing
    -- (handbook evaluation.md). Validated in Python; see the header.
    aggregation   TEXT    NOT NULL,
    -- One of `ChannelNormalization`, or NULL for a row written before migration 014.
    normalization TEXT,
    localized     INTEGER CHECK (localized IS NULL OR localized IN (0, 1)),
    PRIMARY KEY (experiment_id, sample_id)
);

INSERT INTO sample_result_new (experiment_id, sample_id, agg_score, aggregation,
                               normalization, localized)
SELECT experiment_id, sample_id, agg_score, aggregation, normalization, localized
FROM sample_result;

DROP TABLE sample_result;
ALTER TABLE sample_result_new RENAME TO sample_result;

CREATE INDEX idx_sample_result_sample ON sample_result (sample_id);
CREATE INDEX idx_sample_result_score ON sample_result (experiment_id, agg_score);
