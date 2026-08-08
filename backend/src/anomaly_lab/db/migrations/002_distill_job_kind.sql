-- Teacher distillation is a job.
--
-- It is hours long, it needs progress, cancellation, a log and a resume — which is the
-- entire argument for reusing the job system rather than writing a script with its own
-- progress mechanism. It belongs to no experiment, like `import` and `verify`, and its
-- `experiment_id` stays null; what it produces is an *asset* in the model cache that
-- experiments then name in their configuration.
--
-- SQLite cannot alter a CHECK constraint, so the table is rebuilt: copy out, drop, rename.
--
-- **No `PRAGMA foreign_keys` here, deliberately.** The runner wraps every migration in
-- BEGIN/COMMIT and that pragma is a documented no-op inside a transaction, so writing it
-- would claim a protection this file does not have. It needs none: `job` has an outgoing
-- reference to `experiment` and no table references `job`, so neither the drop nor the
-- rename can orphan a row.

CREATE TABLE job_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- `import` scans a tree into a manifest, `verify` re-hashes a dataset's recorded
    -- files, `prewarm` renders its thumbnail tiers, `train` and `infer` run a method
    -- against an experiment, and `distill` compresses a frozen source model into the
    -- compact PDN teacher those methods are distilled against.
    kind          TEXT    NOT NULL
                          CHECK (kind IN ('import', 'verify', 'prewarm', 'train', 'infer',
                                          'distill')),
    -- Nullable: import, verify, prewarm and distill jobs belong to no experiment.
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

DROP TABLE job;
ALTER TABLE job_new RENAME TO job;

CREATE INDEX idx_job_experiment ON job (experiment_id);
CREATE INDEX idx_job_status ON job (status, id);
