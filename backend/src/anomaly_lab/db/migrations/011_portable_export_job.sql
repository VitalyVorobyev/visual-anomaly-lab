-- Portable export is one more kind-agnostic worker job (ADR-0034). SQLite cannot widen
-- a CHECK in place, so preserve every row while rebuilding the table.
CREATE TABLE job_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL
                          CHECK (kind IN ('import', 'reference_import', 'verify', 'prewarm',
                                          'train', 'infer', 'distill', 'model_asset_download',
                                          'region_prepare', 'export')),
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
