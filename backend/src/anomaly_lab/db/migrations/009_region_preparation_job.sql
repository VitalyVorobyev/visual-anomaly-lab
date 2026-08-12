-- A profile now owns the interpolation used to materialise prepared pixels. Existing
-- revisions were authored under the bridge's bilinear default.
ALTER TABLE region_profile_revision
    ADD COLUMN resample TEXT NOT NULL DEFAULT 'bilinear'
        CHECK (resample IN ('nearest', 'bilinear', 'bicubic', 'lanczos'));

-- Preview and full preparation are two modes of one kind-agnostic handler. SQLite
-- cannot widen this CHECK in place, so preserve every job row while rebuilding.
CREATE TABLE job_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL
                          CHECK (kind IN ('import', 'reference_import', 'verify', 'prewarm',
                                          'train', 'infer', 'distill',
                                          'model_asset_download', 'region_prepare')),
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
