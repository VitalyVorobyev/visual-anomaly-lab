-- 001_initial.sql — schema v1 (ADR-0004, ADR-0005)
--
-- This file is wrapped in BEGIN/COMMIT by anomaly_lab.db.migrate, which also sets
-- PRAGMA user_version. Do not add transaction control statements here.
--
-- One table per canonical domain entity, snake_case singular:
--   Dataset Channel Sample Image Mask Split SplitAssignment
--   Experiment Job ImageResult SampleResult MetricSet
--
-- Two invariants carry most of the model's weight (ADR-0005):
--
--   * The Sample owns the label and the split assignment. There is deliberately no
--     image-level assignment table, so every view of a part necessarily shares a
--     subset and cross-channel leakage is structurally impossible.
--
--   * Channel is data, not schema. Channels are rows in a per-dataset dictionary,
--     never columns and never an enum; `image.channel_id` is nullable so single-view
--     datasets need no synthetic channel. No count of channels is encoded anywhere.
--
-- Every table uses AUTOINCREMENT: ids are referenced by artifact paths on disk
-- (`data/artifacts/exp-<id>/`) and by the thumbnail cache (keyed by image id, §9), so
-- rowid reuse after a delete could silently attach stale files to a new row.

CREATE TABLE dataset (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    root_path   TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes       TEXT
);

CREATE TABLE channel (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    position    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (dataset_id, name)
);

CREATE INDEX idx_channel_dataset ON channel (dataset_id, position);

CREATE TABLE sample (
    id            INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    group_key     TEXT    NOT NULL,
    external_id   TEXT    NOT NULL,
    label         TEXT    NOT NULL DEFAULT 'unlabeled'
                          CHECK (label IN ('normal', 'defect', 'unlabeled')),
    label_source  TEXT    NOT NULL DEFAULT 'import'
                          CHECK (label_source IN ('import', 'manual')),
    notes         TEXT,
    -- Numeric sample ids collide across capture groups in real data, so identity is
    -- the pair, not the id alone.
    UNIQUE (dataset_id, group_key, external_id)
);

CREATE INDEX idx_sample_dataset_label ON sample (dataset_id, label);

CREATE TABLE image (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id    INTEGER NOT NULL REFERENCES sample (id) ON DELETE CASCADE,
    channel_id   INTEGER REFERENCES channel (id) ON DELETE RESTRICT,
    path         TEXT    NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    bit_depth    INTEGER NOT NULL,
    file_size    INTEGER NOT NULL,
    sha256       TEXT    NOT NULL,
    imported_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_image_sample ON image (sample_id);
CREATE INDEX idx_image_channel ON image (channel_id);
-- Duplicate-hash detection at import, and `verify` lookups (ADR-0006).
CREATE INDEX idx_image_sha256 ON image (sha256);

-- Pixel-level ground truth. Defined in schema v1 but deliberately unpopulated: the
-- reference dataset has no masks, so only image-level metrics are computable (§8).
CREATE TABLE mask (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id  INTEGER NOT NULL REFERENCES image (id) ON DELETE CASCADE,
    path      TEXT    NOT NULL,
    kind      TEXT    NOT NULL
);

CREATE INDEX idx_mask_image ON mask (image_id);

CREATE TABLE split (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    strategy    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Splits are immutable once created; changing one means creating a new one.
    UNIQUE (dataset_id, name)
);

CREATE TABLE split_assignment (
    split_id   INTEGER NOT NULL REFERENCES split (id) ON DELETE CASCADE,
    sample_id  INTEGER NOT NULL REFERENCES sample (id) ON DELETE CASCADE,
    subset     TEXT    NOT NULL CHECK (subset IN ('train', 'val', 'test')),
    PRIMARY KEY (split_id, sample_id)
);

CREATE INDEX idx_split_assignment_sample ON split_assignment (sample_id);
CREATE INDEX idx_split_assignment_subset ON split_assignment (split_id, subset);

CREATE TABLE experiment (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    name                 TEXT    NOT NULL,
    -- RESTRICT, not CASCADE: an experiment freezes its configuration at creation, so
    -- deleting the dataset or split it was computed against must not silently orphan it.
    dataset_id           INTEGER NOT NULL REFERENCES dataset (id) ON DELETE RESTRICT,
    split_id             INTEGER NOT NULL REFERENCES split (id) ON DELETE RESTRICT,
    -- Registry key from MODEL_REGISTRY (ADR-0007), e.g. 'classical_circular'.
    model_type           TEXT    NOT NULL,
    model_config         TEXT    NOT NULL DEFAULT '{}',
    preprocessing_config TEXT    NOT NULL DEFAULT '{}',
    eval_config          TEXT    NOT NULL DEFAULT '{}',
    status               TEXT    NOT NULL DEFAULT 'draft'
                                 CHECK (status IN ('draft', 'training', 'trained', 'failed')),
    artifact_dir         TEXT    NOT NULL,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes                TEXT
);

CREATE INDEX idx_experiment_dataset ON experiment (dataset_id);
CREATE INDEX idx_experiment_split ON experiment (split_id);

CREATE TABLE job (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL CHECK (kind IN ('import', 'train', 'infer')),
    -- Nullable: import jobs belong to no experiment.
    experiment_id INTEGER REFERENCES experiment (id) ON DELETE CASCADE,
    status        TEXT    NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    progress      REAL    NOT NULL DEFAULT 0.0 CHECK (progress BETWEEN 0.0 AND 1.0),
    message       TEXT,
    log_path      TEXT,
    -- Per-kind payload as JSON. `experiment_id` identifies what a train/infer job acts
    -- on, but an import job needs its dataset, adapter and manifest path recorded too.
    params        TEXT    NOT NULL DEFAULT '{}',
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT
);

CREATE INDEX idx_job_experiment ON job (experiment_id);
-- Queue reads and the startup reconciliation of stale `running` jobs (ADR-0009).
CREATE INDEX idx_job_status ON job (status, id);

CREATE TABLE image_result (
    experiment_id INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    image_id      INTEGER NOT NULL REFERENCES image (id) ON DELETE CASCADE,
    -- Higher means more anomalous (ADR-0007).
    score         REAL    NOT NULL,
    -- float32 .npy under the experiment's maps/ dir; NULL when the model emits no map.
    map_path      TEXT,
    inference_ms  REAL    NOT NULL,
    PRIMARY KEY (experiment_id, image_id)
);

CREATE INDEX idx_image_result_image ON image_result (image_id);
CREATE INDEX idx_image_result_score ON image_result (experiment_id, score);

CREATE TABLE sample_result (
    experiment_id INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    sample_id     INTEGER NOT NULL REFERENCES sample (id) ON DELETE CASCADE,
    agg_score     REAL    NOT NULL,
    -- Recorded per row so a stored result is self-describing (ADR-0011).
    aggregation   TEXT    NOT NULL CHECK (aggregation IN ('max', 'mean')),
    PRIMARY KEY (experiment_id, sample_id)
);

CREATE INDEX idx_sample_result_sample ON sample_result (sample_id);
CREATE INDEX idx_sample_result_score ON sample_result (experiment_id, agg_score);

CREATE TABLE metric_set (
    experiment_id INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    subset        TEXT    NOT NULL CHECK (subset IN ('train', 'val', 'test')),
    -- Threshold-independent metrics only. Nothing that depends on a decision
    -- threshold is persisted; those are computed on demand (ADR-0011).
    metrics       TEXT    NOT NULL DEFAULT '{}',
    computed_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (experiment_id, subset)
);
