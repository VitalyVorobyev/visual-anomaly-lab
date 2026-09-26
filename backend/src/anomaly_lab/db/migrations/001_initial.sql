-- The catalogue schema (ADR-0004, ADR-0005).
--
-- One script, applied once to an empty database by anomaly_lab.db.migrate, which wraps it
-- in BEGIN/COMMIT and stamps `PRAGMA user_version` with `SCHEMA_VERSION`. Do not add
-- transaction control statements here. Changing this script means bumping
-- `SCHEMA_VERSION`: a catalogue stamped with any other version is refused, never migrated.
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
-- (`data/artifacts/exp-<id>/`), by job log files and by the thumbnail cache (keyed by
-- image id), so rowid reuse after a delete could silently attach stale files to a new row.
--
-- Extensible vocabularies — `job.kind`, `experiment.task`, `sample_result.aggregation`
-- and `normalization`, `region_profile_revision.sample_alignment` — are validated in
-- Python (`JobKind`, `Task`, `Aggregation`, `ChannelNormalization`, `SampleAlignment`),
-- where every write goes through them. SQLite cannot alter a CHECK, so a closed list
-- here would make each new value a table rebuild. The CHECKs that remain describe
-- shape: a lifecycle, a range, a three-valued flag.

CREATE TABLE dataset (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL UNIQUE,
    -- Unique so that re-importing a directory updates the dataset it already produced
    -- instead of creating a second one beside it (handbook import.md).
    root_path        TEXT    NOT NULL UNIQUE,
    -- Which adapter proposed this dataset, and the manifest that was accepted. Together
    -- they answer "how did this dataset come to look like this?" months later (ADR-0006).
    adapter          TEXT,
    manifest_path    TEXT,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes            TEXT,
    -- Free text rather than a `collection` table: a collection carries no attributes of
    -- its own. NULL means "not overridden"; a reference pack's title is derived for the
    -- datasets that came from one.
    collection       TEXT,
    -- Whether annotation truth is edited per image or once per sample for every channel
    -- of a part (handbook annotations.md). A property of the data, not a preference.
    annotation_scope TEXT    NOT NULL DEFAULT 'image'
                             CHECK (annotation_scope IN ('image', 'sample')),
    -- The channel a part is normally read in, by name so it survives a re-import that
    -- renumbers the dictionary. NULL resolves to the first channel by position.
    default_channel  TEXT
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
    -- RESTRICT so a channel cannot be dropped out from under the images that use it.
    -- The cost: `DELETE FROM dataset` cannot rely on cascades, because SQLite does not
    -- order them and the channel cascade may run while images still reference it. The
    -- datasets repository therefore deletes explicitly, children first, in one
    -- transaction.
    channel_id   INTEGER REFERENCES channel (id) ON DELETE RESTRICT,
    path         TEXT    NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    bit_depth    INTEGER NOT NULL,
    file_size    INTEGER NOT NULL,
    sha256       TEXT    NOT NULL,
    imported_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- The upsert key that makes re-import idempotent: one file contributes one row to
    -- the sample it belongs to, so a second commit of the same manifest updates rather
    -- than duplicates (handbook import.md).
    UNIQUE (sample_id, path)
);

CREATE INDEX idx_image_sample ON image (sample_id);
CREATE INDEX idx_image_channel ON image (channel_id);
-- Duplicate-hash detection at import, and `verify` lookups (ADR-0006).
CREATE INDEX idx_image_sha256 ON image (sha256);

-- Imported pixel-level ground truth: source provenance, referenced in place.
CREATE TABLE mask (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id  INTEGER NOT NULL REFERENCES image (id) ON DELETE CASCADE,
    path      TEXT    NOT NULL,
    kind      TEXT    NOT NULL,
    -- NULL until the mask is pinned as the base of an annotation draft, which hashes it
    -- then; import does not read mask bytes.
    sha256    TEXT
);

CREATE INDEX idx_mask_image ON mask (image_id);

CREATE TABLE split (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    strategy    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    -- Ratios and stratification key as JSON. A seed alone does not reproduce a split;
    -- the parameters it was drawn under are part of the record (handbook evaluation.md).
    params      TEXT    NOT NULL DEFAULT '{}',
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

-- Dataset-owned immutable input-region configurations (ADR-0033): the frozen
-- configuration an experiment pins, never a mutable "current" value. Build products and
-- their status live on disk under the profile's directory.
CREATE TABLE region_profile_revision (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id        INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    name              TEXT    NOT NULL,
    revision_no       INTEGER NOT NULL CHECK (revision_no >= 1),
    extractor_type    TEXT    NOT NULL,
    extractor_config  TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(extractor_config)),
    prepared_width    INTEGER NOT NULL CHECK (prepared_width > 0),
    prepared_height   INTEGER NOT NULL CHECK (prepared_height > 0),
    padding_fraction  REAL    NOT NULL DEFAULT 0.05
                              CHECK (padding_fraction BETWEEN 0.0 AND 1.0),
    -- The interpolation used to materialise prepared pixels.
    resample          TEXT    NOT NULL DEFAULT 'bilinear'
                              CHECK (resample IN ('nearest', 'bilinear', 'bicubic', 'lanczos')),
    -- One of `SampleAlignment`: each image keeps its own crop, or every image of a sample
    -- gets the union of their crops so the channels of one part stay registered.
    sample_alignment  TEXT    NOT NULL DEFAULT 'per_image',
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (dataset_id, name, revision_no)
);

CREATE INDEX idx_region_profile_revision_dataset
    ON region_profile_revision (dataset_id, name, revision_no);

-- Revisions are immutable configuration. Dataset deletion still cascades, but no edit
-- endpoint or maintenance statement may rewrite a row in place.
CREATE TRIGGER region_profile_revision_immutable
BEFORE UPDATE ON region_profile_revision
BEGIN
    SELECT RAISE(ABORT, 'region profile revisions are immutable');
END;

CREATE TABLE experiment (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT    NOT NULL,
    -- RESTRICT, not CASCADE: an experiment freezes its configuration at creation, so
    -- deleting the dataset, split or region profile it was computed against must not
    -- silently orphan it.
    dataset_id             INTEGER NOT NULL REFERENCES dataset (id) ON DELETE RESTRICT,
    split_id               INTEGER NOT NULL REFERENCES split (id) ON DELETE RESTRICT,
    region_profile_id      INTEGER NOT NULL
                                   REFERENCES region_profile_revision (id) ON DELETE RESTRICT,
    -- The digest of the prepared-region manifest the run was built on.
    region_manifest_sha256 TEXT    NOT NULL,
    -- Registry key from MODEL_REGISTRY (ADR-0007), e.g. 'pixel_reference'.
    model_type             TEXT    NOT NULL,
    model_config           TEXT    NOT NULL DEFAULT '{}',
    preprocessing_config   TEXT    NOT NULL DEFAULT '{}',
    eval_config            TEXT    NOT NULL DEFAULT '{}',
    status                 TEXT    NOT NULL DEFAULT 'draft'
                                   CHECK (status IN ('draft', 'training', 'trained', 'failed')),
    artifact_dir           TEXT    NOT NULL,
    created_at             TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    notes                  TEXT,
    -- The acquisition channels the run reads, as a JSON list of names; '[]' means every
    -- channel. Names rather than ids, so a frozen record stays legible in a log or a
    -- manifest.
    channels               TEXT    NOT NULL DEFAULT '[]',
    -- One of `Task` (ADR-0039), frozen at creation: it decides the training set, the
    -- evaluator and the result screens.
    task                   TEXT    NOT NULL DEFAULT 'anomaly',
    -- The one annotation class a targeted task segments (ADR-0040); NULL for a task that
    -- is about no single class.
    target_label           TEXT,
    -- The classes a supervised run learns, as a JSON list of annotation class keys in the
    -- dataset's taxonomy order at creation. `classes[i]` is label index `i + 1` in every
    -- target, label map and confusion matrix of the run; 0 is background. '[]' for every
    -- other task.
    classes                TEXT    NOT NULL DEFAULT '[]' CHECK (json_valid(classes))
);

CREATE INDEX idx_experiment_dataset ON experiment (dataset_id);
CREATE INDEX idx_experiment_split ON experiment (split_id);
CREATE INDEX idx_experiment_region_profile ON experiment (region_profile_id);

CREATE TABLE job (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- One of `JobKind`, validated in Python.
    kind          TEXT    NOT NULL,
    -- Nullable: most kinds belong to no experiment.
    experiment_id INTEGER REFERENCES experiment (id) ON DELETE CASCADE,
    status        TEXT    NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    progress      REAL    NOT NULL DEFAULT 0.0 CHECK (progress BETWEEN 0.0 AND 1.0),
    message       TEXT,
    log_path      TEXT,
    -- Per-kind payload as JSON. `experiment_id` identifies what a train/infer job acts
    -- on, but an import job needs its dataset, adapter and manifest path recorded too.
    params        TEXT    NOT NULL DEFAULT '{}',
    -- What the job produced, from its `done` event: a manifest path, a count of images
    -- rendered, a drift report. Input and output are kept apart so re-reading a finished
    -- job never has to guess which is which.
    result        TEXT    NOT NULL DEFAULT '{}',
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
    -- The map file under the experiment's maps/ dir; NULL when the model emits no map.
    map_path      TEXT,
    inference_ms  REAL    NOT NULL,
    -- Where the map's largest value sits, in source-frame pixels. A property of the map
    -- alone; NULL when there is no readable map.
    peak_x        INTEGER,
    peak_y        INTEGER,
    -- Whether the peak landed inside the annotated region, within the tolerance. A
    -- threshold-free qualifier of the verdict, not a fifth outcome (ADR-0028). NULL is
    -- "not applicable" — a normal image, a defect with no mask, an unreadable map — and
    -- never 0: "we did not check" is not "the model missed".
    localized     INTEGER CHECK (localized IS NULL OR localized IN (0, 1)),
    PRIMARY KEY (experiment_id, image_id)
);

CREATE INDEX idx_image_result_image ON image_result (image_id);
CREATE INDEX idx_image_result_score ON image_result (experiment_id, score);

CREATE TABLE sample_result (
    experiment_id INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    sample_id     INTEGER NOT NULL REFERENCES sample (id) ON DELETE CASCADE,
    agg_score     REAL    NOT NULL,
    -- One of `Aggregation` and one of `ChannelNormalization`, recorded per row so a
    -- stored result stays self-describing after a default changes (handbook evaluation.md).
    aggregation   TEXT    NOT NULL,
    normalization TEXT    NOT NULL,
    -- The same three-valued verdict as `image_result.localized`, resolved from the image
    -- that produced the aggregate score.
    localized     INTEGER CHECK (localized IS NULL OR localized IN (0, 1)),
    PRIMARY KEY (experiment_id, sample_id)
);

CREATE INDEX idx_sample_result_sample ON sample_result (sample_id);
CREATE INDEX idx_sample_result_score ON sample_result (experiment_id, agg_score);

CREATE TABLE metric_set (
    experiment_id       INTEGER NOT NULL REFERENCES experiment (id) ON DELETE CASCADE,
    subset              TEXT    NOT NULL CHECK (subset IN ('train', 'val', 'test')),
    -- Threshold-independent metrics only. Nothing that depends on a decision
    -- threshold is persisted; those are computed on demand (handbook evaluation.md).
    metrics             TEXT    NOT NULL DEFAULT '{}',
    computed_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- Which resolved labels and masks the metrics measured (ADR-0032); a different
    -- current digest renders the set stale until it is re-evaluated.
    ground_truth_digest TEXT    NOT NULL,
    PRIMARY KEY (experiment_id, subset)
);

CREATE TABLE annotation_label (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    key         TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    color       TEXT    NOT NULL,
    position    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (dataset_id, key)
);

CREATE INDEX idx_annotation_label_dataset
    ON annotation_label (dataset_id, position, id);

-- App-owned, versioned annotation truth (ADR-0032). Imported masks stay referenced
-- provenance; a completed revision materialises its own checksummed files.
CREATE TABLE annotation_revision (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL REFERENCES image (id) ON DELETE CASCADE,
    revision_no         INTEGER NOT NULL CHECK (revision_no > 0),
    document            TEXT    NOT NULL CHECK (json_valid(document)),
    document_sha256     TEXT    NOT NULL,
    -- The binary mask every anomaly consumer reads.
    mask_path           TEXT    NOT NULL,
    mask_sha256         TEXT    NOT NULL,
    source_mask_id      INTEGER,
    source_mask_path    TEXT,
    source_mask_sha256  TEXT,
    completed_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- The class-index mask and the class table pinned at completion — every class the
    -- dataset had, with its index and pixel count (ADR-0040). When NULL, a class's
    -- presence and region are read from the document.
    class_mask_path     TEXT,
    class_mask_sha256   TEXT,
    class_table         TEXT    CHECK (class_table IS NULL OR json_valid(class_table)),
    -- The object instances file: each instance's class, tight box and pixel count
    -- (ADR-0039). When NULL, boxes are read from the document.
    instances_path      TEXT,
    instances_sha256    TEXT,
    UNIQUE (image_id, revision_no)
);

CREATE INDEX idx_annotation_revision_image
    ON annotation_revision (image_id, revision_no DESC);

-- A revision may be removed by a dataset cascade, but it may never be rewritten.
CREATE TRIGGER annotation_revision_immutable
BEFORE UPDATE ON annotation_revision
BEGIN
    SELECT RAISE(ABORT, 'annotation revisions are immutable');
END;

-- An image draft exists only once it has been saved: opening the editor never writes.
CREATE TABLE annotation_draft (
    image_id            INTEGER PRIMARY KEY REFERENCES image (id) ON DELETE CASCADE,
    base_revision_id    INTEGER,
    document            TEXT    NOT NULL CHECK (json_valid(document)),
    version             INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    source_mask_id      INTEGER,
    source_mask_path    TEXT,
    source_mask_sha256  TEXT,
    updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- A sample-scoped draft is its own row rather than a draft stored against a nominated
-- "carrier" image. A document about to be written onto N images cannot carry one image's
-- source-mask provenance, nor name one `base_revision_id` when it is seeded from the
-- newest revision of each of its images. Truth stays image-keyed; only the editing scope
-- moves up.
CREATE TABLE annotation_sample_draft (
    sample_id   INTEGER PRIMARY KEY REFERENCES sample (id) ON DELETE CASCADE,
    document    TEXT    NOT NULL CHECK (json_valid(document)),
    version     INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
