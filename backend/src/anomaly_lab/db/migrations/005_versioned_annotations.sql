-- App-owned, editable annotation truth. Source masks remain referenced provenance;
-- completed revisions materialise their own checksummed binary masks.
ALTER TABLE mask ADD COLUMN sha256 TEXT;

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

-- Existing datasets get the same initial taxonomy as datasets created after this
-- migration. Reads therefore remain reads.
INSERT INTO annotation_label (dataset_id, key, name, color, position)
SELECT id, 'defect', 'Defect', '#ef4444', 0 FROM dataset;

CREATE TABLE annotation_revision (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL REFERENCES image (id) ON DELETE CASCADE,
    revision_no         INTEGER NOT NULL CHECK (revision_no > 0),
    document            TEXT    NOT NULL CHECK (json_valid(document)),
    document_sha256     TEXT    NOT NULL,
    mask_path           TEXT    NOT NULL,
    mask_sha256         TEXT    NOT NULL,
    source_mask_id      INTEGER,
    source_mask_path    TEXT,
    source_mask_sha256  TEXT,
    completed_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
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
