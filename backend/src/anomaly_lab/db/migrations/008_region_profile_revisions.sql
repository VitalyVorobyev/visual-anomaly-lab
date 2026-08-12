-- Dataset-owned immutable input-region configurations (ADR-0033).
-- Build products and operational status arrive separately: this row is only the frozen
-- configuration a future experiment can pin without inheriting a mutable "current" value.
CREATE TABLE region_profile_revision (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id        INTEGER NOT NULL REFERENCES dataset (id) ON DELETE CASCADE,
    name              TEXT    NOT NULL,
    revision_no       INTEGER NOT NULL CHECK (revision_no >= 1),
    extractor_type    TEXT    NOT NULL,
    extractor_config  TEXT    NOT NULL DEFAULT '{}'
                                  CHECK (json_valid(extractor_config)),
    prepared_width    INTEGER NOT NULL CHECK (prepared_width > 0),
    prepared_height   INTEGER NOT NULL CHECK (prepared_height > 0),
    padding_fraction  REAL    NOT NULL DEFAULT 0.05
                                  CHECK (padding_fraction BETWEEN 0.0 AND 1.0),
    failure_policy    TEXT    NOT NULL DEFAULT 'fail'
                                  CHECK (failure_policy = 'fail'),
    seed              INTEGER NOT NULL,
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
