-- Annotate the part, not each photograph of it.
--
-- ADR-0005 makes `Sample` the unit of identity and `Channel` data, so one physical part
-- photographed under three illuminations is one sample owning three images. Annotation
-- truth (ADR-0032) is image-keyed all the way down -- `annotation_draft.image_id` is a
-- primary key, `annotation_revision` is unique on `(image_id, revision_no)`, and the one
-- resolver every consumer reads is keyed by image. A defect that is visible in dark field
-- and invisible in bright field therefore had to be traced once per illumination, by hand,
-- on frames the import probe measures as registered to within 0 px.
--
-- The scope is per dataset because it is a property of the *data*, not a preference: a
-- corpus whose channels are three exposures of one part shares its truth, and a corpus
-- whose "channels" are unrelated views does not. It defaults to `image`, so every existing
-- dataset keeps exactly today's behaviour with no backfill.
--
-- SQLite enforces an inline CHECK added with ADD COLUMN, so the vocabulary is closed here
-- rather than in Python alone.

ALTER TABLE dataset ADD COLUMN annotation_scope TEXT NOT NULL DEFAULT 'image'
    CHECK (annotation_scope IN ('image', 'sample'));

-- A sample-scoped draft is its own row rather than a draft stored against a nominated
-- "carrier" image, because the two are not the same shape. `annotation_draft` carries the
-- pinned provenance of one image's imported source mask (`source_mask_id`, `_path`,
-- `_sha256`); a document that is about to be written onto N images cannot carry image A's
-- provenance, so those three columns would have to be permanently NULL and the ETag
-- namespace would silently overlap with a real image draft's.
--
-- There is deliberately no `base_revision_id` either: a sample draft is seeded from the
-- newest revision of *each* of its images, and a single id would have to name one of them
-- arbitrarily. Truth stays image-keyed; only the editing scope moves up.

CREATE TABLE annotation_sample_draft (
    sample_id   INTEGER PRIMARY KEY REFERENCES sample (id) ON DELETE CASCADE,
    document    TEXT    NOT NULL CHECK (json_valid(document)),
    version     INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
