-- A region profile says whether the images of one sample share one crop.
--
-- `per_image` is what every revision before this migration did: the extractor's box for
-- each image, independently. `union` gives every image of a sample the union of their
-- boxes, so the channels of one part stay registered. Existing revisions are immutable and
-- were built per image, so that is their value. The vocabulary lives in Python
-- (`SampleAlignment`), as migration 020 decided for extensible lists.

ALTER TABLE region_profile_revision
    ADD COLUMN sample_alignment TEXT NOT NULL DEFAULT 'per_image';
