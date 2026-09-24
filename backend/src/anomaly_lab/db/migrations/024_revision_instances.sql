-- A completed revision also records its object instances (ADR-0039).
--
-- Completion writes `revision-<n>.instances.json` beside the two masks: every instance --
-- the add shapes that share an `instance_id`, or one shape alone -- with its class, the tight
-- bounding box of the pixels it still owns and their count. Detection and instance
-- segmentation read their truth from it. Revisions completed before this migration keep null
-- in both columns: nothing is re-rendered behind the annotator's back, since a revision is
-- immutable.

ALTER TABLE annotation_revision ADD COLUMN instances_path TEXT;
ALTER TABLE annotation_revision ADD COLUMN instances_sha256 TEXT;
