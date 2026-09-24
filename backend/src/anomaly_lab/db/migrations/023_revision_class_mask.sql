-- A completed revision also materialises a class-index mask and pins its class table
-- (ADR-0040).
--
-- The binary mask stays what every anomaly consumer reads. The class mask holds, per pixel,
-- the index of the class drawn there, and `class_table` is the JSON list of every class the
-- dataset had at completion with its index and pixel count. A few-shot task reads a class's
-- presence from the table and its region from the mask. Revisions completed before this
-- migration keep null in all three: their presence is read from the document, and nothing
-- is re-rendered behind the annotator's back, since a revision is immutable.

ALTER TABLE annotation_revision ADD COLUMN class_mask_path TEXT;
ALTER TABLE annotation_revision ADD COLUMN class_mask_sha256 TEXT;
ALTER TABLE annotation_revision ADD COLUMN class_table TEXT CHECK (class_table IS NULL OR json_valid(class_table));
