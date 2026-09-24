-- A supervised segmentation run pins the classes it segments (ADR-0039).
--
-- A JSON list of annotation class keys, in the dataset's taxonomy order at creation. Class
-- `classes[i]` is label index `i + 1` in every target the run fits on, every label map it
-- writes and every confusion matrix it is read by; 0 is background. Pinned rather than read
-- from the taxonomy at train time because a class added or reordered between training and
-- evaluation would renumber the labels under a stored model. '[]' for every other task and
-- for every existing row: an anomaly run and a few-shot run are about no class list.

ALTER TABLE experiment ADD COLUMN classes TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(classes));
