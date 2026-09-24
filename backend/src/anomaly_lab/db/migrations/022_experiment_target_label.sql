-- A targeted task names the one annotation class it segments (ADR-0040).
--
-- Frozen at creation with the task: a few-shot run's references, truth and metrics are all
-- about this class, so changing it would make every stored result describe a different run.
-- Null for `anomaly`, which ranks images and is about no class, and for every existing row.
-- It is checked in Python against the dataset's classes at creation, like the task: a class
-- is keyed by (dataset, key), and a key never changes once created.

ALTER TABLE experiment ADD COLUMN target_label TEXT;
