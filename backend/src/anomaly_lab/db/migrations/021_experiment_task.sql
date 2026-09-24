-- An experiment records which task it is (ADR-0039).
--
-- Every run so far ranks images by how anomalous they are, so every existing row is an
-- `anomaly` run and the default says exactly that — no backfill beyond it. The value is one
-- of `Task` and is validated in Python, like `job.kind` since migration 020: the list of
-- tasks will grow, and a CHECK here would make each new one a table rebuild.
--
-- Frozen at creation, with the split and the region profile: the task decides the training
-- set, the evaluator and the result screens, so changing it would make every stored result
-- describe a different run.

ALTER TABLE experiment ADD COLUMN task TEXT NOT NULL DEFAULT 'anomaly';
