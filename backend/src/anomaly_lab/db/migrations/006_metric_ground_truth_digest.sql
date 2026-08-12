-- Which resolved labels and masks a stored metric set actually measured (ADR-0032).
-- Existing metric rows are deliberately NULL: their truth snapshot was never recorded,
-- so they must render stale until reevaluated.
ALTER TABLE metric_set ADD COLUMN ground_truth_digest TEXT;
