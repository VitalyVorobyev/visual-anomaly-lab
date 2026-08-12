-- M10 makes the spatial input an explicit, immutable part of every experiment.
--
-- This is an intentionally destructive early-stage migration. Experiments created
-- before region profiles existed cannot truthfully name the prepared pixels they saw,
-- so retaining them with a nullable or invented profile would make the scientific
-- record look more complete than it is. Their dependent rows cascade away.
DELETE FROM experiment;

ALTER TABLE experiment
    ADD COLUMN region_profile_id INTEGER NOT NULL
        REFERENCES region_profile_revision (id) ON DELETE RESTRICT;

ALTER TABLE experiment
    ADD COLUMN region_manifest_sha256 TEXT NOT NULL;

CREATE INDEX idx_experiment_region_profile ON experiment (region_profile_id);
