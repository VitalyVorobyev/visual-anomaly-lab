-- Datasets can say what they belong to.
--
-- VisA registers as twelve ordinary datasets, one per object class, and the catalogue had
-- no way to say they are one benchmark: `registered_dataset_id` recomputes pack membership
-- on every read by matching (name, adapter, resolved root_path) against hard-coded specs,
-- and stores nothing. That is enough to *describe* the group and not enough to represent
-- one, so the twelve sat beside every user dataset in a flat grid.
--
-- Nullable and free-text rather than a `collection` table with a foreign key: a collection
-- carries no attributes of its own yet, so a row per name would be a join that answers no
-- question. NULL means "not overridden", and the reference pack's title is derived for
-- datasets that came from one — so this migration needs no backfill, and a user who types
-- a collection over a reference dataset wins.

ALTER TABLE dataset ADD COLUMN collection TEXT;
