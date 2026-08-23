-- A dataset can say which channel it is normally read in.
--
-- Everywhere the app needs one photograph to stand for a whole part — the annotation queue,
-- a grid tile, the sample viewer's opening tab — it takes the sample's first image, and
-- "first" means the lowest `channel.position`, which is the order the channel folders
-- happened to be scanned in at import. That is an accident of the source tree, not a
-- statement about which illumination the part is judged under, and on a dataset whose useful
-- view is the third one it costs two keystrokes on every sample, forever.
--
-- A name rather than a channel id, for the reason `013_experiment_channels.sql` gives about
-- `experiment.channels`: `upsert_channel` matches on (dataset_id, name), so a name survives a
-- re-import that renumbers the dictionary. NULL means "not overridden" and resolves to the
-- first channel, so this migration needs no backfill and a name that a later import renames
-- away degrades to the old behaviour instead of to nothing.

ALTER TABLE dataset ADD COLUMN default_channel TEXT;
