-- How a sample's per-channel scores were put on one scale before they were reduced.
--
-- ADR-0011 chose `max` and recorded its own caveat: `max` assumes per-channel scores are
-- comparable, which is not automatic for a deep model whose bright-field distribution simply
-- sits higher than its dark-field one. On a single-channel dataset that caveat is inert. On a
-- grouped one it decides every sample score, so the normalization is now a configured step
-- and, like `aggregation` beside it, is recorded per row rather than only in `eval_config`.
-- A stored result stays self-describing after the default changes.
--
-- Nullable, so every existing row means `none` truthfully and nothing has to be backfilled —
-- the same reasoning migration 012 used. A separate column rather than widening the
-- `aggregation` CHECK to hold pairs: that would be combinatorial, and widening a CHECK in
-- SQLite means the whole table-rebuild dance of migration 011 for no gain.

ALTER TABLE sample_result ADD COLUMN normalization TEXT;
