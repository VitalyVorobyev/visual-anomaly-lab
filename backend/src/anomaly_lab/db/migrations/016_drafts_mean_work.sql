-- A draft records work, not a page view.
--
-- Until now `POST .../annotations/draft` was an upsert called from the editor's *read* path,
-- so merely opening an image persisted a row -- and completing one resurrected it, because the
-- completion invalidated the query while its observer was still mounted. Neither left anything
-- a re-open would not reproduce, but `count_open_image_drafts` counts rows, so every image ever
-- looked at became a permanent blocker on `annotation_scope` (ADR-0036). Creation now happens
-- on the first save.
--
-- This deletes the litter that write path left behind. `version` defaults to 1 and is
-- incremented only by a save, so under the *old* semantics `version = 1` is exactly "created by
-- opening, never saved". Nothing in such a row is lost: `base_revision_id` is written and read
-- nowhere in the backend, and `source_mask_*` restate the `mask` row plus a digest that
-- `render_binary_mask` recomputes at completion anyway.
--
-- **This predicate is valid exactly once and can never become a runtime rule.** After this
-- migration a draft is created by the first save, so `version = 1` means "saved once" -- the
-- opposite -- and litter is indistinguishable from real work by version forever afterwards.
-- That is why the fix is a migration plus a corrected write path, and not a WHERE clause on
-- the count.

DELETE FROM annotation_draft WHERE version = 1;

DELETE FROM annotation_sample_draft WHERE version = 1;
