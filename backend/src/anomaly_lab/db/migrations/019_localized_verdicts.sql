-- Did the map's peak land on the defect, or somewhere else entirely?
--
-- An image-level verdict is `agg_score >= threshold` against the sample's label, which is
-- blind to *where* the evidence came from: a defective part whose anomaly map fires on the
-- conveyor behind it is counted a true positive, and a hundred of them read as a working
-- method. Pixel ROC-AUC and AU-PRO already measure agreement over the whole frame, but they
-- are a subset-wide summary — there is no per-image answer to "was this particular hit on
-- the part or beside it", which is the question a reviewer asks with one sample open.
--
-- `localized` is that answer, and it is deliberately **not** a fifth outcome value.
-- `outcome_of` names four buckets that the comparison layer and the frontend both key on
-- (ADR-0028); widening that vocabulary would change the meaning of every existing tp. This
-- is an orthogonal qualifier: a true positive that is `localized = 0` is still a true
-- positive, and now says how it earned the label.
--
-- Persisted rather than recomputed on demand, which is the opposite of what evaluation does
-- for the confusion matrix — and consistent with it. That record persists what is
-- threshold-*independent* and recomputes what moves with the slider. This verdict never
-- moves with the slider: it compares the map's own peak against the annotated region and
-- would be identical at every threshold, so recomputing it per tick would mean reading a
-- `.npy` and a mask PNG per image on every drag of the slider. It goes stale the same way
-- pixel metrics do, through `metric_set.ground_truth_digest`, and re-evaluation is the one
-- path that refreshes both.
--
-- Nullable, with three columns rather than one, because they answer different questions and
-- go stale at different times:
--
--   * `peak_x` / `peak_y` are a property of the **map alone** — where its largest value sits,
--     in source-frame pixels. Editing an annotation cannot change them, so they are computed
--     once and backfilled for a legacy run on its next evaluation.
--   * `localized` is a property of (map, ground truth): 1 the peak is inside the annotated
--     region within the tolerance, 0 it is outside, NULL not applicable — a normal image, a
--     defect with no mask, or an image whose map could not be read. NULL is the honest answer
--     and never 0: "we did not check" is not "the model missed".
--
-- `sample_result.localized` carries the same three values for the part, resolved from the
-- image that actually produced the aggregate score.
--
-- Every existing row is NULL in all four columns and therefore means "not checked", which is
-- true: no run before this migration recorded a peak. No backfill, for the reason migrations
-- 012, 014 and 018 give — a re-evaluation is the backfill path, and it is cheap.

ALTER TABLE image_result ADD COLUMN peak_x INTEGER;
ALTER TABLE image_result ADD COLUMN peak_y INTEGER;
ALTER TABLE image_result ADD COLUMN localized INTEGER
    CHECK (localized IS NULL OR localized IN (0, 1));

ALTER TABLE sample_result ADD COLUMN localized INTEGER
    CHECK (localized IS NULL OR localized IN (0, 1));
