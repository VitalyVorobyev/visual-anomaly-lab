-- Write the teacher an experiment actually used into its own record.
--
-- `efficientad_custom.teacher_source` arrived after these experiments ran, so their stored
-- configuration does not mention it — and an absent field takes the *current* default. The
-- default is about to change from 'anomalib' to 'nelson1425', which would silently rewrite
-- history: every run measured before the field existed would start describing itself as
-- having used a teacher it never saw, on the comparison screen and in `fit_more`.
--
-- An experiment's configuration is the record of what was run. Backfilling the explicit
-- value is what makes that record independent of what the default becomes later, which is
-- the property it was supposed to have all along.
--
-- Scoped to `efficientad_custom`: no other method has this field, and adding it to a
-- configuration whose schema rejects it would make those experiments unreadable.

UPDATE experiment
SET model_config = json_set(model_config, '$.teacher_source', 'anomalib')
WHERE model_type = 'efficientad_custom'
  AND json_extract(model_config, '$.teacher_source') IS NULL;
