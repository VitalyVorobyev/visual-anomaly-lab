# Evaluation layer

The evaluation layer is **model-independent by construction** (ADR-0011). Its only inputs are:

- `ImageResult.score` rows for an experiment,
- `Sample.label`,
- `SplitAssignment.subset`,
- the resolved ground-truth mask for each image, and the float32 map on disk, when pixel or localization
  metrics are computed.

It never imports a model module and never re-runs inference. Every method is therefore evaluated by exactly
the same code, which is the precondition for the comparison view to mean anything.

**Which code evaluates a run is chosen by its task** (ADR-0039), from the table in
`eval/evaluators.py`. Everything on this page is the `anomaly` evaluator, which is `eval/runner.py`
behind that seam and unchanged by it. The `infer` job and re-evaluation both call
`evaluator_for(experiment.task)`, and a task with no entry cannot be created, so no run reaches the
end of inference with nothing to read it.

## Channel → sample aggregation

A part is scored from its per-image scores. The **default aggregation is `max`**: a defect visible under any
single illumination makes the part defective, and `mean` dilutes single-channel evidence by averaging a strong
signal with two uninformative views — precisely the failure mode this dataset invites. `mean` is available as
an option, and the method used is recorded in both `SampleResult.aggregation` and `Experiment.eval_config`, so
a stored result always says how it was produced.

**`max` assumes per-channel scores are comparable in scale, and that is now a configured step rather than a
footnote.** The assumption holds for a method whose scores are per-channel z-scores by construction (handbook methods.md).
It is not automatic for a deep model: if one illumination's score distribution simply sits higher than
another's, every maximum comes from that channel and the sample score measures which view the method finds
noisiest, not which part is defective.

`EvalConfig.channel_normalization` puts them on one scale first. It defaults to `none`, so no stored
experiment changes meaning, and the choice is recorded on `SampleResult.normalization` beside the aggregation
— a sample-level number is uninterpretable without both halves.

- **`robust_z`** centres each channel on its own median and divides by its scaled MAD. Robust to the outliers
  that are the signal.
- **`rank`** is the rank fraction within the channel. Scale-free, and blunter than it looks: it keeps only the
  ordering, so a part that is dramatically the most anomalous thing one channel ever saw and a part that
  merely tops a flat channel both become `1.0`. Under `max` they then tie.

**The transform is fitted over every image the experiment has scored, with labels ignored.** Two facts force
this rather than making it a preference. `sample_result` is keyed `(experiment_id, sample_id)` with no subset
column, so a per-subset fit is not representable in the storage shape. And using labels to build the transform
would make the metric partly a function of the answer. It is transductive — a test image's score depends on
the other test images — which is the honest cost, and the same one the pixel metrics already accept when they
adapt their histogram bins to a run's observed range.

**Image-level ROC-AUC stays on raw scores.** ADR-0011 keeps it as the number that isolates model quality from
how the channels were combined, and normalization is part of combining them.

## Did it find the defect, or merely score it?

An image-level verdict is `agg_score >= threshold` against the sample's label, and it is blind to *where* the
evidence came from. A defective part whose anomaly map fires on the background behind it counts as a true
positive; a hundred of them read as a working method until somebody opens one of the pictures. Pixel ROC-AUC
and AU-PRO already measure agreement over the frame, but they are a subset-wide summary — there is no
per-sample answer to "was **this** hit on the part".

`localized` is that answer: **does the map's peak fall inside the annotated region, within a tolerance?**
It is deliberately *not* a fifth outcome. `outcome_of` names four buckets the comparison layer and the
frontend both key on (ADR-0028), and widening that vocabulary would change what every existing `tp` means.
A true positive that is `localized = false` is still a true positive — one that got the right answer from
the wrong pixels.

**It is the map's peak, not the score's location.** The stored map has been blurred, upsampled from the patch
grid and projected back into source coordinates, so its argmax sits near the grid cell that produced the
score without being that cell — and a method scoring at a percentile below 100 (`dino_memory`) is not reading
a single cell at all. The verdict is therefore a statement about the picture the reviewer is looking at,
which is the only thing that can be checked against a region drawn by hand.

**The tolerance is a fraction of the image diagonal**, `EvalConfig.localization_tolerance`, default `0.02`,
resolved to a pixel radius per image and reported beside the counts. Not a fixed pixel count: a map's real
resolution is its patch stride — 14 or 16 source pixels for a DINO-family encoder before any upsampling — so
"within 8 px" means *inside the same patch* on a small frame and four patches away on a large one. The radius
is floored at 1, so a tolerance of exactly zero is still a 3x3 window; a radius of 0 would decide the verdict
on a single pixel of a bilinearly resampled map. The test itself is a Chebyshev window clamped at the frame's
edges — a square, because the difference from a disc is the corners of a box that is already a tolerance,
and a negative slice start would silently select from the far side of the image.

**Why this one is persisted while the confusion counts are not.** ADR-0011's line is threshold-dependence,
not storage. Nothing here moves with the slider: the peak is a property of the map alone, and the verdict
compares that peak against ground truth. Recomputing it per tick would mean one `.npy` and one mask PNG per
image on every drag. Four nullable columns arrived in migration 019 — `image_result.peak_x`, `peak_y`,
`localized`, and `sample_result.localized` — each three-valued: `1` hit, `0` miss, `NULL` **not applicable**,
which is a normal image, a defect with no resolved mask, or a map that could not be read. `NULL` is never a
miss. Staleness needs no new machinery: the verdicts are refreshed by the same re-evaluation that refreshes
the pixel metrics, and go stale with the same `metric_set.ground_truth_digest`.

Peaks are recorded as maps are written — `InferContext.write_map` already scans the projected array for its
display extremes and takes the argmax in the same pass, so **no plugin changes and no plugin knows this
exists**. A run scored before migration 019 has none; re-evaluating backfills them from the maps already on
disk, once, and never re-reads a row that has one.

**A part's verdict follows the image that produced its score.** Under `max` that is the argmax after channel
normalization — the winning channel — so changing `channel_normalization` moves the winner and the verdict
with it. An "any channel hit" rule would report a hit for a part whose score came from a channel that fired
on the background while a discarded channel happened to find the defect, which is *exactly* the failure this
records. Under `mean` no single image produced the number, so there is no winner and any hit counts. Under
`feature_concat`-style methods one map is computed per sample and written once per image, so every channel
carries the same peak and the argmax picks between identical verdicts — arbitrary, and harmless for that
reason.

Per subset, `metrics["localization"]` carries the tolerance and both populations:

```json
{
  "tolerance_fraction": 0.02,
  "tolerance_pixels": 23,
  "defect_images_with_truth": 40,
  "peak_on_target_images": 33,
  "defect_samples_with_truth": 40,
  "localized_samples": 33,
  "unannotated_defect_samples": 2
}
```

`tolerance_pixels` is `null` when the subset's scored images differ in size, because there is then a
different radius per image and a single figure would name none of them. `*_with_truth` counts what was
actually tested — a resolved mask *and* a readable map — so it is the denominator its numerator belongs to.
**The whole block is absent when nothing was checked**, the way `pixel` is absent for a dataset with no
masks: `0 of 0` would read as "this method never localized anything", which is a claim about the method made
out of the absence of ground truth.

## Which channels a run reads

`Experiment.channels` is a frozen list of channel **names**; empty means every channel. It is applied in
`list_images_for_split`, which is the one place the "which images" question is answered, so training,
inference and the on-demand diagnostic path all narrow identically — asking a bright-field-only run about a
dark-field image is a 404 that names the channel rather than a scored answer to a question the run cannot
have.

This is what makes one multi-channel dataset strictly better than one dataset per illumination. The split,
the labels and the region build are shared underneath every such run, so a difference between "bright only"
and "all three" is the channel and not two separate imports — and because splits are sample-level, no part's
views can straddle train and test in either.

An image whose `channel_id` is `NULL` is **excluded by a non-empty selection**: it belongs to no named
channel, and "unassigned" is not a synonym for "all of them". A single-view dataset has nothing to select and
is only ever read with no filter.

## Metrics

**Headline metric: sample-level ROC-AUC.** It is threshold-free, robust to the class imbalance of the
reference data, and evaluated on the unit that matters — the physical part. Image-level ROC-AUC is reported
alongside it as a diagnostic (it reveals when a model scores individual views well but aggregation is losing
that signal).

Persisted in `MetricSet`: **threshold-independent metrics only** — sample-level and image-level ROC-AUC,
average precision, per-subset sample counts, timing summaries, the peak-on-target counts, and the digest of
the labels and masks those numbers measured.

### Ground-truth snapshot and freshness

Metrics, image overlays and mask-presence reads share one resolver. For an image it selects the newest
completed app-owned annotation revision, otherwise the imported source mask, otherwise no mask. Evaluation
hash-verifies the selected bytes before using them. A changed pinned file is a named failure, not a new truth
silently measured under the old experiment.

Ground truth is now resolved **once per run**, not once per subset: `resolve_ground_truth_masks` is keyed by
image id, so a superset dictionary is exactly as correct per subset, and the byte verification happens one
time instead of once per subset.

Every subset's `ground_truth_digest` covers its current sample labels and each resolved mask's kind, row id
and SHA256. Experiment detail recomputes that metadata digest without opening image files. A mismatch—or a
legacy `NULL` digest—marks the metric set stale. The stored localization verdicts go stale by exactly the
same signal, because they were computed against exactly the same resolved masks. The UI keeps old values
visible with a warning, hides charts that would otherwise combine current labels with old areas, and offers
reevaluation from stored scores. The
comparison view carries the same signal per run and will not draw mixed-snapshot curves.

**Threshold-dependent outputs are computed on demand** from the persisted scores: confusion matrix,
precision / recall / F1, and the FP/FN sample lists.
`GET /api/experiments/{experiment_id}/threshold?value=…` returns
them for any threshold, computed in milliseconds from a few hundred stored floats. Persisting metrics per
threshold would be storing a derived function of data already in the database — and would make the UI's
threshold slider feel like a database write instead of an instant filter.

### Pixel-level metrics

**Pixel ROC-AUC and AU-PRO** are computed for every subset that has masks, and simply do not appear for the
datasets that have none — which is most of them, including the showcase tree. They needed no schema change
and no re-inference, exactly as the `Mask` table and the float32 `.npy` maps were meant to allow
([the domain model](domain-model.md), [methods](methods.md)).

**Nothing is accumulated.** A hundred test images at 1.5 MPix in float32 is ~600 MB if the maps are held to
compute a curve, and the comparison view multiplies that by the number of runs being compared. Each image is
instead folded into three fixed-bin arrays and discarded: a positive-pixel score histogram, a negative-pixel
one, and an accumulator holding, per bin, the sum over ground-truth regions of *that region's fraction* of
pixels in the bin. Memory is constant in the number of test images and in their resolution.

The region accumulator is **exact, not an approximation**. PRO at a threshold is the mean over regions of each
region's covered fraction; dividing each region's per-bin counts by its own size *before* summing is what
collapses "one histogram per region" into one array while computing the same quantity. AU-PRO integrates to a
false-positive rate of **0.3**, the convention every paper reports, with the endpoint interpolated so the
limit is exactly 0.3 rather than the nearest bin.

Three protocol choices move the number, and are decisions rather than details:

- **Normal images are included, with an all-zero mask.** This is the MVTec and VisA convention and it is what
  makes the reported figure comparable to a published one. Evaluating over anomalous images alone — the
  tempting reading — draws every negative pixel from inside a defective image and produces a false-positive
  rate no published number is comparable to.
- **A *defective* image with no mask is skipped and counted.** Its defect pixels are somewhere, and assuming
  they are nowhere would reward a model for missing it.
- **The map is upsampled to the mask's resolution**, bilinearly — never the mask downsampled to the map's,
  which invents labels. Only one image is in flight, so the cost is bounded.

Bins adapt to the observed score range in one cheap pass over the map files, because a fixed `[0, 1]`
assumption is wrong for every method that does not normalize its output. Scores are binned at 2¹⁶ levels, so
the number is not bit-identical to a direct computation; the error is far below the noise of a 100-image test
set, and a comparison against a paper's figure inherits it.

**Everything skipped is reported** in the metric set and stated in words on the results screen. A pixel metric
computed over half the defects is not the metric it claims to be, and a reader must not have to dig for that.

One known performance weakness remains: AU-PRO's connected-component labelling is Python-level union-find,
which is fine for the sparse masks these datasets have and would be the slowest part of evaluation on masks
that cover most of the frame. Source masks imported before migration 005 can still begin unhashed, but the
first evaluation or annotation-base read pins them; `verify` reports digest coverage separately until then.

## Rankings

Most-normal and most-anomalous lists are a sort on `SampleResult.agg_score` — no separate computation.
**Unlabeled samples are included in rankings but excluded from metrics.** Ranking a model's most-anomalous
unlabeled samples is one of the most useful things this workbench does: it turns the model into a labeling
aid for the ~113 unlabeled samples, while never letting unlabeled data contaminate a reported number.

## Timing

Per-sample inference time is aggregated from `ImageResult.inference_ms` (mean, median, p95, total), reported
per experiment and compared across methods. Since methods differ by orders of magnitude in cost — seconds on
CPU for the classical baseline versus GPU-bound deep inference — the accuracy/latency trade-off is a first
class part of the comparison, not a footnote.

## Splits, and what a missing subset means

Anomaly detection trains on normals only, so a split must reserve enough normals for fitting while keeping both
classes available for threshold selection and final reporting. Splits are assigned at **sample** level, so a
part's channels never straddle subsets ([the domain model](domain-model.md)), and there are two ways to obtain one.

**Drawn here** (`normal_only_train`): seeded and stratified by capture group, so an acquisition-batch effect
cannot land entirely on one side. The fractions are configurable; a reasonable starting shape is a train subset
of normals only, a validation subset of held-out normals plus some defects for threshold selection, and the
remainder as test. What is right depends entirely on how many samples exist, which is why it is a form field
and not a constant in the code.

**Adopted from the source** (`imported`, handbook import.md): the partition the benchmark published, read out of the
manifest the dataset was committed from. A number computed on a partition we drew ourselves is not comparable
to a paper's number, so for any dataset that ships a split table this is the strategy that makes the
comparison mean anything.

**A missing `val` subset is normal, not an error.** VisA's official one-class protocol has train and test and
nothing else, and that is the protocol its published figures are computed under. Every layer therefore has to
tolerate an empty subset rather than assume three:

- the split machinery leaves samples the manifest does not place *out*, rather than sweeping them into `test`;
- the training handler passes an empty validation sequence, and logs that it did;
- a method that calibrates on held-out normals — EfficientAD fits its score-normalization quantiles on them —
  **falls back visibly**, with a warning naming what it used instead and what that costs;
- threshold selection has nothing to fit against, so it returns the highest normal score in the subset **and
  the sentence explaining that choice**, which the results screen prints under the slider. A slider that opens
  at a fabricated position with no explanation invites the operator to read it as a recommendation.

## Auditing a run from what it already wrote

`scripts/audit-run.py <experiment_id>` reads a scored run's stored maps and rows and reports
two things the results screen does not: **one CSV row per image** — label, subset, score,
inference time and the paths to its map and ground-truth mask, which is what makes a reported
figure checkable by somebody who does not trust it — and the **image-level ROC-AUC under every
map aggregation**, recomputed from the maps on disk.

Read-only, numpy only, no torch and no GPU, so it can be pointed at a run from months ago at
no cost. That property is the point: it made a wrong conclusion cheap to overturn. The
aggregation sweep first said the reducer did not matter, which was true of a run whose maps
were not localized and false in general — see
[the measurements](../measurements.md).

Maps from a region-prepared experiment are stored in source coordinates. Pixels outside the selected
source crop are `NaN`, not fabricated observations. Renderers make those pixels transparent. Evaluation keeps
them in the source-frame denominator at the run's score floor, so a localisation that misses a defect is
penalised rather than rewarded by hiding it. It reports `covered_pixel_fraction`,
`uncovered_defect_pixels`, and `uncovered_normal_pixels` alongside the evaluated counts.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
