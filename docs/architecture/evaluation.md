# Evaluation

The evaluation layer is **model-independent by construction** (ADR-0011). Its only inputs are:

- `ImageResult.score` rows for an experiment,
- `Sample.label`,
- `SplitAssignment.subset`,
- the resolved ground-truth mask for each image and the float32 map on disk, when pixel or localization
  metrics are computed.

It never imports a model module and never re-runs inference, so every method is evaluated by the same code.

**The evaluator is chosen by task** (ADR-0039) from the table in `eval/evaluators.py`. Most of this page
describes the `anomaly` evaluator, `eval/runner.py`; [few-shot segmentation](#few-shot-segmentation) has
its own section. The `infer` job, re-evaluation and the staleness check all go through
`evaluator_for(experiment.task)`, and a task with no entry cannot be created. Each evaluator names a
`headline` metric (`sample_roc_auc`, `foreground_iou`), which the `infer` log prints per subset.

## Channel selection

`Experiment.channels` is a frozen list of channel **names**; empty means every channel. It is applied in
`list_images_for_split`, the one place that answers "which images", so training, inference and on-demand
diagnostics narrow identically — asking a bright-field-only run about a dark-field image is a 404 naming the
channel.

Split, labels and region build are shared by every run over one multi-channel dataset, so a difference
between "one channel" and "all channels" is the channel, not two imports. An image whose `channel_id` is
`NULL` is **excluded by a non-empty selection**; a single-view dataset is only read unfiltered.

## Channel → sample aggregation

A part is scored from its per-image scores. The **default aggregation is `max`**: a defect visible under any
single illumination makes the part defective, while `mean` dilutes single-channel evidence with
uninformative views. `mean` is an option. The method is recorded in `SampleResult.aggregation` and
`Experiment.eval_config`.

**`max` assumes per-channel scores share a scale.** If one illumination's scores simply sit higher, every
maximum comes from that channel and the sample score measures which view the method finds noisiest.
`EvalConfig.channel_normalization` puts channels on one scale first. It defaults to `none`, and the choice
is recorded on `SampleResult.normalization` beside the aggregation.

- **`robust_z`** — centre on each channel's median, divide by its scaled MAD; robust to the outliers that
  are the signal.
- **`rank`** — rank fraction within the channel; scale-free but blunt: every channel's top part becomes
  `1.0`, and under `max` they tie.

**The transform is fitted over every image the experiment scored, labels ignored.** `sample_result` is keyed
`(experiment_id, sample_id)` with no subset column, so a per-subset fit is not representable, and using
labels would make the metric a function of the answer. It is transductive, the same cost the pixel metrics
accept by adapting their bins to a run's range.

**Image-level ROC-AUC stays on raw scores** (ADR-0011): it isolates model quality from how channels were
combined.

## Metrics

**Headline: sample-level ROC-AUC** — threshold-free, robust to class imbalance, and on the unit that
matters, the physical part. Image-level ROC-AUC is reported beside it as a diagnostic of aggregation losing
signal.

`MetricSet` persists **threshold-independent metrics only**: sample- and image-level ROC-AUC, average
precision, per-subset sample counts, timing summaries, localization counts, pixel metrics, and the digest of
the labels and masks they measured. A metric that cannot be computed is `None` and renders as a dash.

**Threshold-dependent outputs are computed on demand** from persisted scores: confusion matrix,
precision / recall / F1 and the FP/FN sample lists, via
`GET /api/experiments/{experiment_id}/threshold?value=…`. Persisting them per threshold would store a
derived function of data already in the database and make the slider a write. Across runs, thresholds are
resolved per run by one shared rule (ADR-0028); `outcome_of` (`eval/threshold.py`) names the buckets
(`tp`, `fp`, `tn`, `fn`, and `unlabeled`) the comparison layer and frontend key on.

### Ground-truth snapshot and freshness

Metrics, overlays and mask-presence reads share one resolver: the newest completed annotation revision,
else the imported source mask, else no mask ([annotations](annotations.md)). Evaluation hash-verifies the
selected bytes; a changed pinned file is a named failure. `resolve_ground_truth_masks` runs **once per run**,
keyed by image id. Source masks without a digest are pinned on their first evaluation or annotation-base
read; `verify` reports digest coverage separately.

Every subset's `ground_truth_digest` covers its sample labels and each resolved mask's kind, row id and
SHA-256. Experiment detail recomputes it without opening image files; a mismatch or a `NULL` digest marks
the metric set stale. The UI keeps stale values visible with a warning, hides charts that would mix current
labels with old areas, and offers re-evaluation from stored scores. The comparison view carries the same
signal per run and will not draw mixed-snapshot curves.

### Pixel-level metrics

**Pixel ROC-AUC and AU-PRO** are computed for every subset with masks and are absent otherwise.

**Nothing is accumulated.** Each image is folded into three fixed-bin arrays and discarded — a
positive-pixel histogram, a negative-pixel histogram, and a per-bin sum over ground-truth regions of *that
region's fraction* of pixels in the bin. Memory is constant in image count and resolution. The region
accumulator is **exact**: dividing each region's counts by its size before summing computes the mean
per-region overlap. AU-PRO integrates to a false-positive rate of **0.3**, with the endpoint interpolated.

Protocol choices that move the number:

- **Normal images are included with an all-zero mask** — the MVTec/VisA convention, which makes the figure
  comparable to published ones.
- **A defective image with no mask is skipped and counted**; assuming its defect is nowhere would reward a
  miss.
- **The map is upsampled bilinearly to the mask's resolution**, never the mask downsampled (which invents
  labels).

Bins adapt to the observed score range in one pass over the map files, at 2¹⁶ levels, so the result is not
bit-identical to a direct computation; the error is far below test-set noise. **Everything skipped is
reported** in the metric set and in words on the results screen.

AU-PRO's connected-component labelling is Python-level union-find: fine for sparse masks, the slowest part
of evaluation on masks covering most of the frame.

### Cropped maps

Maps from a region-prepared experiment are stored in source coordinates. Pixels outside the source crop are
`NaN`, rendered transparent, and kept by evaluation in the source-frame denominator at the run's score
floor, so a crop that misses a defect is penalised. Evaluation reports `covered_pixel_fraction`,
`uncovered_defect_pixels` and `uncovered_normal_pixels` beside the evaluated counts.

## Localization

An image-level verdict is blind to *where* the evidence came from: a defective part whose map fires on the
background counts as a true positive. `localized` answers per sample: **does the map's peak fall inside the
annotated region, within a tolerance?** It is not a fifth outcome — a true positive with
`localized = false` is still a true positive, one right for the wrong pixels.

**It is the stored map's peak**, after blur, upsampling and projection to source coordinates — the picture a
reviewer checks against a hand-drawn region — not the patch that produced the score.

**The tolerance is a fraction of the image diagonal**, `EvalConfig.localization_tolerance`, default `0.02`,
resolved to a pixel radius per image and reported. A fixed pixel count would mean "inside one patch" on a
small frame and "four patches away" on a large one. The radius is floored at 1 (a 3×3 window at zero). The
test is a Chebyshev window clamped at the frame edges.

**Persisted, because it does not move with the threshold.** Columns `image_result.peak_x`, `peak_y`,
`localized` and `sample_result.localized` are three-valued: `1` hit, `0` miss, `NULL` **not applicable** (a
normal image, a defect with no resolved mask, an unreadable map). `NULL` is never a miss. The verdicts are
refreshed by re-evaluation and go stale with `metric_set.ground_truth_digest`.

Peaks are recorded as maps are written: `InferContext.write_map` takes the argmax in the pass that finds the
display extremes, so no plugin knows this exists. Re-evaluating a run without peaks backfills them from the
maps on disk, once.

**A part's verdict follows the image that produced its score.** Under `max` that is the winning channel
after normalization, so changing `channel_normalization` moves the verdict too; "any channel hit" would hide
exactly the case of a winning channel firing on background. Under `mean` no single image produced the
number, so any hit counts. A method computing one map per sample writes it to every image, so the choice
between identical verdicts is harmless.

Per subset, `metrics["localization"]`:

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

`tolerance_pixels` is `null` when the subset's images differ in size. `*_with_truth` counts what was
actually tested — a resolved mask *and* a readable map — so it is the denominator of its numerator. **The
block is absent when nothing was checked**, as `pixel` is for a dataset without masks: `0 of 0` would be a
claim about the method made from absent ground truth.

## Rankings

Most-normal and most-anomalous lists are a sort on `SampleResult.agg_score`. **Unlabeled samples are ranked
but excluded from metrics**, which turns a model into a labelling aid without letting unlabeled data reach a
reported number.

## Timing

Per-sample inference time is aggregated from `ImageResult.inference_ms` (mean, median, p95, total), per
experiment and across methods. Methods differ by orders of magnitude in cost, so the accuracy/latency
trade-off is part of the comparison.

## Comparing runs

`GET /api/compare` (`api/routers/compare.py`, `eval/compare.py`) reads N runs against each other and
recomputes nothing: the threshold-independent columns — sample and image ROC-AUC, AP, pixel ROC-AUC, AU-PRO,
timing — are the stored `MetricSet`s. Scores are never compared as values (ADR-0028).

- **One operating-point rule, resolved per run** on that run's own scores, with the value and its rationale
  returned together: `f1` delegates to `suggest_threshold` (the same F1-optimal cut the results screen opens
  at), `recall` is the highest threshold still catching `recall_target` of the defects (default `0.95`). A run
  that cannot meet the rule, or has nothing scored in the subset, has no operating point.
- **Different datasets or splits are refused with 422**; so are fewer than two runs, a repeated run, and more
  than `MAX_RUNS = 6`. Differing preprocessing, evaluation config or region profile, an unscored run, and a
  run trained after it was scored are **warnings** printed on screen (`_differing_keys` names the keys).
- **The per-sample agreement table is computed server-side at the resolved thresholds and is not capped** —
  every disagreeing sample is worth what it costs.
- **Maps are each drawn on their own run's range**, both ranges printed; one cut slider is a *fraction* of
  each range, never a value. Pixel-level curves are absent, because the pixel accumulator keeps no curve.
- **Nothing detects a stale comparison beyond the ground-truth digest and the trained-after-scoring
  warning**; `POST /api/experiments/{id}/reevaluate` is the fix.

## Splits

Splits are assigned at **sample** level, so a part's channels never straddle subsets
([domain model](domain-model.md)). Two strategies serve `anomaly`:

- **`normal_only_train`** — drawn here, seeded and stratified by capture group so an acquisition-batch
  effect cannot land on one side. Fractions are form fields: typically train of normals only, validation of
  held-out normals plus some defects for threshold selection, and the rest as test.
- **`imported`** — the partition the benchmark published, read from the committed manifest
  ([import](import.md)). Only this makes a number comparable to a paper's.

A few-shot task's split holds its references in `train` and its queries in `test`: `manual` lists them,
and `few_shot` draws them from the samples that show the target class ([domain model](domain-model.md)).

**A missing `val` subset is normal.** VisA's official protocol has train and test only, so every layer
tolerates an empty subset:

- samples the manifest does not place are left out, not swept into `test`;
- the train handler passes an empty validation sequence and logs it;
- a method that calibrates on held-out normals (EfficientAD's quantiles) **falls back visibly**, naming what
  it used instead;
- threshold selection returns the highest normal score in the subset **and the sentence explaining that
  choice**, printed under the slider.

## Few-shot segmentation

`eval/segmentation.py` scores one class against background (ADR-0040). Its inputs are the same kind of stored
facts: each image's presence `score`, its map or written mask, and the class truth that
`annotations/class_truth.py` resolves per image.

- **Per image.** No sample-level rule for a class on a multi-channel part has been decided, so every count
  is of images and every rate is named `image_*`. The sample rows are still rebuilt from presence scores,
  so the ranked list and the gallery work unchanged.
- **Unlabelled images are excluded and counted** in `images.unlabeled`. A scored image with neither a map
  nor a mask is counted in `images.without_prediction`, not scored as empty.
- **The prediction** is the method's own mask (`maps/<id>.mask.png`) when it wrote one. Otherwise it is
  the map, read as foreground probability and cut by one fixed rule, `foreground probability >= 0.5`.
  The rule's name and value are stored beside the metrics (ADR-0028). Pixels a region crop left uncovered
  are background.
- **Pixel metrics are pooled counts**, in constant memory:
  - `foreground_iou` and `foreground_dice` over every answered image, so a false positive on an absent
    image counts;
  - `boundary_f1`: boundary pixels matched within `boundary_tolerance_px` (2) either way.
- **Image metrics**:
  - `image_present_recall`: a present image whose predicted region touches the truth;
  - `image_absent_false_positive_rate`: an absent image with any predicted pixel;
  - `image_small_region_recall`: present images whose region is at most `small_region_fraction` (1%) of
    the frame;
  - `image_presence_roc_auc`: presence scores against present and absent, threshold-free;
  - `timing`.

  Each is `None` when its denominator is empty.
- **Per sample, on request.** `sample_outcomes` (`GET /api/experiments/{id}/segmentation-outcomes`) pools a
  sample's answered images and classifies it: `hit` (IoU at least 0.5), `low_iou`, `miss`,
  `false_presence`, `correct_absence`, or `unlabeled`. The IoU travels with each row. Like the anomaly
  threshold report, it is computed from the stored maps and never stored.
- **The ground-truth digest** hashes the class and each image's pinned answer, so a completed revision,
  a relabelled sample or a new class table makes the stored metrics read as stale.

## Run audit

`scripts/audit-run.py <experiment_id>` reads a scored run's stored maps and rows and reports **one CSV row
per image** — label, subset, score, inference time, map and mask paths — and the **image-level ROC-AUC under
every map aggregation**, recomputed from the maps on disk. It is read-only, numpy only, with no torch, so any
old run can be re-examined at no cost. Findings are in [measurements](../measurements.md).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
