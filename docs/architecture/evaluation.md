# Evaluation

The evaluation layer is **model-independent by construction**. Its only inputs are:

- `ImageResult.score` rows for an experiment,
- `Sample.label`,
- `SplitAssignment.subset`,
- the resolved ground-truth mask for each image and the float32 map on disk, read through
  `map_files.read_map`, when pixel or localization metrics are computed.

It never imports a model module and never re-runs inference, so every method is evaluated by the same code.

**The evaluator is chosen by task** (ADR-0039) from the table in `eval/evaluators.py`. Most of this page
describes the `anomaly` evaluator, `eval/runner.py`; [few-shot segmentation](#few-shot-segmentation),
[semantic segmentation](#semantic-segmentation) and [object detection](#object-detection) have their own
sections. The `infer` job, re-evaluation and the staleness check all go through
`evaluator_for(experiment.task)`, and a task with no entry cannot be created. Each evaluator names a
`headline` metric (`sample_roc_auc`, `foreground_iou`, `mean_iou`, `ap`), which the `infer` log prints
per subset.

## Channel selection

`Experiment.channels` is a frozen list of channel **names**; empty means every channel. It is applied in
`list_images_for_split`, the one place that answers "which images", so training, inference and on-demand
diagnostics narrow identically — asking a bright-field-only run about a dark-field image is a 404 naming the
channel.

Split, labels and region build are shared by every run over one multi-channel dataset, so a difference
between "one channel" and "all channels" is the channel, not two imports. An image whose `channel_id` is
`NULL` is **excluded by a non-empty selection**; a single-view dataset is only read unfiltered.

The selection is the experiment's and not the split's: a split decides *which samples* and exists to
prevent leakage, and a selection on it would make "one channel" and "all channels" incomparable by
construction. Importing each channel as a dataset of its own is the other thing it replaces — separate
datasets have independent splits, so one view of a part could train while another is tested. A valid name
that no sample in the split carries yields a legal, empty run; the unknown-name check catches typos only.
Wherever runs with different selections are shown side by side, images align by channel name, never by
position.

## Channel → sample aggregation

A part is scored from its per-image scores. The **default aggregation is `max`**: a defect visible under any
single illumination makes the part defective, while `mean` dilutes single-channel evidence with
uninformative views. `mean` is an option. The method is recorded in `SampleResult.aggregation` and
`Experiment.eval_config`. The reduction lives here and not in the methods, so no method's number is partly
a measure of its own fusion; a learned fusion would need labels the workbench cannot assume. `max` is the
least robust choice — one noisy view, a specular flare or a registration failure, sets the part's score.

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

**Image-level ROC-AUC stays on raw scores**: it isolates model quality from how channels were
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
Every other sample of the dataset is a query, so on a dataset of many classes the other classes' images
are the target's negatives — confirmed absent by their own completed annotations, not by a label
(ADR-0041).
A semantic segmentation or detection run fits on the annotated images of whatever `train` holds, so it is offered
`class_stratified` — annotated samples drawn under the seed, stratified by the set of classes each
shows, with the rest parked in `test` where they are scored but measured against nothing — and
`manual`. The anomaly strategies put normals alone in `train`, and a supervised run cannot learn a
class from them.

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
- **Present, absent and unlabelled come from class truth** (`class_truth.resolve_class_truth`, the
  presence rule of [annotations](annotations.md)): an image whose newest completed revision shows the
  target is present, one whose revision answers for the target without showing it is absent, whatever
  other class it shows. A sample's anomaly label is read only for the default class `defect`, where a
  `normal` sample without a revision is absent (ADR-0041). One multi-class dataset therefore measures
  exactly what a dataset per class, its other classes labelled normal, would.
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
- **The map's ranking, threshold-free.** Where the method stored a map, its foreground probability is
  folded into the anomaly evaluator's fixed-bin histograms (`eval/pixel.py`, over `[0, 1]`) against the
  class truth, over present and absent images alike. `pixel_average_precision` and `pixel_roc_auc` are
  read from them, so the map is measured apart from the cut; `pixel_map_images` counts the images that
  contributed. A written mask adds nothing here, and both are `None` when the subset has no map or no
  foreground pixel.
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
  a relabelled sample or a new class table makes the stored metrics read as stale. An answer read from a
  document in memory also names the polygon rule it was rasterised by when the document holds a polygon
  ([annotations](annotations.md#completion-and-storage)).

## Semantic segmentation

`eval/semantic.py` scores every class a run pinned, per pixel (ADR-0039). Its inputs are each image's label
map as the method wrote it (`maps/<id>.labels.png`, source frame) and the label truth that
`annotations/class_truth.py` resolves over the pinned classes.

- **One confusion matrix per subset**, `(classes + 1)²` counts with background first, rows true and
  columns predicted. Each image adds its counts and is dropped, so memory is constant in pixels. The
  matrix is stored with the metrics under `confusion`.
- **Per image.** As for few-shot, no sample-level rule for classes has been decided; the sample rows are
  rebuilt from the image scores, so ranking and the gallery work.
- **Labelled means answered for every pinned class** (see [annotations](annotations.md#raster-contract)).
  Other images are excluded and counted in `images.unlabeled`; a scored image with no label map is
  `images.without_prediction`. A pixel drawn in a class the run was not created with is `IGNORE_INDEX`
  (255): counted in `ignored_pixels` and nowhere else.
- **Nothing is thresholded.** A label map is the method's decision, read as written, so no cut needs a
  per-run rule (ADR-0028).
- **Metrics**, each `None` when its denominator is empty:
  - `per_class_iou` and `per_class_accuracy`, by class key. A class absent from truth *and* prediction in a
    subset has no IoU — never 0 — and one absent from truth has no accuracy;
  - `mean_iou` (the headline) and `mean_class_accuracy`, averaged over the annotation classes that have a
    value. Background is left out of both, because a background covering most of every frame would carry
    the mean; its own `background_iou` sits beside them;
  - `pixel_accuracy` and `frequency_weighted_iou`, which count background, as their definitions do;
  - `timing`.
- **Per sample, on request.** `semantic.sample_outcomes` (the same
  `GET /api/experiments/{id}/segmentation-outcomes`, with `threshold_rule` saying the label map is read as
  written) pools a sample's labelled images into one `(classes + 1)²` matrix — never pixels — and reads
  it: shows no class → `correct_absence` or `false_presence`; a class it shows found nowhere → `miss`; a
  class predicted that it does not show → `false_class`; otherwise `hit` or `low_iou` by mean IoU at 0.5.
  The mean IoU over the classes shown or predicted travels as `iou`, `None` when it shows none; a sample
  with no labelled image is `unlabeled`. Computed from the stored maps and never stored.
- **Label maps for drawing.** `GET /api/experiments/{id}/images/{iid}/labels` serves an image's stored
  label map, and `?truth=true` its truth over the pinned classes, as a value plane of class indices
  (`media/values.py`; a truth pixel of a class the run does not know is NaN). Only for a
  `semantic_segmentation` run (409 otherwise) and a scored image (404 otherwise); colour is the
  interface's. `…/label-map?colours=&truth=` draws the same map as a PNG for a gallery tile
  ([media](media.md)).
- **The ground-truth digest** hashes the pinned class list and each image's pinned answer, and, for one
  read from a document in memory, the polygon rule it was rasterised by.

## Object detection

`eval/detection.py` scores every class a run pinned, as boxes, by COCO's protocol (ADR-0039). Its inputs
are each image's detections as the method wrote them (`maps/<id>.instances.json`, source frame, most
confident first) and the box truth that `annotations/class_truth.py` resolves over the pinned classes
(see [annotations](annotations.md#detection-truth)).

- **Matching.** Per image and class, detections most confident first each take the unmatched truth box
  they overlap most, if that IoU reaches the threshold — greedy by confidence, not an assignment — at
  each of ten IoU thresholds, 0.50 to 0.95. Boxes are pixel-edge, so a box's area is
  `(x1 - x0) · (y1 - y0)`.
- **AP** is COCO's 101-point interpolated area under the precision envelope, per class and threshold,
  from the class's detections pooled over the subset and ranked by confidence (ties in stored order).
- **Bounded memory.** Each class keeps one confidence and one row of ten matched flags per detection,
  and a truth count — linear in detections, which the write seam caps at 100 an image, never in pixels.
- **Per image**, as for the segmentation tasks; the sample rows are rebuilt from the image scores (the
  top confidence), so ranking and the gallery's order work.
- **Labelled means answered for every pinned class**, the rule semantic segmentation labels by. Other
  images are `images.unlabeled`; a scored image with no detection file is `images.without_prediction`.
  A truth box of a class the run was not created with is counted in `ignored_instances` and nowhere
  else; a detection of one is refused by name.
- **No metric is cut.** AP and recall read a run's detections in its own confidence order, so they need
  no per-run rule (ADR-0028). The IoU thresholds are the protocol's, the same for every run.
- **One confidence cut per subset, for the verdicts.** A drawn box and a per-sample verdict need a cut,
  so the evaluator resolves one by one rule — the confidence that maximises F1 at IoU 0.5 over the
  subset, every class pooled (`CUT_RULE`) — and stores it beside the metrics: `confidence_cut`,
  `cut_rule`, `cut_iou`, and `f1_at_cut`, `precision_at_cut`, `recall_at_cut`. Only a confidence some
  detection has is a candidate, a cut keeps every detection at or above it (ties go together), and among
  equal F1 the highest cut wins. Matching is greedy by confidence, so the detections above a cut match as
  they would alone. A subset with no truth box or no detection has no cut (`null`), and its rule says
  every detection counts. The cut is the run's own and never crosses to another run (`CUT_KEYS`).
- **Metrics**, each `None` when it cannot be computed:
  - `per_class_ap` (AP@[.5:.95]), `per_class_ap50`, `per_class_ap75`, and `per_class_recall` (the share
    of truth boxes matched, averaged over the thresholds) with `per_class_recall50`. A class with no
    truth box in the subset has none of them — its detections are false positives no recall can be
    measured against. A class with truth and no detection has AP 0, measured;
  - `ap` (the headline), `ap50`, `ap75`, `recall` and `recall50`: each averaged over the classes that
    have a value;
  - `truth_instances` and `predicted_instances` per class, `iou_thresholds`, and `timing`.
- **Per image, at the stored cut.** `GET /api/experiments/{id}/images/{iid}/boxes` serves one scored
  image's detections (at most 100, most confident first, each with its class index, confidence, `kept`
  and `matched`) and its true boxes (each `found`, and a class the run does not pin with no index),
  source frame and pixel-edge, matched at IoU 0.5 and the cut stored for the image's subset, which it
  returns with its rule. A list is `null` when there is no such answer; the route is 404 for an
  unscored image or one with neither, 409 for another task. `…/box-map?colours=&predictions=&truth=`
  draws the kept detections solid and the pinned truth dashed as an SVG for a gallery tile
  ([media](media.md)).
- **Per sample, at the stored cut.** `GET /api/experiments/{id}/detection-outcomes?subset=`
  (`sample_outcomes`) pools each sample's labelled images into three counts — truth boxes missed, kept
  detections matched, kept detections that matched nothing — never a box list: no truth is
  `false_presence` if a kept detection is left and `correct_absence` otherwise; a miss and an unmatched
  detection together are `mixed`; a miss alone `miss`; an unmatched detection alone `false_presence`;
  neither `hit`; no labelled image `unlabeled`. It returns the cut and its rule, and answers 409 for
  another task; the segmentation outcome and label routes answer 409 for a detection run.
- **Compared on what needs no cut.** `GET /api/compare/detection?ids=&subset=` takes runs of one dataset,
  split and class list and returns each one's stored metrics without `CUT_KEYS`.
- **The ground-truth digest** hashes the pinned class list and each image's pinned answer — its
  instances file's digest, its document's source provenance and polygon rule, or its imported mask's.

## Run audit

`scripts/audit-run.py <experiment_id>` reads a scored run's stored maps and rows and reports **one CSV row
per image** — label, subset, score, inference time, map and mask paths — and the **image-level ROC-AUC under
every map aggregation**, recomputed from the maps on disk. It is read-only, numpy only, with no torch, so any
old run can be re-examined at no cost. Findings are in [measurements](../measurements.md).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
