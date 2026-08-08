# Evaluation layer

The evaluation layer is **model-independent by construction** (ADR-0011). Its only inputs are:

- `ImageResult.score` rows for an experiment,
- `Sample.label`,
- `SplitAssignment.subset`.

It never imports a model module and never re-runs inference. Every method is therefore evaluated by exactly
the same code, which is the precondition for the comparison view to mean anything.

## Channel → sample aggregation

A part is scored from its per-image scores. The **default aggregation is `max`**: a defect visible under any
single illumination makes the part defective, and `mean` dilutes single-channel evidence by averaging a strong
signal with two uninformative views — precisely the failure mode this dataset invites. `mean` is available as
an option, and the method used is recorded in both `SampleResult.aggregation` and `Experiment.eval_config`, so
a stored result always says how it was produced.

**Caveat:** `max` assumes per-channel scores are comparable in scale. This holds for the classical baseline,
whose scores are z-score-based per channel by construction (ADR-0010). It is *not* automatic for deep models,
whose raw score scales can differ between channels. **Per-channel quantile normalization before aggregation**
is therefore a backlog item, to be evaluated against the `max`/`mean` baselines rather than assumed to help.

## Metrics

**Headline metric: sample-level ROC-AUC.** It is threshold-free, robust to the class imbalance of the
reference data, and evaluated on the unit that matters — the physical part. Image-level ROC-AUC is reported
alongside it as a diagnostic (it reveals when a model scores individual views well but aggregation is losing
that signal).

Persisted in `MetricSet`: **threshold-independent metrics only** — sample-level and image-level ROC-AUC,
average precision, per-subset sample counts, and timing summaries.

**Threshold-dependent outputs are computed on demand** from the persisted scores: confusion matrix,
precision / recall / F1, and the FP/FN sample lists. `GET /api/eval/{experiment_id}/threshold?value=…` returns
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

Two known weaknesses: `verify` cannot detect mask drift, because schema v1 has no `mask.sha256` and the schema
is frozen — a mask re-exported in place is invisible, and lifting this is a numbered migration. And AU-PRO's
connected-component labelling is Python-level union-find, which is fine for the sparse masks these datasets
have and would be the slowest part of evaluation on masks that cover most of the frame.

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

**Adopted from the source** (`imported`, ADR-0016): the partition the benchmark published, read out of the
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
[the measurements](../measurements-efficientad.md).

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
