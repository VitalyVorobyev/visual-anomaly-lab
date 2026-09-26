/**
 * Reading a stored metric set for display.
 *
 * The backend stores metrics as an open JSON object rather than a fixed set of columns,
 * because what is computable depends on the data: a subset with no defects has no
 * ROC-AUC, and a dataset with no masks has no pixel metrics at all. So this file's job is
 * to turn "whatever was computed" into an ordered list of rows, and — the part that
 * matters — to keep a missing metric *missing* rather than rendering it as zero.
 *
 * A fabricated 0.000 on a results screen is worse than a gap, because a gap prompts the
 * question and a zero answers it wrongly.
 */

export type MetricValue = Record<string, unknown>;

export interface MetricRow {
  key: string;
  label: string;
  /** Already formatted. `null` means the metric does not exist for this subset. */
  value: string | null;
  hint?: string;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A score in `[0, 1]` as three decimals, or `null` when it was not computed. */
export function formatScore(value: unknown): string | null {
  const numeric = asNumber(value);
  return numeric === null ? null : numeric.toFixed(3);
}

export function formatMilliseconds(value: unknown): string | null {
  const numeric = asNumber(value);
  if (numeric === null) return null;
  if (numeric >= 1000) return `${(numeric / 1000).toFixed(2)} s`;
  return `${numeric.toFixed(1)} ms`;
}

export function formatCount(value: unknown): string | null {
  const numeric = asNumber(value);
  return numeric === null ? null : String(Math.round(numeric));
}

function countIn(container: unknown, key: string): number | null {
  if (container === null || typeof container !== "object") return null;
  return asNumber((container as MetricValue)[key]);
}

/**
 * Whether a sample in this run holds more than one image.
 *
 * Read from the counts the evaluation layer recorded, never from a channel count or any
 * other structural fact — channel count is data, not schema, and a run over a dataset with
 * no `Channel` rows at all must answer this correctly.
 *
 * It matters because when a sample *is* one image, aggregating with `max` over a single
 * value returns that value: sample- and image-level metrics are then the same number by
 * construction, and reporting both is one finding printed twice. Only the aggregation
 * step distinguishes them, and here there is none.
 *
 * Unknown counts answer `true`. Showing a duplicate is a smaller failure than hiding a
 * metric that was genuinely computed.
 */
export function isGrouped(metrics: MetricValue): boolean {
  const images = countIn(metrics.images, "total");
  const samples = countIn(metrics.samples, "total");
  if (images === null || samples === null) return true;
  return images > samples;
}

/** Why the image-level rows are absent, or `null` when they are present. */
export function groupingNote(metrics: MetricValue): string | null {
  if (isGrouped(metrics)) return null;
  return "Each sample here is a single image, so image-level and sample-level are the same number. Only the sample-level figures are shown.";
}

/**
 * The headline rows, in the order the evaluation handbook ranks them.
 *
 * Sample-level first: it is the unit that matters, a physical part. Image-level next,
 * because it isolates raw model quality from the aggregation choice — the two diverging
 * is a specific, diagnosable finding rather than noise. On an ungrouped dataset they
 * cannot diverge, so the image rows are dropped rather than duplicated.
 */
export function detectionRows(metrics: MetricValue): MetricRow[] {
  const rows: MetricRow[] = [
    {
      key: "sample_roc_auc",
      label: "Sample ROC-AUC",
      value: formatScore(metrics.sample_roc_auc),
      hint: "The headline number, on the unit that matters — one physical part.",
    },
    {
      key: "sample_average_precision",
      label: "Sample AP",
      value: formatScore(metrics.sample_average_precision),
    },
  ];
  if (!isGrouped(metrics)) return rows;

  return [
    rows[0] as MetricRow,
    {
      key: "image_roc_auc",
      label: "Image ROC-AUC",
      value: formatScore(metrics.image_roc_auc),
      hint: "Raw model quality, before channels are aggregated into a part.",
    },
    rows[1] as MetricRow,
    {
      key: "image_average_precision",
      label: "Image AP",
      value: formatScore(metrics.image_average_precision),
    },
  ];
}

/** Pixel rows, or an empty list when this dataset ships no masks. */
export function pixelRows(metrics: MetricValue): MetricRow[] {
  const pixel = metrics.pixel;
  if (pixel === null || typeof pixel !== "object") return [];
  const values = pixel as MetricValue;

  return [
    {
      key: "pixel_roc_auc",
      label: "Pixel ROC-AUC",
      value: formatScore(values.pixel_roc_auc),
      hint: "Whether the map lands where the defect is, not merely that it fired.",
    },
    {
      key: "au_pro",
      label: "AU-PRO",
      value: formatScore(values.au_pro),
      hint: "Every ground-truth region weighted equally, however small.",
    },
    {
      key: "mask_regions",
      label: "Annotated regions",
      value: formatCount(values.mask_regions),
    },
  ];
}

/**
 * Localization rows, or an empty list when nothing was checked against an annotation.
 *
 * Absent rather than zeroed, exactly as the pixel block is: `0 of 0` on a run whose defects
 * are unannotated reads as "this method never localized anything", which is a claim about
 * the method assembled out of the absence of ground truth.
 *
 * The counts are **threshold-free**. They ask whether the map's peak landed on the annotated
 * region, which is a comparison against the ground truth and never against a cut — so unlike
 * a confusion matrix these belong beside ROC-AUC rather than beside a slider.
 */
export function localizationRows(metrics: MetricValue): MetricRow[] {
  const block = metrics.localization;
  if (block === null || typeof block !== "object") return [];
  const values = block as MetricValue;

  return [
    {
      key: "localized_samples",
      label: "Localized detections",
      value: formatOutOf(values.localized_samples, values.defect_samples_with_truth),
      hint: "Annotated defects whose map peaked on the defect rather than somewhere else on the frame. Threshold-free.",
    },
    {
      key: "localization_tolerance",
      label: "Tolerance",
      value: formatTolerance(values),
      hint: "How far the peak may sit from the region and still count. A fraction of the image diagonal, resolved per frame.",
    },
  ];
}

/**
 * The resolved tolerance radius in pixels, or `null` when there is no single one.
 *
 * `null` on a subset whose images are not all the same size: the radius is a fraction of
 * each frame's own diagonal, so one number would name none of them.
 */
export function localizationTolerancePx(metrics: MetricValue): number | null {
  const block = metrics.localization;
  if (block === null || typeof block !== "object") return null;
  return asNumber((block as MetricValue).tolerance_pixels);
}

/** `4 of 9`, or `null` when there was nothing to count. */
function formatOutOf(part: unknown, whole: unknown): string | null {
  const numerator = asNumber(part);
  const denominator = asNumber(whole);
  if (numerator === null || denominator === null || denominator <= 0) return null;
  return `${Math.round(numerator)} of ${Math.round(denominator)}`;
}

/**
 * The tolerance in the most concrete form this subset supports.
 *
 * Pixels where every image shares a frame, because that is the number a reader can check
 * against the marker on screen; the configured fraction otherwise, which is what a
 * mixed-size subset actually has in common.
 */
function formatTolerance(values: MetricValue): string | null {
  const pixels = asNumber(values.tolerance_pixels);
  if (pixels !== null) return `${Math.round(pixels)} px`;
  const fraction = asNumber(values.tolerance_fraction);
  return fraction === null ? null : `${(fraction * 100).toFixed(1)}% of diagonal`;
}

/**
 * A few-shot segmentation run's metrics (ADR-0040): pooled pixel overlap at the printed cut,
 * the threshold-free ranking of the probability map, then what happened per image. Every
 * image-level rate says "image", because no sample-level rule for a class has been decided
 * and these count images.
 */
export function segmentationRows(metrics: MetricValue): MetricRow[] {
  const tolerance = asNumber(metrics.boundary_tolerance_px);
  const small = asNumber(metrics.small_region_fraction);
  return [
    {
      key: "foreground_iou",
      label: "Foreground IoU",
      value: formatScore(metrics.foreground_iou),
      hint: "Pooled over every answered image, so a false region on an absent image counts.",
    },
    { key: "foreground_dice", label: "Foreground Dice", value: formatScore(metrics.foreground_dice) },
    {
      key: "pixel_average_precision",
      label: "Pixel AP (threshold-free)",
      value: formatScore(metrics.pixel_average_precision),
      hint: "The foreground probability ranked against the truth, over present and absent images; no cut.",
    },
    {
      key: "pixel_roc_auc",
      label: "Pixel ROC-AUC (threshold-free)",
      value: formatScore(metrics.pixel_roc_auc),
    },
    {
      key: "boundary_f1",
      label: tolerance === null ? "Boundary F1" : `Boundary F1 (±${tolerance} px)`,
      value: formatScore(metrics.boundary_f1),
    },
    {
      key: "image_present_recall",
      label: "Found, of images that show it",
      value: formatScore(metrics.image_present_recall),
    },
    {
      key: "image_absent_false_positive_rate",
      label: "Flagged, of images without it",
      value: formatScore(metrics.image_absent_false_positive_rate),
      hint: "Lower is better: an absent image with any predicted pixel.",
    },
    {
      key: "image_small_region_recall",
      label: small === null ? "Small-region recall" : `Small-region recall (≤${small * 100}%)`,
      value: formatScore(metrics.image_small_region_recall),
    },
    {
      key: "image_presence_roc_auc",
      label: "Presence ROC-AUC",
      value: formatScore(metrics.image_presence_roc_auc),
      hint: "Threshold-free: presence scores against present and absent images.",
    },
  ];
}

/**
 * A supervised segmentation subset (ADR-0039): the summary read off its confusion matrix,
 * then one IoU per pinned class. A class nobody drew or predicted in the subset has no IoU
 * and stays a dash.
 */
export function semanticRows(metrics: MetricValue): MetricRow[] {
  const perClass = metrics.per_class_iou;
  const classes = Array.isArray(metrics.classes) ? (metrics.classes as unknown[]) : [];
  return [
    {
      key: "mean_iou",
      label: "Mean IoU",
      value: formatScore(metrics.mean_iou),
      hint: "Over the annotation classes with an IoU; background is reported on its own.",
    },
    { key: "background_iou", label: "Background IoU", value: formatScore(metrics.background_iou) },
    { key: "pixel_accuracy", label: "Pixel accuracy", value: formatScore(metrics.pixel_accuracy) },
    {
      key: "mean_class_accuracy",
      label: "Mean class accuracy",
      value: formatScore(metrics.mean_class_accuracy),
    },
    {
      key: "frequency_weighted_iou",
      label: "Frequency-weighted IoU",
      value: formatScore(metrics.frequency_weighted_iou),
      hint: "Each class's IoU weighted by its share of true pixels, background included.",
    },
    ...classes.map((name) => ({
      key: `iou:${String(name)}`,
      label: `IoU · ${String(name)}`,
      value: formatScore(countIn(perClass, String(name))),
    })),
  ];
}

/**
 * An object detection subset (ADR-0039): COCO's AP and recall, then one AP per pinned class. A
 * class with no truth box in the subset has no AP and stays a dash.
 */
export function objectDetectionRows(metrics: MetricValue): MetricRow[] {
  const perClass = metrics.per_class_ap;
  const classes = Array.isArray(metrics.classes) ? (metrics.classes as unknown[]) : [];
  return [
    {
      key: "ap",
      label: "AP@[.5:.95]",
      value: formatScore(metrics.ap),
      hint: "COCO's AP, averaged over IoU thresholds 0.50 to 0.95 and the classes with truth.",
    },
    { key: "ap50", label: "AP50", value: formatScore(metrics.ap50) },
    { key: "ap75", label: "AP75", value: formatScore(metrics.ap75) },
    {
      key: "recall",
      label: "Recall",
      value: formatScore(metrics.recall),
      hint: "The share of truth boxes found, averaged over the same thresholds and classes.",
    },
    { key: "recall50", label: "Recall at IoU 0.5", value: formatScore(metrics.recall50) },
    ...classes.map((name) => ({
      key: `ap:${String(name)}`,
      label: `AP · ${String(name)}`,
      value: formatScore(countIn(perClass, String(name))),
    })),
  ];
}

export function timingRows(metrics: MetricValue): MetricRow[] {
  const timing = metrics.timing;
  if (timing === null || typeof timing !== "object") return [];
  const values = timing as MetricValue;

  return [
    { key: "mean_ms", label: "Mean per image", value: formatMilliseconds(values.mean_ms) },
    { key: "p95_ms", label: "p95 per image", value: formatMilliseconds(values.p95_ms) },
    { key: "total_ms", label: "Total", value: formatMilliseconds(values.total_ms) },
  ];
}

/** One metric across every compared run, index-aligned with the run list. */
export interface ComparisonRow {
  key: string;
  label: string;
  hint?: string;
  /** Already formatted. `null` means this run does not have that metric. */
  values: (string | null)[];
}

/**
 * Turn N runs' metric rows into one table read across.
 *
 * Built by transposing the *same* row builders the single-run screen uses, rather than by
 * a second set of accessors. Two consequences worth having: a comparison can never print a
 * different number from the experiment screen it was reached from, and a metric one run
 * lacks — no masks, an ungrouped dataset — is a dash in that column instead of a missing
 * row that would silently shift every other column's meaning.
 *
 * Row order follows first appearance across the runs, so a run with pixel metrics
 * contributes those rows even when the run beside it has none.
 */
export function comparisonRows(perRun: MetricRow[][]): ComparisonRow[] {
  const order: string[] = [];
  const seen = new Map<string, MetricRow>();
  for (const rows of perRun) {
    for (const row of rows) {
      if (seen.has(row.key)) continue;
      seen.set(row.key, row);
      order.push(row.key);
    }
  }

  return order.map((key) => {
    const template = seen.get(key) as MetricRow;
    return {
      key,
      label: template.label,
      ...(template.hint === undefined ? {} : { hint: template.hint }),
      values: perRun.map((rows) => rows.find((row) => row.key === key)?.value ?? null),
    };
  });
}

/**
 * Anything the run had to skip, phrased so it reads as a caveat rather than a statistic.
 *
 * Returned as sentences because these are the numbers a reader must not scroll past: a
 * pixel metric computed over half the defects is not the metric it claims to be.
 */
export function caveats(metrics: MetricValue): string[] {
  const notes: string[] = [];
  const pixel = metrics.pixel;
  if (pixel !== null && typeof pixel === "object") {
    const values = pixel as MetricValue;
    const unannotated = asNumber(values.skipped_unannotated_defects) ?? 0;
    const missingMaps = asNumber(values.skipped_missing_maps) ?? 0;
    const unreadable = asNumber(values.skipped_unreadable_masks) ?? 0;

    if (unannotated > 0) {
      notes.push(
        `${unannotated} defective image(s) have no ground-truth mask and were left out of the pixel metrics.`,
      );
    }
    if (missingMaps > 0) {
      notes.push(`${missingMaps} anomaly map file(s) could not be read.`);
    }
    if (unreadable > 0) {
      notes.push(`${unreadable} mask file(s) could not be read.`);
    }
  }

  const localization = metrics.localization;
  if (localization !== null && typeof localization === "object") {
    const unannotated = asNumber((localization as MetricValue).unannotated_defect_samples) ?? 0;
    if (unannotated > 0) {
      notes.push(
        `${unannotated} defective sample(s) have no ground-truth region, so nothing could be said about where their map fired.`,
      );
    }
  }

  const samples = metrics.samples;
  if (samples !== null && typeof samples === "object") {
    const values = samples as MetricValue;
    const unlabeled = asNumber(values.unlabeled) ?? 0;
    if (unlabeled > 0) {
      notes.push(`${unlabeled} sample(s) are unlabeled: ranked here, counted in no metric.`);
    }
    if ((asNumber(values.defect) ?? 0) === 0) {
      notes.push("This subset has no defects, so no ROC-AUC exists for it.");
    }
  }

  return notes;
}
