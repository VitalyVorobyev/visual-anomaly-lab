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
 * The headline rows, in the order ADR-0011 ranks them.
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
