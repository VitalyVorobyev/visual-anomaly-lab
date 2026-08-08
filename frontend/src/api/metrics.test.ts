import { describe, expect, it } from "vitest";

import {
  caveats,
  comparisonRows,
  detectionRows,
  formatMilliseconds,
  formatScore,
  groupingNote,
  isGrouped,
  pixelRows,
  timingRows,
} from "./metrics";

describe("formatting a metric", () => {
  it("shows three decimals for a score", () => {
    expect(formatScore(0.9876543)).toBe("0.988");
  });

  it("returns null rather than zero when a metric does not exist", () => {
    // The whole reason this module exists. A subset with no defects has no ROC-AUC, and
    // rendering 0.000 answers a question that should be asked.
    expect(formatScore(null)).toBeNull();
    expect(formatScore(undefined)).toBeNull();
  });

  it("refuses a non-finite number rather than printing NaN", () => {
    expect(formatScore(Number.NaN)).toBeNull();
    expect(formatScore(Number.POSITIVE_INFINITY)).toBeNull();
  });

  it("switches to seconds once milliseconds stop being readable", () => {
    expect(formatMilliseconds(12.34)).toBe("12.3 ms");
    expect(formatMilliseconds(4500)).toBe("4.50 s");
  });
});

describe("detection rows", () => {
  it("puts the sample-level number first", () => {
    const rows = detectionRows({ sample_roc_auc: 0.91, image_roc_auc: 0.88 });
    expect(rows[0]?.key).toBe("sample_roc_auc");
    expect(rows[0]?.value).toBe("0.910");
    expect(rows[1]?.value).toBe("0.880");
  });

  it("keeps a row present with no value when the metric is absent", () => {
    const rows = detectionRows({ sample_roc_auc: 0.5 });
    const imageAuc = rows.find((row) => row.key === "image_roc_auc");
    expect(imageAuc).toBeDefined();
    expect(imageAuc?.value).toBeNull();
  });

  it("drops the image rows when a sample is one image", () => {
    // 500 images over 500 samples: `max` over a single value is that value, so the two
    // levels are the same number by construction. Printing both is one finding twice.
    const rows = detectionRows({
      sample_roc_auc: 0.727,
      image_roc_auc: 0.727,
      images: { total: 270 },
      samples: { total: 270 },
    });
    expect(rows.map((row) => row.key)).toEqual(["sample_roc_auc", "sample_average_precision"]);
  });

  it("keeps both levels when a sample groups several images", () => {
    const rows = detectionRows({
      sample_roc_auc: 0.9,
      image_roc_auc: 0.84,
      images: { total: 800 },
      samples: { total: 200 },
    });
    expect(rows.map((row) => row.key)).toEqual([
      "sample_roc_auc",
      "image_roc_auc",
      "sample_average_precision",
      "image_average_precision",
    ]);
  });
});

describe("grouping", () => {
  it("reads the counts the evaluation layer recorded, not a channel count", () => {
    // Channel count is data, not schema. The showcase dataset has channels; VisA has no
    // `Channel` rows at all, and both must answer this from the same two numbers.
    expect(isGrouped({ images: { total: 800 }, samples: { total: 200 } })).toBe(true);
    expect(isGrouped({ images: { total: 270 }, samples: { total: 270 } })).toBe(false);
  });

  it("shows both levels when the counts are unknown", () => {
    // Hiding a metric that was genuinely computed is the worse failure of the two.
    expect(isGrouped({ sample_roc_auc: 0.9 })).toBe(true);
    expect(groupingNote({ sample_roc_auc: 0.9 })).toBeNull();
  });

  it("explains the absence rather than leaving two rows unaccounted for", () => {
    expect(groupingNote({ images: { total: 270 }, samples: { total: 270 } })).toContain(
      "single image",
    );
  });
});

describe("pixel rows", () => {
  it("are empty for a dataset with no masks", () => {
    expect(pixelRows({ sample_roc_auc: 0.9 })).toEqual([]);
  });

  it("appear once a pixel block exists", () => {
    const rows = pixelRows({ pixel: { pixel_roc_auc: 0.97, au_pro: 0.83, mask_regions: 42 } });
    expect(rows.map((row) => row.value)).toEqual(["0.970", "0.830", "42"]);
  });
});

describe("timing rows", () => {
  it("are empty when nothing was timed", () => {
    expect(timingRows({})).toEqual([]);
  });

  it("report the distribution rather than only the mean", () => {
    const rows = timingRows({ timing: { mean_ms: 12, p95_ms: 30, total_ms: 2400 } });
    expect(rows.map((row) => row.key)).toEqual(["mean_ms", "p95_ms", "total_ms"]);
  });
});

describe("caveats", () => {
  it("say when defects were left out of the pixel metrics", () => {
    const notes = caveats({ pixel: { skipped_unannotated_defects: 4 } });
    expect(notes[0]).toContain("4 defective image(s)");
  });

  it("say when unlabeled samples were ranked but not counted", () => {
    const notes = caveats({ samples: { total: 10, normal: 5, defect: 2, unlabeled: 3 } });
    expect(notes.join(" ")).toContain("counted in no metric");
  });

  it("explain an absent ROC-AUC rather than leaving a dash unexplained", () => {
    const notes = caveats({ samples: { total: 5, normal: 5, defect: 0, unlabeled: 0 } });
    expect(notes.join(" ")).toContain("no defects");
  });

  it("stay silent when there is nothing to warn about", () => {
    expect(
      caveats({
        samples: { total: 6, normal: 3, defect: 3, unlabeled: 0 },
        pixel: { skipped_unannotated_defects: 0, skipped_missing_maps: 0 },
      }),
    ).toEqual([]);
  });
});

describe("laying several runs out as columns", () => {
  const grouped = { images: { total: 6 }, samples: { total: 3 } };

  it("keeps a metric one run lacks as a dash in that column", () => {
    // Not a dropped row: dropping it would shift every other column's meaning, and the
    // absence — no masks in that run's dataset, an ungrouped split — is itself the finding.
    const rows = comparisonRows([
      pixelRows({ pixel: { pixel_roc_auc: 0.9, au_pro: 0.8, mask_regions: 4 } }),
      pixelRows({}),
    ]);
    expect(rows.map((row) => row.key)).toEqual(["pixel_roc_auc", "au_pro", "mask_regions"]);
    expect(rows[0]?.values).toEqual(["0.900", null]);
  });

  it("contributes rows from whichever run has them", () => {
    // A run with pixel metrics beside one without still gets its rows drawn.
    const rows = comparisonRows([pixelRows({}), pixelRows({ pixel: { pixel_roc_auc: 0.5 } })]);
    expect(rows[0]?.values).toEqual([null, "0.500"]);
  });

  it("reuses the single-run builders, so a comparison cannot print a different number", () => {
    const metrics = { ...grouped, sample_roc_auc: 0.75, image_roc_auc: 0.5 };
    const rows = comparisonRows([detectionRows(metrics)]);
    const single = detectionRows(metrics);
    expect(rows.map((row) => row.values[0])).toEqual(single.map((row) => row.value));
  });

  it("is empty when no run computed anything", () => {
    expect(comparisonRows([[], []])).toEqual([]);
  });
});
