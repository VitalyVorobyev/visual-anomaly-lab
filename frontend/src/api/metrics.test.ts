import { describe, expect, it } from "vitest";

import {
  caveats,
  detectionRows,
  formatMilliseconds,
  formatScore,
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
