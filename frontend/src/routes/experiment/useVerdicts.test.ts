/**
 * What the results view is showing, and in what order.
 *
 * `matches` is pulled out and tested rather than inlined because both the gallery grid and
 * the sample page's prev/next walk this set, and a disagreement between them would show up
 * as "the arrow keys skip samples" — a symptom several steps from its cause.
 */

import { describe, expect, it } from "vitest";

import { OUTCOMES, EMPTY_RESULTS } from "../../api/resultsState";
import { matches } from "./useVerdicts";

describe("the outcome filter", () => {
  it("passes everything when no filter is set", () => {
    for (const outcome of ["tp", "fp", "tn", "fn", "unlabeled"]) {
      expect(matches(outcome, EMPTY_RESULTS)).toBe(true);
    }
  });

  it("narrows to one bucket", () => {
    const state = { ...EMPTY_RESULTS, outcome: "fn" as const };
    expect(matches("fn", state)).toBe(true);
    expect(matches("tp", state)).toBe(false);
  });

  it("treats mistakes as both kinds of wrong", () => {
    // The filter anyone reaches for first, and the one a ranked list cannot assemble.
    const state = { ...EMPTY_RESULTS, mistakesOnly: true };
    expect(matches("fp", state)).toBe(true);
    expect(matches("fn", state)).toBe(true);
    expect(matches("tp", state)).toBe(false);
    expect(matches("tn", state)).toBe(false);
    expect(matches("unlabeled", state)).toBe(false);
  });

  it("counts a segmentation run's misses, false presences and weak finds as mistakes", () => {
    const state = { ...EMPTY_RESULTS, mistakesOnly: true };
    for (const outcome of ["miss", "false_class", "false_presence", "low_iou"]) {
      expect(matches(outcome, state)).toBe(true);
    }
    expect(matches("hit", state)).toBe(false);
    expect(matches("correct_absence", state)).toBe(false);
  });

  it("lets mistakes win over a stale single outcome", () => {
    const state = { ...EMPTY_RESULTS, mistakesOnly: true, outcome: "tp" as const };
    expect(matches("fp", state)).toBe(true);
    expect(matches("tp", state)).toBe(false);
  });

  it("gained no vocabulary from the localization verdict", () => {
    /*
     * `localized` is **orthogonal to the outcome, not a sixth value of it**: a true positive
     * that fired off target is still a true positive. Folding it into this filter would make
     * "false negative" and "off target" mutually exclusive buckets, which they are not, and
     * would quietly change what the gallery's counts mean.
     */
    // The task vocabularies (anomaly, segmentation, detection's `mixed`) and nothing else.
    expect([...OUTCOMES]).toEqual([
      "tp",
      "fp",
      "tn",
      "fn",
      "hit",
      "low_iou",
      "miss",
      "false_class",
      "false_presence",
      "mixed",
      "correct_absence",
      "unlabeled",
    ]);
    for (const outcome of ["localized", "off-target", "off target"]) {
      expect(matches(outcome, { ...EMPTY_RESULTS, outcome: "tp" as const })).toBe(false);
    }
  });

  it("filters the same whether or not the peak layer is on", () => {
    // The new toggle is a display preference, like the heatmap. Nothing it does may reach
    // the set of samples the arrows step through.
    for (const outcome of OUTCOMES) {
      expect(matches(outcome, { ...EMPTY_RESULTS, peak: true })).toBe(
        matches(outcome, EMPTY_RESULTS),
      );
    }
  });
});

/**
 * The ordering, pinned because it was wrong once and looked right.
 *
 * `direction * (right - left)` with `direction = -1` for descending is `left - right`,
 * which is *ascending* — so the control said "most anomalous" and the grid opened on the
 * cleanest samples in the run. A wrong answer wearing a correct label.
 */
describe("rank order", () => {
  const scores = [0.1, 0.9, 0.5];
  const rank = (sort: "score-desc" | "score-asc") =>
    [...scores].sort((left, right) =>
      sort !== "score-asc" ? right - left : left - right,
    );

  it("opens on the most anomalous by default", () => {
    expect(EMPTY_RESULTS.sort).toBe("score-desc");
    expect(rank("score-desc")).toEqual([0.9, 0.5, 0.1]);
  });

  it("reverses when asked for the least", () => {
    expect(rank("score-asc")).toEqual([0.1, 0.5, 0.9]);
  });
});
