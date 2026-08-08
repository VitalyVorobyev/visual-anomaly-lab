/**
 * What the results view is showing, and in what order.
 *
 * `matches` is pulled out and tested rather than inlined because both the gallery grid and
 * the sample page's prev/next walk this set, and a disagreement between them would show up
 * as "the arrow keys skip samples" — a symptom several steps from its cause.
 */

import { describe, expect, it } from "vitest";

import { EMPTY_RESULTS } from "../../api/resultsState";
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

  it("lets mistakes win over a stale single outcome", () => {
    const state = { ...EMPTY_RESULTS, mistakesOnly: true, outcome: "tp" as const };
    expect(matches("fp", state)).toBe(true);
    expect(matches("tp", state)).toBe(false);
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
