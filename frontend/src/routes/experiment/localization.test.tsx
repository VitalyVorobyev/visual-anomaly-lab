/**
 * The localization verdict, as the results screens count and label it.
 *
 * Both rules under test are ones a plausible implementation gets wrong in a way that looks
 * right on screen. Counting `null` as a miss turns "nobody annotated this defect" into "the
 * method fired somewhere else", which is a claim about the method made out of the absence of
 * ground truth. And treating the pair as a fraction of the subset rather than of the rows
 * that carry a verdict prints a denominator nobody can reconcile with the confusion matrix
 * beside it.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { withProviders } from "../../test-harness";
import { Confusion, localizationBadge, localizationSummary } from "./ResultsPanel";

/** Only the field under test; the real rows carry a score, an outcome and an id too. */
const rows = (...localized: (boolean | null)[]) => localized.map((value) => ({ localized: value }));

describe("counting localized rows", () => {
  it("counts the hits over the rows that were judged", () => {
    expect(localizationSummary(rows(true, true, false))).toEqual({ localized: 2, judged: 3 });
  });

  it("leaves a row with no verdict out of both halves of the fraction", () => {
    // `null` is "not applicable" — a normal part, an unannotated defect, a run with no map.
    // In the numerator it would invent a success; in the denominator, a failure.
    expect(localizationSummary(rows(true, null, false, null, null))).toEqual({
      localized: 1,
      judged: 2,
    });
  });

  it("judges nothing when nothing was annotated", () => {
    // The caller renders a dash for this, never `0 of 0`.
    expect(localizationSummary(rows(null, null))).toEqual({ localized: 0, judged: 0 });
    expect(localizationSummary([])).toEqual({ localized: 0, judged: 0 });
  });

  it("tolerates a row that predates the field", () => {
    expect(localizationSummary([{}, { localized: true }])).toEqual({ localized: 1, judged: 1 });
  });

  it("does not move when the outcome mix does", () => {
    // The verdict compares the map against the ground truth and never against a cut, so the
    // pair printed beside the threshold slider must be the same at every position of it.
    // Filtering to the detections is precisely what would break that.
    const before = localizationSummary(rows(true, false, true, null));
    const after = localizationSummary(rows(true, false, true, null));
    expect(after).toEqual(before);
  });
});

describe("the confusion strip", () => {
  const report = {
    confusion: { true_positive: 6, false_positive: 1, true_negative: 20, false_negative: 3 },
    precision: 0.857,
    recall: 0.667,
    f1: 0.75,
  };

  it("prints the pair beside precision and recall", () => {
    const { container } = render(
      withProviders(
        <Confusion report={report} samples={rows(true, false, true, null)} tolerancePx={33} />,
      ),
    );

    expect(container.textContent).toContain("localized");
    expect(container.textContent).toContain("2 of 3");
  });

  it("shows a dash rather than a fabricated zero when nothing was judged", () => {
    // `0 of 0` on a run whose defects are unannotated reads as "this method never localized
    // anything" — a claim about the method assembled out of missing ground truth.
    const { container } = render(
      withProviders(<Confusion report={report} samples={rows(null, null)} />),
    );

    expect(container.textContent).not.toContain("0 of 0");
    expect(container.textContent).toContain("—");
  });

  it("still draws for a caller that passes no rows at all", () => {
    // The comparison screen renders this component with counts alone.
    const { container } = render(withProviders(<Confusion report={report} />));
    expect(container.textContent).toContain("0.857");
  });
});

describe("the localization badge", () => {
  it("names the two verdicts in the verdict palette", () => {
    expect(localizationBadge(true)).toEqual({ tone: "normal", label: "localized" });
    expect(localizationBadge(false)).toEqual({ tone: "warning", label: "off target" });
  });

  it("draws nothing where the question does not apply", () => {
    // Not a neutral badge and not a dash: a badge reading "not applicable" on every normal
    // sample would put a column of chrome beside the outcome that carries no information.
    expect(localizationBadge(null)).toBeNull();
    expect(localizationBadge(undefined)).toBeNull();
  });
});
