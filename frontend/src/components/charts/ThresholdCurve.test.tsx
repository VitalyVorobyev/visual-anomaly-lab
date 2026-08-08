import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ThresholdCurve } from "./ThresholdCurve";

/** Four cuts of a PR curve, descending by score the way the server emits them. */
const curve = {
  x: [0.25, 0.5, 0.75, 1.0], // recall
  y: [1.0, 1.0, 0.75, 0.5], // precision
  t: [0.9, 0.8, 0.7, 0.6],
  total: 4,
  dropped: 0,
};

describe("ThresholdCurve", () => {
  it("draws precision, recall and F1 against the threshold", () => {
    const { container } = render(
      <ThresholdCurve curve={curve} active={0.7} suggested={0.8} domain={[0.5, 1]} />,
    );

    expect(
      screen.getByLabelText("Precision, recall and F1 against the threshold"),
    ).toBeTruthy();
    expect(container.querySelectorAll("path")).toHaveLength(3);
    for (const name of ["precision", "recall", "F1"]) {
      expect(screen.getByText(name)).toBeTruthy();
    }
  });

  it("marks the threshold in force and the opening position differently", () => {
    /**
     * Two rules that looked alike would be worse than one: the reader has to be able to
     * tell "where I put it" from "where the server would have put it", because the gap
     * between them is the whole reason to move the slider.
     */
    const { container } = render(
      <ThresholdCurve curve={curve} active={0.7} suggested={0.8} domain={[0.5, 1]} />,
    );

    // The dashed one is the suggestion; the solid one is the active cut.
    expect(container.querySelectorAll("line[stroke-dasharray]")).toHaveLength(1);
  });

  it("draws no suggestion rule when there is none", () => {
    const { container } = render(
      <ThresholdCurve curve={curve} active={0.7} suggested={null} domain={[0.5, 1]} />,
    );
    expect(container.querySelectorAll("line[stroke-dasharray]")).toHaveLength(0);
  });

  it("states the absence rather than drawing precision against recall", () => {
    /**
     * A ROC curve, or an index written before thresholds were carried, arrives with an
     * empty `t`. Plotting `x` as if it were a score would put recall on an axis labelled
     * "threshold" — a chart that is wrong rather than missing.
     */
    render(
      <ThresholdCurve
        curve={{ ...curve, t: [] }}
        active={0.7}
        suggested={0.8}
        domain={[0.5, 1]}
      />,
    );

    expect(screen.getByText(/only one class in it/)).toBeTruthy();
    expect(screen.queryByLabelText(/against the threshold/)).toBeNull();
  });

  it("states the absence for a curve the server could not compute", () => {
    render(<ThresholdCurve curve={null} active={0} suggested={null} domain={[0, 1]} />);
    expect(screen.getByText(/only one class in it/)).toBeTruthy();
  });

  it("reports a downsampled curve rather than implying it drew every cut", () => {
    render(
      <ThresholdCurve
        curve={{ ...curve, total: 5000, dropped: 4996 }}
        active={0.7}
        suggested={0.8}
        domain={[0.5, 1]}
      />,
    );
    expect(screen.getByText(/5000 cuts, downsampled to 4/)).toBeTruthy();
  });
});
