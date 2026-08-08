import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CurveChart } from "./CurveChart";
import { ScoreHistogram } from "./Histogram";
import { LineChart } from "./LineChart";

function paths(container: HTMLElement): SVGPathElement[] {
  return Array.from(container.querySelectorAll("path"));
}

describe("LineChart", () => {
  it("draws one path per series and names each in the legend", () => {
    const { container } = render(
      <LineChart
        label="Training losses"
        series={[
          {
            name: "loss_st",
            points: [
              { x: 0, y: 1 },
              { x: 1, y: 0.5 },
            ],
          },
          {
            name: "loss_ae",
            points: [
              { x: 0, y: 2 },
              { x: 1, y: 1.5 },
            ],
          },
        ]}
      />,
    );

    expect(paths(container)).toHaveLength(2);
    expect(screen.getByLabelText("Training losses")).toBeTruthy();
    expect(screen.getByText("loss_st")).toBeTruthy();
    expect(screen.getByText("loss_ae")).toBeTruthy();
  });

  it("renders an empty series without throwing, so a pending loss term is visible", () => {
    const { container } = render(
      <LineChart label="Empty" series={[{ name: "loss_st", points: [] }]} />,
    );

    expect(paths(container)).toHaveLength(1);
    expect(paths(container)[0]?.getAttribute("d")).toEqual("");
    expect(screen.getByText("loss_st")).toBeTruthy();
  });

  it("draws a single point as a dot rather than as nothing", () => {
    const { container } = render(
      <LineChart label="One point" series={[{ name: "lr", points: [{ x: 0, y: 0.0001 }] }]} />,
    );

    expect(paths(container)).toHaveLength(0);
    expect(container.querySelectorAll("circle")).toHaveLength(1);
  });

  it("survives a series that is constant, which a learning rate often is", () => {
    const { container } = render(
      <LineChart
        label="Flat"
        series={[
          {
            name: "lr",
            points: [
              { x: 0, y: 0.001 },
              { x: 100, y: 0.001 },
            ],
          },
        ]}
      />,
    );

    expect(paths(container)[0]?.getAttribute("d")).not.toContain("NaN");
  });

  it("puts every coordinate on a log axis at a finite pixel, including a zero", () => {
    const { container } = render(
      <LineChart
        label="Log"
        logY
        series={[
          {
            name: "loss",
            points: [
              { x: 0, y: 1 },
              { x: 1, y: 0 },
            ],
          },
        ]}
      />,
    );

    expect(paths(container)[0]?.getAttribute("d")).not.toContain("NaN");
  });
});

describe("CurveChart", () => {
  const curve = { x: [0, 0.5, 1], y: [0, 0.8, 1], t: [], total: 3, dropped: 0 };

  it("draws the curve and its chance diagonal", () => {
    const { container } = render(<CurveChart curve={curve} kind="roc" label="ROC" area={0.9} />);

    expect(screen.getByLabelText("ROC")).toBeTruthy();
    expect(container.querySelectorAll("line[stroke-dasharray]").length).toBeGreaterThan(0);
    expect(screen.getByText(/0\.9000/)).toBeTruthy();
  });

  it("states the absence rather than drawing a chance diagonal alone", () => {
    render(<CurveChart curve={null} kind="roc" label="ROC" absent="Only normals here." />);
    expect(screen.getByText("Only normals here.")).toBeTruthy();
  });

  it("renders an uncomputable area as a dash, never as zero", () => {
    render(<CurveChart curve={curve} kind="pr" label="PR" area={null} />);
    expect(screen.getByText("—")).toBeTruthy();
  });

  it("says when it downsampled rather than implying it drew everything", () => {
    render(
      <CurveChart
        curve={{ x: [0, 1], y: [0, 1], t: [], total: 5000, dropped: 4998 }}
        kind="roc"
        label="ROC"
      />,
    );
    expect(screen.getByText(/5000 points, downsampled to 2/)).toBeTruthy();
  });
});

describe("ScoreHistogram", () => {
  it("bins both classes on one axis and marks the threshold", () => {
    const { container } = render(
      <ScoreHistogram
        label="Scores"
        normal={[0.1, 0.2, 0.15]}
        defect={[0.8, 0.9]}
        threshold={0.5}
        bins={4}
      />,
    );

    expect(container.querySelectorAll("rect").length).toBeGreaterThan(0);
    expect(container.querySelectorAll("line[stroke-dasharray]")).toHaveLength(1);
    expect(screen.getByText("normal (3)")).toBeTruthy();
    expect(screen.getByText("defect (2)")).toBeTruthy();
  });

  it("renders a subset with no defects without throwing", () => {
    render(<ScoreHistogram label="Normals only" normal={[0.1, 0.2]} defect={[]} />);
    expect(screen.getByText("defect (0)")).toBeTruthy();
  });

  it("renders nothing scored at all without throwing", () => {
    const { container } = render(<ScoreHistogram label="Empty" normal={[]} defect={[]} />);
    expect(container.querySelector("svg")).toBeTruthy();
  });
});
