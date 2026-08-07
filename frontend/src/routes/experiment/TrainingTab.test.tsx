import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { JobSummary } from "../../api/client";
import type { Series } from "../../hooks/useJob";
import { TrainingTab } from "./TrainingTab";

const trainJob = { id: 1, kind: "train", status: "succeeded" } as JobSummary;

function series(name: string, points: { step: number; value: number }[]): Series {
  return { name, points, total: points.length, dropped: 0 };
}

describe("TrainingTab", () => {
  it("charts the loss terms together, on one axis", () => {
    const { container } = render(
      <TrainingTab
        jobs={[trainJob]}
        followingJobId={undefined}
        series={[
          series("loss_st", [
            { step: 0, value: 1 },
            { step: 20, value: 0.5 },
          ]),
          series("loss_ae", [
            { step: 0, value: 2 },
            { step: 20, value: 1 },
          ]),
        ]}
      />,
    );

    expect(screen.getByLabelText("Training losses per step")).toBeTruthy();
    expect(container.querySelectorAll("path")).toHaveLength(2);
  });

  it("prints a series with one point as a number, not as a chart of one dot", () => {
    /**
     * `pixel_reference` reports `reference_images` once and nothing else. Plotting that
     * against a step axis running -1 to 1 is a chart of nothing, and reads as a run that
     * failed to record anything.
     */
    render(
      <TrainingTab
        jobs={[trainJob]}
        followingJobId={undefined}
        series={[series("reference_images", [{ step: 0, value: 128 }])]}
      />,
    );

    expect(screen.getByText("Reported once")).toBeTruthy();
    expect(screen.getByText("reference_images")).toBeTruthy();
    expect(screen.getByText("128")).toBeTruthy();
    expect(screen.queryByLabelText(/per step/)).toBeNull();
  });

  it("says a run recorded nothing rather than drawing an empty frame", () => {
    render(<TrainingTab jobs={[trainJob]} followingJobId={undefined} series={[]} />);
    expect(screen.getByText(/recorded no scalar series/)).toBeTruthy();
  });

  it("says nothing has trained when nothing has", () => {
    render(<TrainingTab jobs={[]} followingJobId={undefined} series={[]} />);
    expect(screen.getByText("Nothing has trained yet.")).toBeTruthy();
  });

  it("discloses a downsampled series rather than implying it drew every point", () => {
    render(
      <TrainingTab
        jobs={[trainJob]}
        followingJobId={undefined}
        series={[
          {
            name: "loss_st",
            points: [
              { step: 0, value: 1 },
              { step: 20000, value: 0.1 },
            ],
            total: 1000,
            dropped: 998,
          },
        ]}
      />,
    );

    expect(screen.getByText(/loss_st \(2 of 1000\)/)).toBeTruthy();
  });
});
