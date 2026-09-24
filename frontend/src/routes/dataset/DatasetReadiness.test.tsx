import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../api/queryKeys";
import { withProviders } from "../../test-harness";
import { DatasetReadiness } from "./DatasetReadiness";

function renderWith(seed: [readonly unknown[], unknown][]) {
  return render(
    withProviders(
      <MemoryRouter>
        <DatasetReadiness datasetId={7} />
      </MemoryRouter>,
      seed,
    ),
  );
}

describe("dataset readiness", () => {
  it("says nothing until it knows", () => {
    const { container } = renderWith([]);
    expect(container.textContent).toBe("");
  });

  it("names the missing steps in order, each a link to where it is done", () => {
    renderWith([
      [queryKeys.regionProfiles(7), []],
      [queryKeys.splits(7), []],
      [queryKeys.experiments({ datasetId: 7 }), []],
    ]);
    const steps = screen.getAllByRole("link");
    expect(steps.map((step) => step.textContent)).toEqual([
      "1. Build a region profile",
      "2. Make a split",
    ]);
    expect(steps[0]?.getAttribute("href")).toBe("/datasets/7/prepare");
    expect(steps[1]?.getAttribute("href")).toBe("/datasets/7/splits");
  });

  it("does not count a build with failed images as prepared", () => {
    renderWith([
      [queryKeys.regionProfiles(7), [{ id: 11 }]],
      [queryKeys.regionBuild(11), { failed: 2, succeeded: 8, total: 10 }],
      [queryKeys.splits(7), [{ id: 3 }]],
      [queryKeys.experiments({ datasetId: 7 }), []],
    ]);
    expect(screen.getByRole("link").textContent).toBe("1. Build a region profile");
  });

  it("says a ready dataset is ready, and how many runs it has", () => {
    renderWith([
      [queryKeys.regionProfiles(7), [{ id: 11 }]],
      [queryKeys.regionBuild(11), { failed: 0, succeeded: 10, total: 10 }],
      [queryKeys.splits(7), [{ id: 3 }]],
      [queryKeys.experiments({ datasetId: 7 }), [{ id: 1 }, { id: 2 }]],
    ]);
    expect(screen.getByText(/Ready to train/)).toBeTruthy();
    expect(screen.getByRole("link").textContent).toBe("2 runs");
  });
});
