/**
 * One catalogue, two mounts, and the chrome is what differs.
 *
 * `/experiments` is a screen and needs a heading of its own; the dataset tab is under a band
 * that already carries the dataset's name and its New-experiment button, so a heading there
 * would be a second `<h1>` restating the tab you just clicked.
 */

import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { DatasetExperimentsRoute, ExperimentsRoute } from "./ExperimentsRoute";
import { withProviders } from "../test-harness";

describe("the experiment catalogue", () => {
  it("names itself when it is the whole screen", () => {
    const { container } = render(
      withProviders(
        <MemoryRouter initialEntries={["/experiments"]}>
          <ExperimentsRoute />
        </MemoryRouter>,
      ),
    );

    expect(container.querySelectorAll("h1")).toHaveLength(1);
    expect(container.querySelector("h1")?.textContent).toBe("Experiments");
    expect(container.querySelectorAll('[data-scroll="tab"]')).toHaveLength(0);
  });

  it("contributes only its scroll region when it is a tab", () => {
    const { container } = render(
      withProviders(
        <MemoryRouter initialEntries={["/datasets/7/experiments"]}>
          <Routes>
            <Route path="datasets/:datasetId/experiments" element={<DatasetExperimentsRoute />} />
          </Routes>
        </MemoryRouter>,
      ),
    );

    expect(container.querySelectorAll("h1")).toHaveLength(0);
    expect(container.querySelectorAll('nav[aria-label="Dataset workspace"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-scroll="tab"]')).toHaveLength(1);
  });
});
