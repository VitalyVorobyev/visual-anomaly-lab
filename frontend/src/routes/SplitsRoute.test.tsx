/**
 * The split form opens on the strategy a prerequisite link asked for, and says what that
 * strategy does with the samples it cannot draw.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { SplitsRoute } from "./SplitsRoute";

function renderAt(url: string) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[url]}>
        <Routes>
          <Route path="datasets/:datasetId/splits" element={<SplitsRoute />} />
        </Routes>
      </MemoryRouter>,
      [[queryKeys.splits(7), []]],
    ),
  );
}

describe("the split form", () => {
  it("opens on a draw of annotated samples by class when a segmentation run asks for one", () => {
    renderAt("/datasets/7/splits?strategy=class_stratified");
    expect(screen.getByRole("combobox", { name: "Strategy" }).textContent).toContain(
      "Draw annotated samples by class",
    );
    expect(screen.getByText("Annotated samples used for training")).toBeTruthy();
    expect(screen.getByText(/Samples without a full annotation go to test/)).toBeTruthy();
  });

  it("ignores a strategy it does not know and opens on the ordinary draw", () => {
    renderAt("/datasets/7/splits?strategy=nonsense");
    expect(screen.getByRole("combobox", { name: "Strategy" }).textContent).toContain("Draw one");
    expect(screen.queryByText("Annotated samples used for training")).toBeNull();
  });
});
