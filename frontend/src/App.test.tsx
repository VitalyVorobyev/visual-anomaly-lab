import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { CanvasLayout, ReadingLayout } from "./App";
import { DatasetLayout } from "./routes/dataset/DatasetLayout";
import { withProviders } from "./test-harness";

function renderLayout(layout: React.ReactNode) {
  return render(
    withProviders(
      <MemoryRouter>
        <Routes>
          <Route element={layout}>
            <Route index element={<div>route content</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    ),
  );
}

describe("screen layout scroll ownership", () => {
  it("gives document-like screens one outer scroll region", () => {
    const { container } = renderLayout(<ReadingLayout />);
    const layout = container.querySelector('[data-layout="reading"]');

    expect(layout?.className).toContain("overflow-y-auto");
    expect(layout?.textContent).toContain("route content");
  });

  it("keeps a dataset workspace fixed so the tab can own the data scroller", () => {
    const { container } = renderLayout(<DatasetLayout />);
    const layout = container.querySelector('[data-layout="dataset"]');

    expect(layout?.className).toContain("overflow-hidden");
    expect(layout?.className).not.toContain("overflow-y-auto");
  });

  it("keeps an image canvas fixed too", () => {
    const { container } = renderLayout(<CanvasLayout />);
    const layout = container.querySelector('[data-layout="canvas"]');

    expect(layout?.className).toContain("overflow-hidden");
    expect(layout?.className).not.toContain("overflow-y-auto");
  });
});
