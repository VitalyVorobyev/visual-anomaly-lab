import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { CanvasLayout, ReadingLayout, WorkspaceLayout } from "./App";

function renderLayout(layout: React.ReactNode) {
  return render(
    <MemoryRouter>
      <Routes>
        <Route element={layout}>
          <Route index element={<div>route content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("screen layout scroll ownership", () => {
  it("gives document-like screens one outer scroll region", () => {
    const { container } = renderLayout(<ReadingLayout />);
    const layout = container.querySelector('[data-layout="reading"]');

    expect(layout?.className).toContain("overflow-y-auto");
    expect(layout?.textContent).toContain("route content");
  });

  it("keeps a workspace fixed so the route can own the data scroller", () => {
    const { container } = renderLayout(<WorkspaceLayout />);
    const layout = container.querySelector('[data-layout="workspace"]');

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
