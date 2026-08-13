/**
 * One scroller per tab, and nothing scrolling inside it.
 *
 * `docs/architecture/frontend.md` states the rule; this is the rule made executable. The
 * third assertion is the one that matters: a `max-h-* overflow-y-auto` pane stacked inside
 * a page that already scrolls is exactly how two scrollbars ended up on screen together.
 */

import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { AnnotationQueueRoute } from "../AnnotationQueueRoute";
import { DatasetRoute } from "../DatasetRoute";
import { DatasetExperimentsRoute } from "../ExperimentsRoute";
import { RegionPreparationRoute } from "../RegionPreparationRoute";
import { SplitsRoute } from "../SplitsRoute";
import { withProviders } from "../../test-harness";

const TABS: [string, () => React.JSX.Element][] = [
  ["Browse", DatasetRoute],
  ["Annotate", AnnotationQueueRoute],
  ["Prepare", RegionPreparationRoute],
  ["Splits", SplitsRoute],
  ["Experiments", DatasetExperimentsRoute],
];

describe.each(TABS)("the %s tab", (_name, Tab) => {
  function renderTab() {
    return render(
      withProviders(
        <MemoryRouter initialEntries={["/datasets/7"]}>
          <Routes>
            <Route path="datasets/:datasetId" element={<Tab />} />
          </Routes>
        </MemoryRouter>,
      ),
    );
  }

  it("declares exactly one scroll region", () => {
    const { container } = renderTab();

    expect(container.querySelectorAll('[data-scroll="tab"]')).toHaveLength(1);
  });

  it("makes that region the one that scrolls", () => {
    const { container } = renderTab();
    const scroller = container.querySelector('[data-scroll="tab"]')!;

    expect(scroller.className).toContain("overflow-y-auto");
  });

  it("puts no same-axis scroller inside it", () => {
    const { container } = renderTab();
    const scroller = container.querySelector('[data-scroll="tab"]')!;

    const nested = [...scroller.querySelectorAll("*")].filter((element) =>
      String(element.className).includes("overflow-y-auto"),
    );

    expect(nested).toHaveLength(0);
  });

  it("renders no page heading of its own", () => {
    const { container } = renderTab();

    // The band above is the only `<h1>` on a dataset screen, and the tab strip is the only
    // copy of the strip. A tab that grows either is how the drift started.
    expect(container.querySelectorAll("h1")).toHaveLength(0);
    expect(container.querySelectorAll('nav[aria-label="Dataset workspace"]')).toHaveLength(0);
  });

  it("offers no back link to the browser the strip already reaches", () => {
    const { container } = renderTab();
    const back = [...container.querySelectorAll("a")].filter((link) =>
      link.textContent?.trimStart().startsWith("←"),
    );

    expect(back.map((link) => link.getAttribute("href"))).not.toContain("/datasets/7");
  });
});
