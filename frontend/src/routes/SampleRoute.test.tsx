/**
 * The viewer's contract: the image gets the window, and the editor is one click away.
 *
 * The scroll assertion is the one worth keeping. This screen is a canvas route with a rail
 * beside it, and the rail is the *only* thing on it allowed to scroll — the moment the
 * image region grows a scroller, the picture has stopped filling the window and has gone
 * back to being a box inside a page.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { SampleRoute } from "./SampleRoute";

const IMAGE = {
  id: 501,
  channel: "bright",
  channel_id: 1,
  width: 1280,
  height: 1024,
  bit_depth: 24,
  file_size: 2_400_000,
  path: "/roots/fixture/set1/bright/12.png",
};

const SAMPLE = {
  id: 12,
  dataset_id: 7,
  group_key: "set1/no-defect",
  external_id: "12",
  label: "normal",
  label_source: "import",
  notes: null,
  images: [IMAGE],
};

const PAGE = {
  total: 3,
  limit: 200,
  offset: 0,
  items: [
    { ...SAMPLE, id: 11, external_id: "11" },
    SAMPLE,
    { ...SAMPLE, id: 13, external_id: "13" },
  ],
};

function renderViewer(over: { page?: unknown; sample?: unknown } = {}) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/datasets/7/samples/12"]}>
        <Routes>
          <Route path="datasets/:datasetId/samples/:sampleId" element={<SampleRoute />} />
        </Routes>
      </MemoryRouter>,
      [
        [queryKeys.sample(7, 12), over.sample ?? SAMPLE],
        [queryKeys.samples(7, { limit: 200, offset: 0 }), over.page ?? PAGE],
      ],
    ),
  );
}

describe("the sample viewer", () => {
  it("opens the annotation editor on the image being shown", () => {
    renderViewer();

    const link = screen.getByRole("link", { name: /Open in editor/ });
    expect(link.getAttribute("href")).toBe("/datasets/7/annotate/12/501");
  });

  it("gives the rail the only scroller on the screen", () => {
    const { container } = renderViewer();

    const scrollers = [...container.querySelectorAll("*")].filter((node) =>
      node.className.toString().includes("overflow-y-auto"),
    );

    expect(scrollers).toHaveLength(1);
    expect(scrollers[0]!.getAttribute("data-scroll")).toBe("rail");
  });

  it("pages within the filtered set the grid was showing", () => {
    renderViewer();

    expect(screen.getByText("2 of 3")).toBeTruthy();
    expect(screen.getByLabelText("Previous sample").hasAttribute("disabled")).toBe(false);
    expect(screen.getByLabelText("Next sample").hasAttribute("disabled")).toBe(false);
  });

  it("deadens both arrows once the sample has left its own filtered set", () => {
    // Labelling under `label=unlabeled` is the ordinary way to get here. The back arrow
    // used to stay enabled and do nothing, because its disabled test only asked about the
    // offset and never about whether the sample was still in the page.
    renderViewer({ page: { ...PAGE, items: PAGE.items.filter((item) => item.id !== 12) } });

    expect(screen.getByText(/off the filtered set/)).toBeTruthy();
    expect(screen.getByLabelText("Previous sample").hasAttribute("disabled")).toBe(true);
    expect(screen.getByLabelText("Next sample").hasAttribute("disabled")).toBe(true);
  });

  it("keeps the file path off the screen and behind a disclosure", () => {
    const { container } = renderViewer();

    const disclosure = container.querySelector("details")!;
    expect(disclosure.open).toBe(false);
    expect(disclosure.textContent).toContain(IMAGE.path);
  });

  it("hides the channel controls when a sample is one image", () => {
    renderViewer();
    expect(screen.queryByRole("button", { name: /Side by side/ })).toBeNull();

    screen.getByRole("button", { name: /Reset view/ });
  });
});
