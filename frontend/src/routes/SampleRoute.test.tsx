/**
 * The viewer's contract: the image gets the window, and the editor is one click away.
 *
 * The scroll assertion is the one worth keeping. This screen is a canvas route with a rail
 * beside it, and the rail is the *only* thing on it allowed to scroll — the moment the
 * image region grows a scroller, the picture has stopped filling the window and has gone
 * back to being a box inside a page.
 */

import { fireEvent, render, screen } from "@testing-library/react";
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

function renderViewer(
  over: { page?: unknown; sample?: unknown; dataset?: unknown; truth?: unknown } = {},
) {
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
        ...(over.dataset ? [[queryKeys.dataset(7), over.dataset] as [readonly unknown[], unknown]] : []),
        ...(over.truth
          ? [[queryKeys.imageTruth(IMAGE.id), over.truth] as [readonly unknown[], unknown]]
          : []),
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

  it("offers the label rail on a dataset of classes rather than showing it", () => {
    renderViewer({
      sample: { ...SAMPLE, label: "unlabeled" },
      dataset: { id: 7, name: "FSS-1000", truth: ["classes"], class_counts: [] },
    });

    // ADR-0041: normal and defect are anomaly truth, which this dataset does not hold.
    expect(screen.queryByRole("button", { name: /^normal/ })).toBeNull();
    const optIn = screen.getByRole("button", { name: "Label for anomaly detection" });

    fireEvent.click(optIn);
    expect(screen.getByRole("button", { name: /^normal/ })).toBeTruthy();
  });

  it("draws the image's truth by default, with a legend, and hides it on request", () => {
    const { container } = renderViewer({
      truth: {
        image_id: IMAGE.id,
        source: "revision",
        revision_id: 4,
        classes: [{ key: "missing_hole", name: "Missing hole", color: "#1c7ed6" }],
        boxes: [
          { label_key: "missing_hole", x: 10, y: 10, width: 40, height: 30 },
          { label_key: "missing_hole", x: 200, y: 90, width: 40, height: 30 },
        ],
        regions_url: null,
        outline: false,
      },
    });

    const legend = screen.getByLabelText("Truth legend");
    expect(legend.textContent).toContain("Missing hole");
    expect(legend.textContent).toContain("2 boxes");
    expect(container.querySelectorAll("[data-shape=box]")).toHaveLength(2);
    expect(screen.getByLabelText("Truth opacity")).toBeTruthy();

    fireEvent.click(screen.getByRole("switch", { name: /Truth/ }));
    expect(container.querySelectorAll("[data-shape=box]")).toHaveLength(0);
    expect(screen.queryByLabelText("Truth opacity")).toBeNull();
  });

  it("shows the label rail on an anomaly dataset", () => {
    renderViewer({ dataset: { id: 7, name: "candle", truth: ["labels"], class_counts: [] } });
    expect(screen.getByRole("button", { name: /^defect/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Label for anomaly detection" })).toBeNull();
  });
});
