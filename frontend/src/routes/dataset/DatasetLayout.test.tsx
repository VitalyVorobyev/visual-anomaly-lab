/**
 * The band is the fix, so the band is what these assert.
 *
 * Every one of these renders with the dataset query still pending, which is the state the
 * old screens got wrong: three of the five tabs returned a skeleton *before* their header,
 * so the strip vanished on load and popped back in, and two rendered their `meta` line only
 * once the query settled, dropping the strip by a line when it did.
 */

import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../api/queryKeys";
import { withProviders } from "../../test-harness";
import { DatasetLayout } from "./DatasetLayout";

const TABS = [
  "/datasets/7",
  "/datasets/7/annotate",
  "/datasets/7/prepare",
  "/datasets/7/splits",
  "/datasets/7/experiments",
];

function detail(over: Record<string, unknown> = {}) {
  return {
    id: 7,
    name: "candle",
    root_path: "/datasets/VisA_20220922/candle",
    adapter: "csv_table",
    created_at: "2026-08-13T09:00:00.000Z",
    notes: null,
    samples: 1100,
    images: 1100,
    label_counts: { normal: 1000, defect: 100, unlabeled: 0 },
    collection: "VisA",
    description: null,
    cover_image_id: 3,
    manifest_path: null,
    channels: [],
    group_keys: [],
    splits: 0,
    ...over,
  };
}

/** Stub children: this is about the band, not about what any one tab draws. */
function renderAt(path: string, loaded?: Record<string, unknown>) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="datasets/:datasetId" element={<DatasetLayout />}>
            <Route index element={<div>browse</div>} />
            <Route path="annotate" element={<div>annotate</div>} />
            <Route path="prepare" element={<div>prepare</div>} />
            <Route path="splits" element={<div>splits</div>} />
            <Route path="experiments" element={<div>experiments</div>} />
            <Route path="experiments/new" element={<div>new</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
      loaded ? [[queryKeys.dataset(7), loaded]] : [],
    ),
  );
}

function nav(container: HTMLElement) {
  return container.querySelector('nav[aria-label="Dataset workspace"]');
}

describe("the dataset band", () => {
  it("shows the section strip before the dataset has loaded", () => {
    const { container } = renderAt("/datasets/7");

    expect(nav(container)?.querySelectorAll("a")).toHaveLength(5);
  });

  it("is rendered exactly once on every tab", () => {
    for (const path of TABS) {
      const { container, unmount } = renderAt(path);

      // Five copies of this collapsing into one is the whole change.
      expect(container.querySelectorAll("h1")).toHaveLength(1);
      expect(
        container.querySelectorAll('nav[aria-label="Dataset workspace"]'),
      ).toHaveLength(1);

      unmount();
    }
  });

  it("does not change shape between tabs", () => {
    const shapes = TABS.map((path) => {
      const { container, unmount } = renderAt(path);
      const band = container.querySelector('[data-band="dataset"]')!;
      const shape = `${band.className}|${band.firstElementChild!.className}`;
      unmount();
      return shape;
    });

    // The band's height is now a function of its classes alone, so identical classes on
    // every tab is a real proxy for "the strip below it does not move".
    expect(new Set(shapes).size).toBe(1);
  });

  it("marks the tab the URL is on, and only that one", () => {
    const { container } = renderAt("/datasets/7/splits");
    const current = nav(container)!.querySelectorAll('[aria-current="page"]');

    expect(current).toHaveLength(1);
    expect(current[0]!.textContent).toBe("Splits");
  });

  it("keeps Experiments marked on its own sub-page", () => {
    const { container } = renderAt("/datasets/7/experiments/new");
    const current = nav(container)!.querySelector('[aria-current="page"]');

    expect(current?.textContent).toBe("Experiments");
  });

  it("counts in samples, and does not spend a line on the path", () => {
    const { container } = renderAt("/datasets/7", detail());
    const band = container.querySelector('[data-band="dataset"]')!;

    expect(band.textContent).toContain("1100 samples");
    expect(band.textContent).toContain("1000 normal");
    expect(band.textContent).toContain("100 defect");
    // Two denominators for one dataset, and an absolute path as permanent furniture.
    // `label_counts` is stored per sample (ADR-0005), so samples is the honest unit.
    expect(band.textContent).not.toContain("images");
    expect(band.textContent).not.toContain("/datasets/VisA_20220922");
    expect(band.textContent).not.toContain("csv_table");
  });

  it("names the channel count only when a sample is more than one image", () => {
    const single = renderAt("/datasets/7", detail());
    expect(single.container.textContent).not.toContain("channels");
    single.unmount();

    const many = renderAt(
      "/datasets/7",
      detail({
        samples: 189,
        images: 567,
        channels: [
          { id: 1, dataset_id: 7, name: "bright", position: 0 },
          { id: 2, dataset_id: 7, name: "dark", position: 1 },
          { id: 3, dataset_id: 7, name: "dome", position: 2 },
        ],
      }),
    );
    expect(many.container.textContent).toContain("189 samples");
    expect(many.container.textContent).toContain("3 channels");
  });

  it("keeps its shape whether or not the dataset has a description", () => {
    const shapes = [detail(), detail({ description: "A tea light, four to a frame." })].map(
      (data) => {
        const { container, unmount } = renderAt("/datasets/7", data);
        const band = container.querySelector('[data-band="dataset"]')!;
        const shape = `${band.className}|${band.firstElementChild!.className}`;
        unmount();
        return shape;
      },
    );

    expect(new Set(shapes).size).toBe(1);
  });

  it("offers the same one action from every tab", () => {
    for (const path of TABS) {
      const { container, unmount } = renderAt(path);
      const action = container.querySelector('[data-band="dataset"] a[href$="/experiments/new"]');

      expect(action?.textContent).toContain("New experiment");
      unmount();
    }
  });
});
