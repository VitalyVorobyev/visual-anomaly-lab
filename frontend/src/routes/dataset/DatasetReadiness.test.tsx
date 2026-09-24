import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../api/queryKeys";
import { withProviders } from "../../test-harness";
import { DatasetReadiness } from "./DatasetReadiness";

function method(key: string, tasks: string[]) {
  return { key, capabilities: { tasks } };
}

const ANOMALY_ONLY: [readonly unknown[], unknown] = [
  queryKeys.modelTypes(),
  { methods: [method("pixel_reference", ["anomaly"])] },
];
const BOTH: [readonly unknown[], unknown] = [
  queryKeys.modelTypes(),
  {
    methods: [
      method("pixel_reference", ["anomaly"]),
      method("color_prototype", ["few_shot_segmentation"]),
    ],
  },
];

const WITH_SEGMENT: [readonly unknown[], unknown] = [
  queryKeys.modelTypes(),
  {
    methods: [
      method("pixel_reference", ["anomaly"]),
      method("color_classifier", ["semantic_segmentation"]),
    ],
  },
];

function renderWith(seed: [readonly unknown[], unknown][], catalog = ANOMALY_ONLY) {
  return render(
    withProviders(
      <MemoryRouter>
        <DatasetReadiness datasetId={7} />
      </MemoryRouter>,
      [catalog, ...seed],
    ),
  );
}

const BUILT: [readonly unknown[], unknown][] = [
  [queryKeys.regionProfiles(7), [{ id: 11 }]],
  [queryKeys.regionBuild(11), { failed: 0, succeeded: 10, total: 10 }],
  [queryKeys.experiments({ datasetId: 7 }), { items: [], total: 0, next_cursor: null }],
];

describe("dataset readiness", () => {
  it("says nothing until it knows", () => {
    const { container } = render(
      withProviders(
        <MemoryRouter>
          <DatasetReadiness datasetId={7} />
        </MemoryRouter>,
        [],
      ),
    );
    expect(container.textContent).toBe("");
  });

  it("names the missing steps in order, each a link to where it is done", () => {
    renderWith([
      [queryKeys.regionProfiles(7), []],
      [queryKeys.splits(7), []],
      [queryKeys.experiments({ datasetId: 7 }), { items: [], total: 0, next_cursor: null }],
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
      [queryKeys.splits(7), [{ id: 3, strategy: "imported" }]],
      [queryKeys.experiments({ datasetId: 7 }), { items: [], total: 0, next_cursor: null }],
    ]);
    expect(screen.getByRole("link").textContent).toBe("1. Build a region profile");
  });

  it("says a ready dataset is ready, and how many runs it has", () => {
    renderWith([
      [queryKeys.regionProfiles(7), [{ id: 11 }]],
      [queryKeys.regionBuild(11), { failed: 0, succeeded: 10, total: 10 }],
      [queryKeys.splits(7), [{ id: 3, strategy: "normal_only_train" }]],
      [queryKeys.experiments({ datasetId: 7 }), { items: [{ id: 1 }, { id: 2 }], total: 2, next_cursor: null }],
    ]);
    expect(screen.getByText(/Ready to train/)).toBeTruthy();
    expect(screen.getByRole("link").textContent).toBe("2 runs");
  });

  it("with two tasks, puts the shared first step first and nothing else", () => {
    renderWith(
      [
        [queryKeys.regionProfiles(7), []],
        [queryKeys.splits(7), []],
        [queryKeys.experiments({ datasetId: 7 }), { items: [], total: 0, next_cursor: null }],
        [queryKeys.classCoverage(7), []],
      ],
      BOTH,
    );
    expect(screen.getAllByRole("link").map((step) => step.textContent)).toEqual([
      "1. Build a region profile",
    ]);
  });

  it("then says per task whether it is ready, or what it needs next", () => {
    renderWith(
      [
        ...BUILT,
        [queryKeys.splits(7), [{ id: 3, strategy: "imported" }]],
        [queryKeys.classCoverage(7), [{ label_key: "scratch", present: 0, absent: 5, unlabeled: 0 }]],
      ],
      BOTH,
    );
    const band = screen.getByRole("navigation", { name: "Readiness by task" });
    expect(band.textContent).toContain("Anomaly");
    const next = screen.getByRole("link", { name: /Annotate a class/ });
    expect(next.getAttribute("href")).toBe("/datasets/7/annotate");
  });

  it("asks for references once a class can supply them", () => {
    renderWith(
      [
        ...BUILT,
        [queryKeys.splits(7), [{ id: 3, strategy: "imported" }]],
        [queryKeys.classCoverage(7), [{ label_key: "scratch", present: 3, absent: 5, unlabeled: 0 }]],
      ],
      BOTH,
    );
    expect(screen.getByRole("link", { name: /Choose references/ }).getAttribute("href")).toBe(
      "/datasets/7/splits",
    );
  });

  it("names supervised segmentation beside anomaly, waiting on annotation, then a split", () => {
    const annotated = [
      [queryKeys.classCoverage(7), [{ label_key: "scratch", present: 3, absent: 5, unlabeled: 0 }]],
    ] as [readonly unknown[], unknown][];
    const { unmount } = renderWith(
      [
        ...BUILT,
        [queryKeys.splits(7), [{ id: 3, strategy: "imported" }]],
        [queryKeys.classCoverage(7), [{ label_key: "scratch", present: 0, absent: 5, unlabeled: 0 }]],
      ],
      WITH_SEGMENT,
    );
    const band = screen.getByRole("navigation", { name: "Readiness by task" });
    expect(band.textContent).toContain("Segment:");
    expect(screen.getByRole("link", { name: /Annotate a class/ })).toBeTruthy();
    unmount();

    // Annotated, but only an anomaly split: a supervised run needs one of annotated samples.
    const second = renderWith(
      [...BUILT, [queryKeys.splits(7), [{ id: 3, strategy: "imported" }]], ...annotated],
      WITH_SEGMENT,
    );
    expect(screen.getByRole("link", { name: /Make a split/ })).toBeTruthy();
    second.unmount();

    renderWith(
      [
        ...BUILT,
        [
          queryKeys.splits(7),
          [
            { id: 3, strategy: "imported" },
            { id: 4, strategy: "class_stratified" },
          ],
        ],
        ...annotated,
      ],
      WITH_SEGMENT,
    );
    const ready = screen.getByRole("navigation", { name: "Readiness by task" });
    expect(ready.textContent).toContain("Segment");
    expect(ready.textContent).not.toContain("Segment:");
    expect(screen.queryByRole("link", { name: /Make a split/ })).toBeNull();
  });
});
