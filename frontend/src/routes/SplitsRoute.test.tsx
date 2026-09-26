/**
 * The splits screen leads with the presets this dataset can serve, keeps the strategy form
 * folded away under "Custom split" with only the strategies its truth can feed, and reports
 * each split's task, composition and the runs holding it.
 */

import { render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { SplitsRoute } from "./SplitsRoute";

type Seed = [readonly unknown[], unknown][];

const ROWS = (train: number, test: number) => [
  { subset: "train", total: train, normal: train, defect: 0, unlabeled: 0, classes: [] },
  { subset: "val", total: 0, normal: 0, defect: 0, unlabeled: 0, classes: [] },
  { subset: "test", total: test, normal: test - 2, defect: 2, unlabeled: 0, classes: [] },
];

const ANOMALY_DATASET = { id: 7, truth: ["labels"], label_counts: {}, class_counts: [] };
const CLASS_DATASET = {
  id: 7,
  truth: ["classes"],
  label_counts: {},
  class_counts: [
    { key: "bucket", name: "Bucket", color: "#aa0000", samples: 10 },
    { key: "cup", name: "Cup", color: "#00aa00", samples: 6 },
  ],
};

const STANDARD = {
  key: "standard",
  label: "Standard · 60/20/20, normals only",
  meaning: "Train on 60% of the normals.",
  tasks: ["anomaly"],
  params: { strategy: "normal_only_train" },
  seed: 0,
  name: "Standard · 60/20/20, normals only · seed 0",
  composition: ROWS(6, 6),
  classes: [],
};

function renderAt(url: string, seed: Seed) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[url]}>
        <Routes>
          <Route path="datasets/:datasetId/splits" element={<SplitsRoute />} />
        </Routes>
      </MemoryRouter>,
      seed,
    ),
  );
}

describe("the splits screen", () => {
  it("offers a preset with its task, its composition and one press to create it", () => {
    renderAt("/datasets/7/splits", [
      [queryKeys.dataset(7), ANOMALY_DATASET],
      [queryKeys.splits(7), []],
      [queryKeys.splitPresets(7), [STANDARD]],
    ]);
    const card = screen.getByRole("article", { name: STANDARD.label });
    expect(within(card).getByText("Anomaly")).toBeTruthy();
    expect(within(card).getByRole("img", { name: "Composition: train 6, test 6" })).toBeTruthy();
    expect(within(card).getByText(STANDARD.name)).toBeTruthy();
    expect(within(card).getByRole("button", { name: "Create" })).toBeTruthy();
    expect(screen.getByText(/Create one from a preset above/)).toBeTruthy();
  });

  it("offers a few-shot preset with a class picker on its most frequent class", () => {
    renderAt("/datasets/7/splits", [
      [queryKeys.dataset(7), CLASS_DATASET],
      [queryKeys.splits(7), []],
      [
        queryKeys.splitPresets(7),
        [
          {
            ...STANDARD,
            key: "few_shot_1",
            label: "1-shot",
            tasks: ["few_shot_segmentation"],
            params: { strategy: "few_shot", label_key: "bucket", shots: 1 },
            name: "1-shot · bucket · seed 0",
            classes: [
              { key: "bucket", name: "Bucket", samples: 10 },
              { key: "cup", name: "Cup", samples: 6 },
            ],
          },
        ],
      ],
    ]);
    const card = screen.getByRole("article", { name: "1-shot" });
    expect(within(card).getByRole("combobox", { name: "1-shot class" }).textContent).toContain(
      "Bucket",
    );
    expect(within(card).getByText("Few-shot")).toBeTruthy();
  });

  it("opens the custom form on the strategy a prerequisite link asked for", () => {
    renderAt("/datasets/7/splits?strategy=class_stratified", [
      [queryKeys.dataset(7), CLASS_DATASET],
      [queryKeys.splits(7), []],
      [queryKeys.splitPresets(7), []],
    ]);
    expect(screen.getByRole("combobox", { name: "Strategy" }).textContent).toContain(
      "Draw annotated samples by class",
    );
    expect(screen.getByText("Annotated samples used for training")).toBeTruthy();
    expect(screen.getByText(/Samples without a full annotation go to test/)).toBeTruthy();
  });

  it("offers only the strategies the dataset's truth can feed, each with its task", () => {
    renderAt("/datasets/7/splits?strategy=nonsense", [
      [queryKeys.dataset(7), ANOMALY_DATASET],
      [queryKeys.splits(7), []],
      [queryKeys.splitPresets(7), [STANDARD]],
    ]);
    // An unknown strategy opens nothing; the form stays folded on the first one it can draw.
    const custom = screen.getByText("Custom split").closest("details");
    expect(custom?.open).toBe(false);
    const strategy = screen.getByRole("combobox", { name: "Strategy", hidden: true });
    expect(strategy.textContent).toContain("Draw one, normals only in train");
    expect(screen.queryByText("Annotated samples used for training")).toBeNull();
  });

  it("points at labelling or annotation when no preset fits", () => {
    renderAt("/datasets/7/splits", [
      [queryKeys.dataset(7), { id: 7, truth: [], label_counts: {}, class_counts: [] }],
      [queryKeys.splits(7), []],
      [queryKeys.splitPresets(7), []],
    ]);
    expect(screen.getByText("No preset fits this dataset yet")).toBeTruthy();
    expect(screen.queryByText("Custom split")).toBeNull();
  });

  it("reports each split's task, the runs holding it, and offers to delete it", () => {
    renderAt("/datasets/7/splits", [
      [queryKeys.dataset(7), ANOMALY_DATASET],
      [
        queryKeys.splits(7),
        [
          {
            id: 3,
            dataset_id: 7,
            name: "held",
            strategy: "normal_only_train",
            seed: 0,
            created_at: "2026-01-01",
            tasks: ["anomaly"],
            params: { strategy: "normal_only_train", sample_ids: [], classes: [] },
            composition: ROWS(6, 6),
            experiments: [{ experiment_id: 12, name: "patchcore run" }],
          },
        ],
      ],
    ]);
    expect(screen.getByRole("link", { name: "patchcore run" }).getAttribute("href")).toBe(
      "/experiments/12",
    );
    expect(screen.getByText("no defect ✓")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Delete" })).toBeTruthy();
  });
});
