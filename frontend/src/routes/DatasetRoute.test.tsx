/**
 * The browse filters follow the dataset's truth (ADR-0041): a dataset of classes is filtered
 * by class and offered no label filter or bulk labelling; an anomaly dataset keeps both.
 */

import { render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { DatasetRoute } from "./DatasetRoute";

const SAMPLE = {
  id: 1,
  dataset_id: 7,
  group_key: "bucket",
  external_id: "1",
  label: "unlabeled",
  label_source: "import",
  notes: null,
  images: [],
  annotation: "complete",
};

function detail(truth: string[]) {
  return {
    id: 7,
    name: "panel",
    root_path: "/roots/panel",
    adapter: "folder_classes",
    created_at: "2026-09-01T00:00:00Z",
    notes: null,
    samples: 1,
    images: 1,
    label_counts: { normal: 0, defect: 0, unlabeled: 1 },
    truth,
    class_counts: truth.includes("classes")
      ? [{ key: "bucket", name: "Bucket", color: "#1c7ed6", samples: 10 }]
      : [],
    collection: null,
    description: null,
    cover_image_id: null,
    default_channel: null,
    manifest_path: null,
    channels: [],
    group_keys: [],
    splits: 0,
    annotation_scope: "image",
  };
}

function renderBrowse(truth: string[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/datasets/7"]}>
        <Routes>
          <Route path="datasets/:datasetId" element={<DatasetRoute />} />
        </Routes>
      </MemoryRouter>,
      [
        [queryKeys.dataset(7), detail(truth)],
        [queryKeys.splits(7), []],
        [queryKeys.samples(7, { limit: 200, offset: 0 }), { total: 1, limit: 200, offset: 0, items: [SAMPLE] }],
      ],
    ),
  );
}

describe("the browse filters", () => {
  it("filter a dataset of classes by class, with no label filter or bulk labelling", () => {
    renderBrowse(["classes"]);
    const rail = screen.getByRole("complementary", { name: "Dataset filters" });
    expect(within(rail).getByRole("combobox", { name: "Class" })).toBeTruthy();
    expect(within(rail).queryByRole("combobox", { name: "Label" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Label all/ })).toBeNull();
  });

  it("filter an anomaly dataset by label, and offer to label what matches", () => {
    renderBrowse(["labels"]);
    const rail = screen.getByRole("complementary", { name: "Dataset filters" });
    expect(within(rail).getByRole("combobox", { name: "Label" })).toBeTruthy();
    expect(within(rail).queryByRole("combobox", { name: "Class" })).toBeNull();
    expect(screen.getByRole("button", { name: /Label all 1/ })).toBeTruthy();
  });
});
