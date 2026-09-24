/**
 * The reference studio holds its session in the URL, shows only samples that show the class,
 * and says in words why it cannot freeze yet.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { StudioRoute } from "./StudioRoute";

function sample(id: number) {
  return {
    id,
    dataset_id: 7,
    group_key: "candle",
    external_id: String(id).padStart(3, "0"),
    label: "defect",
    label_source: "import",
    notes: null,
    annotation_state: "complete",
    images: [{ id: 100 + id, channel_id: null, channel: null, width: 64, height: 64 }],
  };
}

const METHOD = {
  key: "fss_dino",
  title: "FSSDINO (few-shot)",
  capabilities: { tasks: ["few_shot_segmentation"] },
  availability: { available: true, reason: null },
};

function renderAt(search: string, built = true) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[`/datasets/7/studio/scratch${search}`]}>
        <Routes>
          <Route path="datasets/:datasetId/studio/:labelKey" element={<StudioRoute />} />
        </Routes>
      </MemoryRouter>,
      [
        [queryKeys.dataset(7), { id: 7, name: "candle", channels: [], default_channel: null }],
        [
          queryKeys.annotationLabels(7),
          [{ id: 2, dataset_id: 7, key: "scratch", name: "Scratch", color: "#00aa00", position: 1 }],
        ],
        [queryKeys.classCoverage(7), [{ label_key: "scratch", present: 2, absent: 9, unlabeled: 1 }]],
        [
          queryKeys.samples(7, { classKey: "scratch", presence: "present", limit: 48, offset: 0 }),
          { total: 2, limit: 48, offset: 0, items: [sample(1), sample(2)] },
        ],
        [queryKeys.sample(7, 1), sample(1)],
        [queryKeys.sample(7, 2), sample(2)],
        [queryKeys.modelTypes(), { methods: [METHOD] }],
        [queryKeys.regionProfiles(7), [{ id: 11, name: "full frame", revision_no: 1, prepared_width: 448, prepared_height: 448 }]],
        [queryKeys.regionBuild(11), built ? { failed: 0, succeeded: 12, total: 12 } : { failed: 3, succeeded: 9, total: 12 }],
      ],
    ),
  );
}

describe("the reference studio", () => {
  it("offers the samples that show the class, and says why it cannot freeze yet", () => {
    renderAt("");
    expect(screen.getByRole("heading", { name: "Reference studio · Scratch" })).toBeTruthy();
    expect(screen.getByRole("checkbox", { name: "Use candle/001 as a reference" })).toBeTruthy();
    expect(screen.getByText("2 show it")).toBeTruthy();
    const freeze = screen.getByRole("button", { name: "Freeze as experiment" });
    expect(freeze.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText("Choose at least one reference.")).toBeTruthy();
  });

  it("keeps its references in the URL and freezes once everything is in place", () => {
    renderAt("?refs=2,1");
    expect(screen.getByText("2 of 10")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Remove candle/002" })).toBeTruthy();
    const freeze = screen.getByRole("button", { name: "Freeze as experiment" });
    expect(freeze.hasAttribute("disabled")).toBe(false);
    expect(screen.getByText(/Makes a split of these references/)).toBeTruthy();
  });

  it("will not freeze onto a profile whose build failed images", () => {
    renderAt("?refs=1", false);
    expect(screen.getByText("The region profile is not built.")).toBeTruthy();
  });
});
