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

function renderAt(
  search: string,
  built = true,
  extra: [readonly unknown[], unknown][] = [],
) {
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
        [
          queryKeys.samples(7, { classKey: "scratch", presence: "absent", limit: 48, offset: 0 }),
          { total: 1, limit: 48, offset: 0, items: [sample(3)] },
        ],
        ...extra,
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

  it("reads the preview of the open image beside the stage", () => {
    renderAt("?refs=1&focus=1", true, [
      [
        ["studio-preview", 7, "scratch", "fss_dino", 11, [1], 101],
        {
          image_id: 101,
          score: 0.9312,
          foreground_share: 0.125,
          generation: "abc",
          map_url: "/api/studio/previews/abc/101.png",
          warm: true,
          elapsed_ms: 84,
        },
      ],
    ]);
    expect(screen.getByText("0.931")).toBeTruthy();
    expect(screen.getByText("12.5%")).toBeTruthy();
    expect(screen.getByText("84 ms")).toBeTruthy();
    const accept = screen.getByRole("button", { name: "Accept the preview as truth" });
    expect(accept.hasAttribute("disabled")).toBe(false);
    expect(screen.getByRole("button", { name: "Mark Scratch absent" })).toBeTruthy();
  });

  it("cannot accept a preview it has not drawn", () => {
    renderAt("?focus=1");
    expect(
      screen.getByRole("button", { name: "Accept the preview as truth" }).hasAttribute("disabled"),
    ).toBe(true);
    // Confirming absence needs no preview: it is a statement about the image.
    expect(
      screen.getByRole("button", { name: "Mark Scratch absent" }).hasAttribute("disabled"),
    ).toBe(false);
  });

  it("lists samples without the class for looking, never for teaching", () => {
    renderAt("?show=absent");
    expect(screen.getByRole("heading", { name: "Samples without it" })).toBeTruthy();
    expect(screen.getByTitle("candle/003")).toBeTruthy();
    expect(screen.queryByRole("checkbox", { name: /as a reference/ })).toBeNull();
  });

  it("orders a page least certain first under the current references", () => {
    const { container } = renderAt("?refs=1&order=uncertain", true, [
      [
        ["studio-batch", 7, "scratch", "fss_dino", 11, [1], [101, 102]],
        {
          generation: "abc",
          warm: true,
          elapsed_ms: 120,
          results: [
            { image_id: 102, score: 0.5, foreground_share: 0.1, uncertainty: 1 },
            { image_id: 101, score: 0.95, foreground_share: 0.4, uncertainty: 0.1 },
          ],
        },
      ],
    ]);
    const tiles = [...container.querySelectorAll('aside[aria-label="Samples that show the class"] button[title]')];
    expect(tiles.map((tile) => tile.getAttribute("title"))).toEqual(["candle/002", "candle/001"]);
    expect(screen.getByText(/least certain first under the current references/)).toBeTruthy();
  });
});
