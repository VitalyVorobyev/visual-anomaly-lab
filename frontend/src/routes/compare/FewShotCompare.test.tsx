/**
 * Few-shot runs of one class read across their reference draws: overlap by method and shot
 * count, every metric per run, and the queries they disagree on.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import type { FewShotComparison } from "../../api/client";
import { withProviders } from "../../test-harness";
import { FewShotCompare } from "./FewShotCompare";

function run(id: number, method: string, references: number, seed: number, iou: number) {
  return {
    id,
    name: `${method} ${references}/${seed}`,
    model_type: method,
    status: "trained" as const,
    split_id: id,
    split_name: null,
    references,
    seed,
    region_profile_id: 1,
    scored: true,
    metrics: { foreground_iou: iou },
    ground_truth_stale: false,
  };
}

const REPORT: FewShotComparison = {
  dataset_id: 7,
  dataset_name: "candle",
  target_label: "defect",
  runs: [run(1, "fss_dino", 1, 0, 0.2), run(2, "fss_dino", 1, 1, 0.4), run(3, "proto_seg", 5, 0, 0.6)],
  samples: [
    {
      sample_id: 9,
      group_key: "candle",
      external_id: "009",
      outcomes: ["hit", "miss", "hit"],
      ious: [0.7, 0, 0.8],
      agree: false,
    },
    {
      sample_id: 10,
      group_key: "candle",
      external_id: "010",
      outcomes: ["correct_absence", "correct_absence", "correct_absence"],
      ious: [null, null, null],
      agree: true,
    },
  ],
  warnings: [],
};

describe("the few-shot comparison", () => {
  it("means over draws, with their spread, per method and shot count", () => {
    render(withProviders(<MemoryRouter>{<FewShotCompare report={REPORT} />}</MemoryRouter>));
    expect(screen.getByText("1 reference")).toBeTruthy();
    expect(screen.getByText("5 references")).toBeTruthy();
    // fss_dino at one reference: two draws, 0.2 and 0.4.
    expect(screen.getByText("0.300")).toBeTruthy();
    expect(screen.getByText("±0.100")).toBeTruthy();
    expect(screen.getByText("1 refs · seed 0")).toBeTruthy();
  });

  it("shows the queries they disagree on first, and all of them on request", () => {
    render(withProviders(<MemoryRouter>{<FewShotCompare report={REPORT} />}</MemoryRouter>));
    expect(screen.getByText("candle/009")).toBeTruthy();
    expect(screen.queryByText("candle/010")).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "all shared · 2" }));
    expect(screen.getByText("candle/010")).toBeTruthy();
  });
});
