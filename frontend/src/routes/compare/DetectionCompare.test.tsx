/**
 * Detection runs are compared on what needs no cut: the AP family per run and per class,
 * and never a confidence, which means nothing outside its run (ADR-0028).
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import type { DetectionComparison } from "../../api/client";
import { withProviders } from "../../test-harness";
import { DetectionCompare } from "./DetectionCompare";

function run(id: number, ap: number, stain: number | null) {
  return {
    id,
    name: `boxes ${id}`,
    model_type: "color_detector",
    status: "trained" as const,
    scored: true,
    metrics: {
      classes: ["scratch", "stain"],
      ap,
      ap50: ap + 0.1,
      ap75: ap - 0.1,
      recall: 0.5,
      recall50: 0.6,
      per_class_ap: { scratch: ap, stain },
    },
    ground_truth_stale: false,
  };
}

const REPORT: DetectionComparison = {
  dataset_id: 1,
  dataset_name: "boxes",
  split_id: 2,
  split_name: "stratified",
  subset: "test",
  subsets: ["test"],
  classes: ["scratch", "stain"],
  runs: [run(1, 0.4, 0.3), run(2, 0.6, null)],
  warnings: [],
};

describe("the detection comparison", () => {
  it("puts each run's AP family and per-class AP side by side, with gaps as dashes", () => {
    render(
      withProviders(
        <MemoryRouter>
          <DetectionCompare report={REPORT} onSubset={() => undefined} />
        </MemoryRouter>,
      ),
    );
    expect(screen.getByText("boxes 1")).toBeTruthy();
    expect(screen.getByText("boxes 2")).toBeTruthy();
    const ap = screen.getByText("AP@[.5:.95]").closest("tr");
    expect(ap?.textContent).toContain("0.400");
    expect(ap?.textContent).toContain("0.600");
    expect(screen.getByText("AP · stain").closest("tr")?.textContent).toContain("—");
    // No row reads a cut or the F1 at one: those are each run's own.
    const rows = [...document.querySelectorAll("tbody th")].map((cell) => cell.textContent);
    expect(rows.some((label) => /cut|F1/i.test(label ?? ""))).toBe(false);
  });
});
