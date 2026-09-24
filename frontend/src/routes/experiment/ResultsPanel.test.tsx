/**
 * The results panel reads and writes the experiment's shared results state.
 *
 * The bug this guards: Overview and Benchmark each kept the threshold in their own
 * `useState`, while the Samples tab read it from a URL nothing wrote — so a cut chosen on
 * Overview never reached the TP/FP badges one tab over. The panel is now controlled by the
 * URL state, and every link it hands to the Samples tab carries the cut in force.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import type { ResultsPage, SampleVerdict, ThresholdReport } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import type { ResultsState } from "../../api/resultsState";
import { EMPTY_RESULTS } from "../../api/resultsState";
import { withProviders } from "../../test-harness";
import { Results } from "./ResultsPanel";

const verdict = (id: number, label: "normal" | "defect", outcome: string): SampleVerdict => ({
  sample_id: id,
  group_key: "g",
  external_id: String(id),
  label,
  notes: null,
  score: id,
  predicted_defect: outcome === "tp" || outcome === "fp",
  outcome,
  localized: null,
});

const ROWS = [verdict(1, "defect", "tp"), verdict(2, "defect", "fn"), verdict(3, "normal", "fp")];

const PAGE: ResultsPage = {
  experiment_id: 7,
  subset: "test",
  suggested_threshold: 0.5,
  threshold_rationale: "maximizes F1",
  score_min: 0,
  score_max: 1,
  samples: ROWS,
};

const REPORT: ThresholdReport = {
  threshold: 0.8,
  confusion: { true_positive: 1, false_positive: 1, true_negative: 0, false_negative: 1 },
  precision: 0.5,
  recall: 0.5,
  f1: 0.5,
  accuracy: 0.33,
  unlabeled: 0,
  samples: ROWS,
};

function renderPanel(state: ResultsState, onChange = vi.fn()) {
  render(
    <MemoryRouter>
      {withProviders(
        <Results experimentId={7} subsets={["test"]} state={state} onChange={onChange} />,
        [
          [queryKeys.results(7, "test"), PAGE],
          [queryKeys.threshold(7, "test", state.threshold ?? 0.5), REPORT],
        ],
      )}
    </MemoryRouter>,
  );
  return onChange;
}

describe("the results panel's threshold", () => {
  it("shows the cut the shared state carries, not a private one", () => {
    renderPanel({ ...EMPTY_RESULTS, subset: "test", threshold: 0.8 });
    expect(screen.getByText("0.8000")).toBeTruthy();
  });

  it("hands the Samples tab the cut in force", () => {
    renderPanel({ ...EMPTY_RESULTS, subset: "test", threshold: 0.8 });
    const link = screen.getByRole("link", { name: /false negatives/ });
    const search = new URLSearchParams(link.getAttribute("href")?.split("?")[1]);
    expect(search.get("tab")).toBe("samples");
    expect(search.get("t")).toBe("0.8");
    expect(search.get("subset")).toBe("test");
    expect(search.get("outcome")).toBe("fn");
  });

  it("offers the way back to the suggestion only once a cut was chosen", () => {
    const onChange = renderPanel({ ...EMPTY_RESULTS, subset: "test", threshold: 0.8 });
    fireEvent.click(screen.getByRole("button", { name: "Back to the suggestion" }));
    expect(onChange).toHaveBeenCalledWith({ threshold: undefined });
  });

  it("opens on the suggestion when nothing was chosen", () => {
    renderPanel({ ...EMPTY_RESULTS, subset: "test" });
    expect(screen.getByText("0.5000")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Back to the suggestion" })).toBeNull();
  });
});
