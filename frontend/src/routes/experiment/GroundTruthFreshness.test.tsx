import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import type { MetricSummary } from "../../api/client";
import { BenchmarkTab } from "./BenchmarkTab";
import { Headline, Metrics } from "./OverviewTab";

const STALE: MetricSummary = {
  subset: "test",
  metrics: { sample_roc_auc: 0.91 },
  computed_at: "2026-08-12T12:00:00Z",
  ground_truth_digest: "older",
  ground_truth_stale: true,
};

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

describe("ground-truth freshness", () => {
  it("marks the headline and explains how to refresh the metrics", () => {
    wrap(
      <>
        <Headline metrics={[STALE]} subset="test" />
        <Metrics experimentId={7} metrics={[STALE]} aggregation="max" />
      </>,
    );

    expect(screen.getByText("truth changed")).toBeTruthy();
    expect(screen.getByText("Ground truth changed")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Recompute" })).toBeTruthy();
  });

  it("does not pair current curves with stale stored areas", () => {
    wrap(<BenchmarkTab experimentId={7} subsets={["test"]} metrics={[STALE]} />);

    expect(screen.getByText("Recompute before reading these curves")).toBeTruthy();
    expect(screen.queryByLabelText(/ROC curve/)).toBeNull();
  });
});
