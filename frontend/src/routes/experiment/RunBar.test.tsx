import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import type { JobSummary } from "../../api/client";
import { RunBar } from "./RunBar";

function job(overrides: Partial<JobSummary>): JobSummary {
  return {
    id: 1,
    kind: "train",
    status: "succeeded",
    progress: 1,
    message: null,
    experiment_id: 3,
    started_at: null,
    finished_at: null,
    error: null,
    ...overrides,
  } as JobSummary;
}

function wrap(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

const noop = () => {};

describe("RunBar", () => {
  it("offers both runs once something has trained", () => {
    wrap(
      <RunBar
        experimentId={3}
        jobs={[job({})]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    expect(screen.getByRole("button", { name: "Train" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /Score & evaluate/ }).hasAttribute("disabled"),
    ).toBe(false);
  });

  it("says why scoring is unavailable rather than offering a button that fails", () => {
    wrap(<RunBar experimentId={3} jobs={[]} hasTrained={false} onFollow={noop} onViewLog={noop} />);

    const score = screen.getByRole("button", { name: /Score & evaluate/ });
    expect(score.hasAttribute("disabled")).toBe(true);
    expect(score.getAttribute("title")).toBe("Nothing has been trained yet.");
  });

  it("shows the live run, its progress and a way to stop it", () => {
    wrap(
      <RunBar
        experimentId={3}
        jobs={[job({ id: 7, status: "running", progress: 0.42, message: "step 1680/4000" })]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    expect(screen.getByText("#7 train")).toBeTruthy();
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("42");
    expect(screen.getByText(/step 1680\/4000/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
  });

  it("blocks a second run while one is live, since the queue runs one at a time", () => {
    wrap(
      <RunBar
        experimentId={3}
        jobs={[job({ status: "running" })]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    expect(screen.getByRole("button", { name: "Train" }).hasAttribute("disabled")).toBe(true);
  });

  it("draws no progress bar when nothing is running", () => {
    /** A bar frozen at 100% after a run finished reads as a run that is still going. */
    wrap(<RunBar experimentId={3} jobs={[job({})]} hasTrained onFollow={noop} onViewLog={noop} />);
    expect(screen.queryByRole("progressbar")).toBeNull();
  });
});
