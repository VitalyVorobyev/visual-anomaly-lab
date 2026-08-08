import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import type { ExperimentDetail, JobSummary, TrainingState } from "../../api/client";
import { RunBar } from "./RunBar";

function detail(overrides: Partial<ExperimentDetail> = {}): ExperimentDetail {
  return {
    id: 3,
    name: "run",
    model_type: "pixel_reference",
    config: {},
    supports_resume: false,
    training_state: null,
    ...overrides,
  } as ExperimentDetail;
}

function trained(steps: number, runs = 1): TrainingState {
  return {
    format: 1,
    completed_steps: steps,
    runs,
    last_run_steps: steps,
    model_type: "efficientad_anomalib",
    written_at: "",
    resumable: true,
  };
}

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
        detail={detail()}
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
    wrap(
      <RunBar
        experimentId={3}
        detail={detail()}
        jobs={[]}
        hasTrained={false}
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    const score = screen.getByRole("button", { name: /Score & evaluate/ });
    expect(score.hasAttribute("disabled")).toBe(true);
    expect(score.getAttribute("title")).toBe("Nothing has been trained yet.");
  });

  it("shows the live run, its progress and a way to stop it", () => {
    wrap(
      <RunBar
        experimentId={3}
        detail={detail()}
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
        detail={detail()}
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
    wrap(
      <RunBar
        experimentId={3}
        detail={detail()}
        jobs={[job({})]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("offers no continue control for a method with no steps", () => {
    // `pixel_reference` builds a median; there is nothing to continue, and a disabled
    // control for an idea that does not apply is worse than no control.
    wrap(
      <RunBar
        experimentId={3}
        detail={detail()}
        jobs={[job({})]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
  });

  it("makes continuing the obvious action once something is trained", () => {
    /**
     * Train used to look like a button that repeated the same run for no reason. On a
     * trained experiment it becomes the secondary, named, confirmed action.
     */
    wrap(
      <RunBar
        experimentId={3}
        detail={detail({
          supports_resume: true,
          training_state: trained(4000),
          config: { max_steps: 4000 },
        })}
        jobs={[job({})]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    expect(screen.getByRole("button", { name: "Retrain from scratch" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Train" })).toBeNull();
    expect(screen.getByRole("button", { name: "Continue" })).toBeTruthy();
    // Defaults to the per-run budget, so "another 4000" is one click.
    expect(screen.getByLabelText("Additional steps").getAttribute("value")).toBe("4000");
  });

  it("prints what continuing will do to the learning rate before the run", () => {
    // The rate goes back up at the resume point. Surprising enough to state rather than
    // leave to be discovered in the chart afterwards (ADR-0025).
    wrap(
      <RunBar
        experimentId={3}
        detail={detail({
          supports_resume: true,
          training_state: trained(4000),
          config: { max_steps: 4000 },
        })}
        jobs={[job({})]}
        hasTrained
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    expect(screen.getByText(/Continuing to 8000/)).toBeTruthy();
    expect(screen.getByText(/drop to step 7600/)).toBeTruthy();
    expect(screen.getByText(/penalty-set order\s+restarts/)).toBeTruthy();
  });

  it("disables continue when nothing has trained, and says why", () => {
    wrap(
      <RunBar
        experimentId={3}
        detail={detail({ supports_resume: true, config: { max_steps: 4000 } })}
        jobs={[]}
        hasTrained={false}
        onFollow={noop}
        onViewLog={noop}
      />,
    );

    const button = screen.getByRole("button", { name: "Continue" });
    expect(button.hasAttribute("disabled")).toBe(true);
    expect(button.getAttribute("title")).toBe("Nothing has been trained yet.");
  });
});
