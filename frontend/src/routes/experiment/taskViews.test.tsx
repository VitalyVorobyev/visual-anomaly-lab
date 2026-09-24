/**
 * A segmentation run's results are read in its own terms, through the task registry, and an
 * anomaly run's are untouched by it.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { EMPTY_RESULTS } from "../../api/resultsState";
import { withProviders } from "../../test-harness";
import { taskView, type ResultsBodyProps } from "./taskViews";

const METRICS = [
  {
    subset: "test" as const,
    computed_at: "2026-09-24T00:00:00Z",
    ground_truth_digest: null,
    ground_truth_stale: false,
    metrics: {
      foreground_iou: 0.6123,
      foreground_dice: 0.75,
      boundary_f1: null,
      boundary_tolerance_px: 2,
      image_present_recall: 1,
      image_absent_false_positive_rate: 0.1,
      image_small_region_recall: null,
      small_region_fraction: 0.01,
      image_presence_roc_auc: 0.9,
      images: { present: 3, absent: 10, unlabeled: 2, without_prediction: 0 },
    },
  },
];

function props(): ResultsBodyProps {
  return {
    experimentId: 5,
    subsets: ["test"],
    metrics: METRICS,
    state: EMPTY_RESULTS,
    onChange: () => undefined,
    verdicts: {
      shown: [],
      all: [],
      threshold: 0,
      rationale: undefined,
      label: "",
      isPending: false,
      error: null,
    },
    aggregation: "max",
    targetLabel: "scratch",
  };
}

describe("the task views", () => {
  it("reads a few-shot run as overlap and per-image outcomes, with gaps as dashes", () => {
    const view = taskView("few_shot_segmentation");
    render(withProviders(<MemoryRouter>{view.MetricTables(props())}</MemoryRouter>));
    expect(screen.getByText("Foreground IoU")).toBeTruthy();
    expect(screen.getByText("0.612")).toBeTruthy();
    expect(screen.getByText("Boundary F1 (±2 px)")).toBeTruthy();
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("Sample ROC-AUC")).toBeNull();
  });

  it("reads the probability map's ranking beside the overlap, as its own threshold-free row", () => {
    const view = taskView("few_shot_segmentation");
    const withRanking = METRICS.map((entry) => ({
      ...entry,
      metrics: { ...entry.metrics, pixel_average_precision: 0.4321 },
    }));
    render(
      withProviders(
        <MemoryRouter>{view.MetricTables({ ...props(), metrics: withRanking })}</MemoryRouter>,
      ),
    );
    expect(screen.getByText("Pixel AP (threshold-free)")).toBeTruthy();
    expect(screen.getByText("0.432")).toBeTruthy();
    // A run without maps has no pixel ROC-AUC: a dash beside its label, never a zero.
    expect(screen.getByText("Pixel ROC-AUC (threshold-free)")).toBeTruthy();
  });

  it("reads a supervised segmentation run off its confusion matrix, per class", () => {
    const view = taskView("semantic_segmentation");
    const semantic = [
      {
        subset: "test" as const,
        computed_at: "2026-09-24T00:00:00Z",
        ground_truth_digest: null,
        ground_truth_stale: false,
        metrics: {
          classes: ["scratch", "stain"],
          mean_iou: 0.4667,
          per_class_iou: { scratch: 0.3333, stain: null },
          background_iou: 0.6667,
          pixel_accuracy: 0.75,
          mean_class_accuracy: 0.625,
          frequency_weighted_iou: 0.6208,
          images: { labelled: 4, unlabeled: 1, without_prediction: 0 },
        },
      },
    ];
    render(
      withProviders(
        <MemoryRouter>{view.MetricTables({ ...props(), metrics: semantic })}</MemoryRouter>,
      ),
    );
    expect(screen.getByText("Mean IoU")).toBeTruthy();
    expect(screen.getByText("0.467")).toBeTruthy();
    expect(screen.getByText("IoU · scratch")).toBeTruthy();
    // A class nobody drew or predicted has no IoU, and reads as a dash rather than a zero.
    expect(screen.getByText("IoU · stain").nextElementSibling?.textContent).toBe("—");
    expect(screen.queryByText("Foreground IoU")).toBeNull();
    expect(view.filters.map((filter) => filter.id)).toEqual(["all"]);
  });

  it("gives each task its own outcome strip and rank words", () => {
    const anomaly = taskView("anomaly");
    const segmentation = taskView("few_shot_segmentation");
    expect(anomaly.filters.map((filter) => filter.id)).toContain("fp");
    expect(segmentation.filters.map((filter) => filter.id)).toEqual([
      "all",
      "mistakes",
      "hit",
      "low_iou",
      "miss",
      "false_presence",
      "correct_absence",
      "unlabeled",
    ]);
    expect(segmentation.filters.find((filter) => filter.id === "mistakes")?.outcomes).toEqual([
      "miss",
      "false_presence",
      "low_iou",
    ]);
    expect(anomaly.rank.desc).toBe("most anomalous");
    expect(segmentation.rank.desc).toBe("most present");
    expect(taskView(undefined)).toBe(anomaly);
  });
});
