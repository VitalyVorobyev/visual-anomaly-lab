/**
 * The Samples tab of a supervised segmentation run draws its label maps, not a cut.
 *
 * The foreground map cut by `predictionUrl` and the anomaly outline would still draw a
 * picture on a semantic tile — of something else, in the wrong colours, beside a legend
 * naming classes. So a semantic run's tile asks for the server-drawn label map, with every
 * pinned class's colour taken from the one function the sample page and the legend paint
 * with, and an anomaly run's tile is untouched.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import type { SamplePreview, SampleVerdict, Task } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { readResultsState } from "../../api/resultsState";
import { classColour } from "../../components/viewer/labelPaint";
import { withProviders } from "../../test-harness";
import { GalleryTab } from "./GalleryTab";
import type { Verdicts } from "./useVerdicts";

const verdict: SampleVerdict = {
  sample_id: 7,
  group_key: "g",
  external_id: "part-7",
  label: "defect",
  notes: null,
  score: 0.5,
  predicted_defect: true,
  outcome: "hit",
  localized: null,
};

const preview: SamplePreview = {
  sample_id: 7,
  image_id: 70,
  has_map: true,
  has_mask: true,
  width: 64,
  height: 48,
};

const verdicts: Verdicts = {
  shown: [verdict],
  all: [verdict],
  threshold: 0.5,
  rationale: undefined,
  label: "",
  isPending: false,
  error: null,
};

function renderGallery(task: Task, classes?: readonly string[]) {
  const state = readResultsState(new URLSearchParams(), task);
  return render(
    <MemoryRouter>
      {withProviders(
        <GalleryTab
          experimentId={3}
          task={task}
          targetLabel={null}
          classes={classes}
          state={{ ...state, region: true, truth: true }}
          onChange={() => undefined}
          verdicts={verdicts}
          subsets={["test"]}
          range={{ low: 0, high: 1 }}
        />,
        [[queryKeys.previews(3, state.subset), [preview]]],
      )}
    </MemoryRouter>,
  );
}

const hex = (colour: string) => colour.replace("#", "").toLowerCase();

describe("a supervised segmentation run's gallery", () => {
  it("draws the server's label maps in the pinned classes' colours", () => {
    const { container } = renderGallery("semantic_segmentation", ["rust", "moss"]);

    const predicted = container.querySelector<HTMLImageElement>('img[data-labels="prediction"]');
    const truth = container.querySelector<HTMLImageElement>('img[data-labels="truth"]');
    expect(predicted).not.toBeNull();
    expect(truth).not.toBeNull();

    const url = new URL(predicted?.src ?? "", "http://localhost");
    expect(url.pathname).toMatch(/\/api\/experiments\/3\/images\/70\/label-map$/);
    expect(url.searchParams.get("colours")).toBe(
      [hex(classColour(1)), hex(classColour(2))].join(","),
    );
    expect(url.searchParams.has("truth")).toBe(false);
    expect(new URL(truth?.src ?? "", "http://localhost").searchParams.get("truth")).toBe("true");

    // Neither the anomaly cut nor the anomaly outline is drawn beside them.
    expect(container.querySelector('img[src*="render=region"]')).toBeNull();
    expect(container.querySelector('img[src$="/mask"]')).toBeNull();
  });

  it("offers the label-map toggles and a legend of its classes", () => {
    renderGallery("semantic_segmentation", ["rust", "moss"]);

    expect(screen.getByText("foreground")).toBeTruthy();
    expect(screen.queryByText("peak")).toBeNull();
    const legend = screen.getByRole("list", { name: "Classes" });
    expect(legend.textContent).toContain("rust");
    expect(legend.textContent).toContain("moss");
  });
});

describe("an anomaly run's gallery", () => {
  it("still draws the cut and the outline, and no label map", () => {
    const { container } = renderGallery("anomaly");

    expect(container.querySelector("img[data-labels]")).toBeNull();
    expect(container.querySelector('img[src*="render=region"]')).not.toBeNull();
    expect(container.querySelector('img[src$="/mask"]')).not.toBeNull();
    expect(screen.getByText("peak")).toBeTruthy();
    expect(screen.queryByRole("list", { name: "Classes" })).toBeNull();
  });
});

describe("an object detection run's gallery", () => {
  it("draws the server's box picture in the verdict tones, and the box legend", () => {
    const { container } = renderGallery("object_detection", ["rust", "moss"]);

    const boxes = container.querySelector<HTMLImageElement>("img[data-boxes]");
    expect(boxes).not.toBeNull();
    const url = new URL(boxes?.src ?? "", "http://localhost");
    expect(url.pathname).toMatch(/\/api\/experiments\/3\/images\/70\/box-map$/);
    // Match, false positive, missed — three tones, no class colour on the tile.
    expect(url.searchParams.get("colours")?.split(",")).toHaveLength(3);
    expect(url.searchParams.has("predictions")).toBe(false);
    expect(url.searchParams.has("truth")).toBe(false);

    // Neither a label map nor the anomaly cut or outline is drawn beside it.
    expect(container.querySelector("img[data-labels]")).toBeNull();
    expect(container.querySelector('img[src*="render=region"]')).toBeNull();
    expect(container.querySelector('img[src$="/mask"]')).toBeNull();

    expect(screen.getByRole("list", { name: "Box verdicts" }).textContent).toContain("missed");
    expect(screen.getByRole("list", { name: "Classes" }).textContent).toContain("moss");
    expect(screen.queryByText("peak")).toBeNull();
  });

  it("opens with the boxes on and the heatmap off", () => {
    const state = readResultsState(new URLSearchParams(), "object_detection");
    expect([state.region, state.truth, state.heatmap, state.peak]).toEqual([
      true,
      true,
      false,
      false,
    ]);
  });
});
