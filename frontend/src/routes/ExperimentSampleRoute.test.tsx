/**
 * The result viewer's registration contract, made executable.
 *
 * Every assertion here is about a fault that *looks correct*. `object-fill` inside a frame
 * the browser had squeezed drew the photograph and all three overlays at the same wrong
 * aspect ratio, so nothing on screen disagreed with anything else — the picture was simply
 * not the shape of the part. And a source-frame ground-truth outline over a prepared-frame
 * diagnostic is a green line around a plausible, wrong place.
 */

import { render } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { ExperimentSampleRoute } from "./ExperimentSampleRoute";

const IMAGE = {
  image_id: 501,
  channel: "bright",
  score: 3.25,
  inference_ms: 18.4,
  has_map: true,
  has_mask: true,
  width: 1280,
  height: 1024,
  peak: { x: 640, y: 512 },
  localized: true,
  tolerance_px: 33,
  map_scale: { low: 0.0, high: 4.0 },
};

const EXPERIMENT = {
  id: 9,
  map_range: { low: 0.0, high: 4.0 },
  produces_diagnostics: true,
};

const DIAGNOSTICS = {
  version: 1,
  entries: [
    {
      key: "map_student_teacher",
      title: "Student–teacher error",
      kind: "map",
      scope: "image",
      origin: "run",
      image_id: 501,
      path: "image-501/map_student_teacher.npy",
      payload: null,
      shape: [256, 256],
      description: null,
    },
  ],
  ranges: {},
  image_budget: 64,
  truncated_images: 0,
};

function renderSample(
  over: { images?: unknown; diagnostics?: unknown; search?: string } = {},
) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[`/experiments/9/samples/12${over.search ?? ""}`]}>
        <Routes>
          <Route
            path="experiments/:experimentId/samples/:sampleId"
            element={<ExperimentSampleRoute />}
          />
        </Routes>
      </MemoryRouter>,
      [
        [queryKeys.sampleImages(9, 12), over.images ?? [IMAGE]],
        [queryKeys.experiment(9), EXPERIMENT],
        [queryKeys.diagnostics(9), over.diagnostics ?? DIAGNOSTICS],
      ],
    ),
  );
}

/** Every layer of the stage, in stacking order — the photograph first. */
function stageLayers(container: HTMLElement): HTMLImageElement[] {
  const stage = container.querySelector("[data-stage]")!;
  return [...stage.querySelectorAll("img")];
}

describe("the experiment sample viewer", () => {
  it("stretches nothing", () => {
    // `object-fill` is what let the frame's shape override the picture's. With the stage
    // laid out at the image's own pixel size there is no box to fill and nothing to fit.
    const { container } = renderSample();

    const stretched = [...container.querySelectorAll("*")].filter((node) =>
      /\bobject-(fill|contain|cover|none|scale-down)\b/.test(node.className.toString()),
    );

    expect(stretched).toHaveLength(0);
  });

  it("lays the stage out at the image's own pixel size", () => {
    // This is what makes layer registration structural: a layer at `inset-0` covers exactly
    // the source frame, at every viewport size, with no letterbox to correct for.
    const { container } = renderSample();
    const stage = container.querySelector("[data-stage]") as HTMLElement;

    expect(stage.style.width).toBe("1280px");
    expect(stage.style.height).toBe("1024px");
  });

  it("draws one layer per enabled overlay, over the photograph", () => {
    // The default results state is heatmap on, ground truth on, segmentation off.
    const { container } = renderSample();
    const layers = stageLayers(container);

    expect(layers).toHaveLength(3);
    expect(layers[0]!.getAttribute("src")).toContain("/api/images/501/preview");
    expect(layers[1]!.getAttribute("src")).toContain("/anomaly-map?");
    expect(layers[2]!.getAttribute("src")).toContain("/api/images/501/mask");
  });

  it("draws no map layer for an image the run recorded none for", () => {
    const { container } = renderSample({ images: [{ ...IMAGE, has_map: false }] });

    expect(stageLayers(container)).toHaveLength(2);
  });

  it("keeps the overlay stack in the source frame", () => {
    // Every layer up here was projected through the pinned transform before it was stored,
    // so asking for a prepared-frame mask would be the misregistration in the other
    // direction.
    const { container } = renderSample();
    const truth = stageLayers(container).at(-1)!;

    expect(truth.getAttribute("src")).not.toContain("frame=prepared");
  });

  it("fetches the diagnostics panes' ground truth in the prepared frame", () => {
    // A diagnostic is drawn at the array's own prepared size, because nothing projects it.
    // The mask has to meet it there.
    const { container } = renderSample();

    const overDiagnostics = [...container.querySelectorAll("img")].filter((node) =>
      node.getAttribute("src")?.includes("/mask?"),
    );

    expect(overDiagnostics.length).toBeGreaterThan(0);
    for (const node of overDiagnostics) {
      const url = new URL(node.getAttribute("src")!, "http://placeholder");
      expect(url.searchParams.get("frame")).toBe("prepared");
      expect(url.searchParams.get("experiment_id")).toBe("9");
    }
  });

  it("draws no peak marker until the layer is asked for", () => {
    // Off by default, like the segmentation. The stage holds raster layers only.
    const { container } = renderSample();

    expect(container.querySelector("[data-stage] svg")).toBeNull();
  });

  it("marks the peak and the window its verdict was decided in", () => {
    /*
     * The two things that produced a one-bit verdict, in the image's own coordinates. The
     * box is a **square** because `eval/localization.hits` tests a Chebyshev neighbourhood —
     * a circle of the same radius would call a peak in its corner a miss on screen while the
     * badge beside it says localized.
     */
    const { container } = renderSample({ search: "?pk=1" });
    const overlay = container.querySelector("[data-stage] svg");

    expect(overlay).not.toBeNull();
    // The cross, drawn as two segments through the peak.
    expect(overlay!.querySelectorAll("line")).toHaveLength(2);
    const box = overlay!.querySelector("path");
    expect(box).not.toBeNull();
    // 2 · 33 + 1 pixels on a side, centred on the peak: 640 ± 33.5, 512 ± 33.5.
    expect(box!.getAttribute("d")).toContain("606.50 478.50");
  });

  it("draws nothing where the run recorded no peak", () => {
    // A method with no map, or a map that is NaN everywhere. `(0, 0)` would be a coordinate
    // nobody measured.
    const { container } = renderSample({
      search: "?pk=1",
      images: [{ ...IMAGE, peak: null, localized: null, tolerance_px: null }],
    });

    expect(container.querySelector("[data-stage] svg")).toBeNull();
  });

  it("marks the peak with no window when no tolerance was resolved", () => {
    const { container } = renderSample({
      search: "?pk=1",
      images: [{ ...IMAGE, localized: null, tolerance_px: null }],
    });
    const overlay = container.querySelector("[data-stage] svg");

    expect(overlay!.querySelectorAll("line")).toHaveLength(2);
    expect(overlay!.querySelector("path")).toBeNull();
  });

  it("leaves the combined map pane's outline in the source frame", () => {
    // The one diagnostics pane that is *not* a prepared-frame quantity: the stored map was
    // projected before it was written and this route renders it at the source's own size.
    const { container } = renderSample();

    const panes = [...container.querySelectorAll("figure")].filter((node) =>
      node.textContent?.includes("Combined anomaly map"),
    );

    expect(panes).toHaveLength(1);
    const outlines = [...panes[0]!.querySelectorAll("img")].filter((node) =>
      node.getAttribute("src")?.includes("/mask"),
    );
    expect(outlines).toHaveLength(1);
    expect(outlines[0]!.getAttribute("src")).not.toContain("frame=prepared");
  });
});
