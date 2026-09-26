import { describe, expect, it } from "vitest";

import type { ImageTruth } from "../../api/client";
import type { RasterLayer } from "../../components/viewer/SampleStage";
import {
  EXPLORE_DIM,
  legendClasses,
  stackLayers,
  truthLayers,
  truthSummary,
  type TruthView,
} from "./truthLayers";

const FISH = { key: "fish", name: "Fish", color: "#e8590c" };
const HOLE = { key: "missing_hole", name: "Missing hole", color: "#1c7ed6" };
const DEFECT = { key: "defect", name: "Defect", color: "#c026d3" };

function truth(over: Partial<ImageTruth> = {}): ImageTruth {
  return {
    image_id: 1,
    source: "revision",
    revision_id: 3,
    classes: [],
    boxes: [],
    regions_url: null,
    outline: false,
    ...over,
  };
}

const ON: TruthView = { on: true, opacity: 0.8, exploring: false };

describe("truthLayers", () => {
  it("draws a class mask as one raster layer at the reader's opacity", () => {
    const { layers, shapes } = truthLayers(
      truth({ classes: [FISH], regions_url: "/api/images/1/truth/regions.png?v=revision-3" }),
      ON,
    );
    expect(shapes).toEqual([]);
    expect(layers).toHaveLength(1);
    expect(layers[0]!.key).toBe("truth");
    expect(layers[0]!.opacity).toBe(0.8);
    expect(layers[0]!.src).toContain("/api/images/1/truth/regions.png?v=revision-3");
    // The palette is part of the address, so a recoloured class is a new picture.
    expect(layers[0]!.src).toContain("c=e8590c");
  });

  it("draws boxes as tagged rectangles in their class's colour, with no raster", () => {
    const { layers, shapes } = truthLayers(
      truth({
        classes: [HOLE],
        boxes: [
          { label_key: "missing_hole", x: 10, y: 20, width: 30, height: 40 },
          { label_key: "unlisted", x: 1, y: 2, width: 3, height: 4 },
        ],
      }),
      ON,
    );
    expect(layers).toEqual([]);
    expect(shapes).toHaveLength(2);
    expect(shapes[0]).toMatchObject({
      kind: "box",
      x: 10,
      y: 20,
      width: 30,
      height: 40,
      label: "Missing hole",
      colour: HOLE.color,
      opacity: 0.8,
    });
    // A class the legend does not know still draws, under its key.
    expect(shapes[1]).toMatchObject({ label: "unlisted", colour: undefined });
  });

  it("draws nothing when truth is off, missing, or has nothing to draw", () => {
    const drawn = truth({ classes: [FISH], regions_url: "/x.png?v=1" });
    expect(truthLayers(drawn, { ...ON, on: false })).toEqual({ layers: [], shapes: [] });
    expect(truthLayers(undefined, ON)).toEqual({ layers: [], shapes: [] });
    expect(truthLayers(truth({ source: "normal" }), ON)).toEqual({ layers: [], shapes: [] });
  });

  it("dims beneath Explore, which always draws above it", () => {
    const drawn = truth({
      classes: [DEFECT],
      regions_url: "/api/images/1/truth/regions.png?v=mask-a",
      outline: true,
      boxes: [{ label_key: "defect", x: 0, y: 0, width: 1, height: 1 }],
    });
    const dimmed = truthLayers(drawn, { ...ON, exploring: true });
    expect(dimmed.layers[0]!.opacity).toBeCloseTo(0.8 * EXPLORE_DIM);
    expect(dimmed.shapes[0]!.opacity).toBeCloseTo(0.8 * EXPLORE_DIM);

    const explore: RasterLayer[] = [{ key: "explore-clusters", src: "/c.png", opacity: 0.8 }];
    expect(stackLayers(dimmed.layers, explore).map((layer) => layer.key)).toEqual([
      "truth",
      "explore-clusters",
    ]);
  });
});

describe("the legend", () => {
  it("lists each class once, whichever channel shows it", () => {
    // Any channel count: three images, two sharing a class, one still loading.
    const classes = legendClasses([
      truth({ classes: [FISH] }),
      undefined,
      truth({ classes: [FISH, HOLE] }),
    ]);
    expect(classes.map((entry) => entry.key)).toEqual(["fish", "missing_hole"]);
  });

  it("says where the truth comes from, or why there is none", () => {
    expect(truthSummary([])).toBe("");
    expect(truthSummary([truth({ classes: [FISH] })])).toBe("Completed annotation");
    expect(truthSummary([truth()])).toBe("Completed annotation: no class present");
    expect(truthSummary([truth({ classes: [FISH] }), truth({ source: "none", classes: [] })])).toBe(
      "Completed annotation on some channels",
    );
    expect(truthSummary([truth({ source: "imported_mask", classes: [DEFECT] })])).toBe(
      "Imported defect mask, outlined",
    );
    expect(truthSummary([truth({ source: "normal" })])).toBe("Labelled normal: nothing to draw");
    expect(truthSummary([truth({ source: "none" })])).toBe("No truth for this sample yet");
  });
});
