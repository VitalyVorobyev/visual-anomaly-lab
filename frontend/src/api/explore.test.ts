import { describe, expect, it } from "vitest";

import { SERIES_COLOURS } from "@vitavision/lab-ui";

import type { BitmapShape, SpatialTransform } from "./client";
import { carriedFrom, cellAt, clusterAt, exploreMapUrl, hexDigits } from "./explore";

/** A 200x100 landscape contain-resized into the 448-pixel frame: 112 rows of pad above. */
const LANDSCAPE: SpatialTransform = {
  schema_version: 1,
  source_width: 200,
  source_height: 100,
  crop_left: 0,
  crop_top: 0,
  crop_right: 200,
  crop_bottom: 100,
  prepared_width: 448,
  prepared_height: 448,
  resized_width: 448,
  resized_height: 224,
  pad_left: 0,
  pad_top: 112,
  pad_right: 0,
  pad_bottom: 112,
};

describe("cellAt", () => {
  it("lands a source pixel in the grid cell the backend would", () => {
    // Mirrors `test_letterbox_cells_are_left_out_and_clicks_land_on_the_image`.
    expect(cellAt(LANDSCAPE, 14, 32, 32, { x: 0, y: 0 })).toEqual({ row: 8, col: 0 });
    expect(cellAt(LANDSCAPE, 14, 32, 32, { x: 199, y: 99 })).toEqual({ row: 23, col: 31 });
  });

  it("clamps to the grid", () => {
    expect(cellAt(LANDSCAPE, 14, 32, 32, { x: 10_000, y: 10_000 })).toEqual({ row: 31, col: 31 });
  });
});

describe("clusterAt", () => {
  const cells = Array.from({ length: 32 * 32 }, (_, index) =>
    Math.floor(index / 32) < 8 ? 0 : (index % 32) < 16 ? 1 : 2,
  );
  const answer = { transform: LANDSCAPE, patch_size: 14, grid_rows: 32, grid_cols: 32, cells };

  it("reads the cluster under a click from the cells the answer carried", () => {
    expect(clusterAt(answer, { x: 10, y: 50 })).toBe(1);
    expect(clusterAt(answer, { x: 190, y: 50 })).toBe(2);
  });

  it("answers null without clusters", () => {
    expect(clusterAt({ ...answer, cells: null }, { x: 10, y: 50 })).toBeNull();
  });
});

describe("exploreMapUrl", () => {
  it("adds a threshold and the first series colour for a mask", () => {
    const url = new URL(exploreMapUrl("/api/explore/maps/abc.png", { threshold: 0.6 }));
    expect(url.pathname).toBe("/api/explore/maps/abc.png");
    expect(url.searchParams.get("threshold")).toBe("0.600");
    expect(url.searchParams.get("colours")).toBe(hexDigits(SERIES_COLOURS[0] as string));
  });

  it("names one colour per cluster and keeps a picked one", () => {
    const url = new URL(exploreMapUrl("/m.png", { clusters: 3, cluster: 2 }));
    expect(url.searchParams.get("colours")?.split(",")).toHaveLength(3);
    expect(url.searchParams.get("cluster")).toBe("2");
  });

  it("is the bare map with no options", () => {
    expect(new URL(exploreMapUrl("/m.png")).search).toBe("");
  });
});

describe("carriedFrom", () => {
  const shape: BitmapShape = {
    id: "explore-1",
    label_key: "defect",
    kind: "bitmap",
    operation: "add",
    instance_id: null,
    x: 1,
    y: 2,
    width: 3,
    height: 4,
    png_base64: "AA==",
  };

  it("takes a candidate carried for this image", () => {
    const state = { exploreCandidate: { imageId: 7, shape, area: 9, source: "cluster 2" } };
    expect(carriedFrom(state, 7)).toEqual({ imageId: 7, shape, area: 9, source: "cluster 2" });
  });

  it("ignores one carried for another image, and any other state", () => {
    expect(carriedFrom({ exploreCandidate: { imageId: 8, shape } }, 7)).toBeNull();
    expect(carriedFrom(null, 7)).toBeNull();
    expect(carriedFrom({ something: 1 }, 7)).toBeNull();
  });
});
