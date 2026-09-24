/**
 * The one viewer draws its stack in order, and vector shapes in image pixels at a constant
 * on-screen weight.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SampleStage } from "./SampleStage";
import { labelAnchor, type VectorShape } from "./VectorLayer";

const IMAGE = { id: 42, width: 640, height: 480 };

describe("SampleStage", () => {
  it("stacks the photograph, then the raster layers in order, then the vectors", () => {
    const shapes: VectorShape[] = [
      { kind: "box", id: "a", x: 10, y: 20, width: 100, height: 50, label: "scratch 0.91" },
      { kind: "polygon", id: "b", points: [[5, 5], [60, 5], [30, 40]], tone: "defect", dashed: true },
    ];
    const { container } = render(
      <SampleStage
        image={IMAGE}
        alt="photo"
        view={null}
        onView={() => {}}
        label="canvas"
        layers={[
          { key: "heatmap", src: "/heat.png" },
          { key: "truth", src: "/truth.png" },
        ]}
        shapes={shapes}
      />,
    );

    const images = [...container.querySelectorAll("img")];
    expect(images.map((img) => img.getAttribute("alt"))).toEqual(["photo", "", ""]);
    expect(images[0]?.getAttribute("src")).toContain("/api/images/42/");
    expect(images.slice(1).map((img) => img.getAttribute("src"))).toEqual(["/heat.png", "/truth.png"]);

    const svg = container.querySelector("svg[viewBox='0 0 640 480']");
    expect(svg).not.toBeNull();
    const rect = svg?.querySelector("g[data-shape=box] > rect");
    expect(rect?.getAttribute("x")).toBe("10");
    expect(rect?.getAttribute("vector-effect")).toBe("non-scaling-stroke");
    expect(svg?.textContent).toContain("scratch 0.91");
    const polygon = svg?.querySelector("polygon");
    expect(polygon?.getAttribute("points")).toBe("5,5 60,5 30,40");
    expect(polygon?.getAttribute("stroke-dasharray")).toBe("4 3");
  });

  it("draws no vector layer when there is nothing to draw", () => {
    const { container } = render(
      <SampleStage image={IMAGE} alt="photo" view={null} onView={() => {}} label="canvas" />,
    );
    expect(container.querySelector("svg[viewBox='0 0 640 480']")).toBeNull();
  });
});

describe("labelAnchor", () => {
  it("tags a box at its top-left corner", () => {
    expect(labelAnchor({ kind: "box", id: "a", x: 3, y: 4, width: 5, height: 6 })).toEqual({ x: 3, y: 4 });
  });

  it("tags a polygon at its topmost vertex, leftmost among ties", () => {
    expect(
      labelAnchor({ kind: "polygon", id: "p", points: [[9, 2], [4, 2], [1, 8]] }),
    ).toEqual({ x: 4, y: 2 });
    expect(labelAnchor({ kind: "polygon", id: "p", points: [] })).toBeNull();
  });
});
