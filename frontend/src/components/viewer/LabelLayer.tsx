/**
 * One label map over the photograph, inside a `SampleStage`: a canvas at the plane's size,
 * stretched to the stage's image box.
 *
 * A canvas over the value plane rather than the gallery's server-drawn PNG, which is bounded
 * to a thumbnail: here the map is drawn at the plane's own resolution, in the interface's
 * colours (`labelPaint.ts`); rather than `VectorLayer` because a label map is a raster by nature and
 * tracing it into polygons would be a second, lossy copy of what the method wrote.
 * `pixelated` keeps a decimated plane's class borders hard instead of blurring two classes
 * into a colour neither has.
 */

import { useEffect, useRef } from "react";

import type { ValuePlane } from "@vitavision/lab-ui";

import { paintLabels, type LabelStyle } from "./labelPaint";

export function LabelLayer({ plane, style }: { plane: ValuePlane; style: LabelStyle }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const target = canvas.current;
    const context = target?.getContext("2d");
    if (!target || !context) return;
    const pixels = paintLabels(plane, style);
    const image = context.createImageData(plane.width, plane.height);
    image.data.set(pixels);
    context.putImageData(image, 0, 0);
  }, [plane, style]);

  return (
    <canvas
      ref={canvas}
      width={plane.width}
      height={plane.height}
      aria-hidden
      data-labels={style}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ imageRendering: "pixelated" }}
    />
  );
}
