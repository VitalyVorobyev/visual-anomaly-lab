/**
 * Shapes drawn in image pixels, inside a stage's transform: boxes and polygons, each with an
 * optional label.
 *
 * The raster layers — heatmap, segmentation, ground-truth outline — are PNGs the backend
 * renders at the source frame. A detection result is not: it is a handful of boxes, each
 * with a class and a score, and whether it counts as a hit is decided per box. Rasterising
 * that server-side would bake the label size into the picture and lose the per-box colour,
 * so it is drawn here as SVG in the same `viewBox` the stage is laid out at.
 *
 * Strokes use `vector-effect: non-scaling-stroke` and labels are sized through the stage's
 * scale, so both stay the same on-screen weight at every zoom — a box whose outline grows
 * into a band at 800 % hides the pixels it is pointing at.
 */

import { toneColor, useStage, type MeasureTone } from "@vitavision/lab-ui";

interface ShapeBase {
  id: string;
  /** Printed beside the shape: a class name, a score, or both. */
  label?: string;
  tone?: MeasureTone;
  /** Dashed for what is expected rather than found — ground truth under a prediction. */
  dashed?: boolean;
  /**
   * The label tag's fill, when it names something other than the outline's verdict — a
   * detection's class colour on a box toned by whether it matched.
   */
  labelColour?: string;
}

export interface BoxShape extends ShapeBase {
  kind: "box";
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PolygonShape extends ShapeBase {
  kind: "polygon";
  points: readonly (readonly [number, number])[];
}

export type VectorShape = BoxShape | PolygonShape;

const LABEL_PX = 11;
const STROKE_PX = 1.5;

export function VectorLayer({
  width,
  height,
  shapes,
}: {
  width: number;
  height: number;
  shapes: readonly VectorShape[];
}) {
  const stage = useStage();
  // Image pixels per CSS pixel: what one on-screen pixel measures in the viewBox.
  const px = stage.view.scale > 0 ? 1 / stage.view.scale : 1;
  if (shapes.length === 0) return null;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      aria-hidden
    >
      {shapes.map((shape) => {
        const colour = toneColor(shape.tone);
        const anchor = labelAnchor(shape);
        return (
          <g key={shape.id} data-shape={shape.kind}>
            {shape.kind === "box" ? (
              <rect
                x={shape.x}
                y={shape.y}
                width={shape.width}
                height={shape.height}
                fill="none"
                stroke={colour}
                strokeWidth={STROKE_PX}
                strokeDasharray={shape.dashed ? "4 3" : undefined}
                vectorEffect="non-scaling-stroke"
              />
            ) : (
              <polygon
                points={shape.points.map(([x, y]) => `${x},${y}`).join(" ")}
                fill={colour}
                fillOpacity={0.12}
                stroke={colour}
                strokeWidth={STROKE_PX}
                strokeDasharray={shape.dashed ? "4 3" : undefined}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {shape.label && anchor && (
              <ShapeLabel
                text={shape.label}
                x={anchor.x}
                y={anchor.y}
                px={px}
                colour={shape.labelColour ?? colour}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** A tag sitting on the shape's top-left corner, in the shape's colour. */
function ShapeLabel({
  text,
  x,
  y,
  px,
  colour,
}: {
  text: string;
  x: number;
  y: number;
  px: number;
  colour: string;
}) {
  const size = LABEL_PX * px;
  const pad = 3 * px;
  // Monospace digits at 0.62 em is close enough to size the tag without measuring text,
  // which an SVG in a transformed stage cannot do cheaply.
  const width = text.length * size * 0.62 + pad * 2;
  const height = size + pad * 2;
  return (
    <g transform={`translate(${x} ${y - height})`}>
      <rect width={width} height={height} fill={colour} rx={2 * px} />
      <text
        x={pad}
        y={pad + size * 0.82}
        fontSize={size}
        className="fill-ground font-mono"
      >
        {text}
      </text>
    </g>
  );
}

export function labelAnchor(shape: VectorShape): { x: number; y: number } | null {
  if (shape.kind === "box") return { x: shape.x, y: shape.y };
  if (shape.points.length === 0) return null;
  // The topmost vertex, leftmost among ties: where a reader looks for a tag.
  let best = shape.points[0] as readonly [number, number];
  for (const point of shape.points) {
    if (point[1] < best[1] || (point[1] === best[1] && point[0] < best[0])) best = point;
  }
  return { x: best[0], y: best[1] };
}
