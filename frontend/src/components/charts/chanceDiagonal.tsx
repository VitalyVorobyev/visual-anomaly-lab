import type { Scale } from "@vitavision/charts";

/** The line a coin-flip classifier would draw, for the eye to measure a curve against. */
export function chanceDiagonal(x: Scale, y: Scale) {
  return (
    <line
      x1={x.project(0)}
      y1={y.project(0)}
      x2={x.project(1)}
      y2={y.project(1)}
      stroke="currentColor"
      strokeWidth={0.75}
      strokeDasharray="4 3"
      opacity={0.4}
    />
  );
}
