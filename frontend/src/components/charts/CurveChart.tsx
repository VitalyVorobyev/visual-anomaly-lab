/**
 * ROC and PR curves.
 *
 * Both axes are rates, so both domains are fixed at `[0, 1]` rather than fitted to the
 * data: a ROC curve auto-scaled to its own extent is a different picture of the same
 * model every time, and the whole value of the chart is that its shape is comparable.
 *
 * A curve the backend could not compute arrives as `null` and renders as a stated absence.
 * Drawing the chance diagonal alone would be a picture of a model that guesses, which is
 * not what "this subset has one class in it" means.
 */

import type { Curve } from "../../api/client";
import { Empty, LineChart, type Scale } from "@vitavision/lab-ui";

export interface CurveChartProps {
  curve: Curve | null | undefined;
  kind: "roc" | "pr";
  label: string;
  /** The scalar this curve integrates to, shown beside it when it exists. */
  area?: number | null;
  areaLabel?: string;
  /** Why the curve is missing, when it is. */
  absent?: string;
}

export function CurveChart({ curve, kind, label, area, areaLabel, absent }: CurveChartProps) {
  if (!curve || curve.x.length === 0) {
    return <Empty>{absent ?? "Not enough labelled data in this subset to draw a curve."}</Empty>;
  }

  const points = curve.x.map((x, index) => ({ x, y: curve.y[index] ?? 0 }));
  const dropped =
    curve.dropped > 0
      ? `${curve.total} points, downsampled to ${curve.x.length} for drawing`
      : null;

  return (
    <LineChart
      label={label}
      series={[{ name: kind === "roc" ? "ROC" : "PR", points }]}
      xLabel={kind === "roc" ? "false-positive rate" : "recall"}
      yLabel={kind === "roc" ? "true-positive rate" : "precision"}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      showLegend={false}
      underlay={kind === "roc" ? chanceDiagonal : undefined}
      footer={
        <div className="flex flex-wrap gap-x-4">
          {area !== undefined && (
            <span className="font-mono">
              {areaLabel ?? (kind === "roc" ? "AUC" : "AP")}{" "}
              {/* A metric that could not be computed renders as a dash, never as 0.000. */}
              {area === null ? <span className="text-fg-subtle">—</span> : area.toFixed(4)}
            </span>
          )}
          {dropped && <span>{dropped}</span>}
        </div>
      }
    />
  );
}

/** The line a coin-flip classifier would draw, for the eye to measure the curve against. */
function chanceDiagonal(x: Scale, y: Scale) {
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
