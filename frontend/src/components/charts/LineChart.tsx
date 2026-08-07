/**
 * Multi-series line chart — the training curves, and the base of the ROC/PR chart.
 *
 * A series with no points draws nothing but still claims its legend entry, so a loss term
 * that has not reported yet is visibly pending rather than silently absent. A series with
 * one point draws a dot: at the start of a training run that is genuinely all there is,
 * and an empty chart would read as "nothing is happening".
 */

import { Frame, Legend, areaFor, seriesColour } from "./Frame";
import type { Variant } from "./Frame";
import type { Scale } from "./scale";
import { extent, linePath, linearScale, logScale } from "./scale";

export interface Series {
  name: string;
  points: { x: number; y: number }[];
  colour?: string;
}

export interface LineChartProps {
  series: Series[];
  label: string;
  xLabel?: string;
  yLabel?: string;
  /** Losses span decades; a linear axis flattens two of the three onto the floor. */
  logY?: boolean;
  /** Fixed domains, for charts whose axes are known — ROC and PR are both `[0, 1]`. */
  xDomain?: [number, number];
  yDomain?: [number, number];
  /** Drawn behind the series, in plot pixels. The chance diagonal uses this. */
  underlay?: (x: Scale, y: Scale) => React.ReactNode;
  footer?: React.ReactNode;
  showLegend?: boolean;
  variant?: Variant;
}

export function LineChart({
  series,
  label,
  xLabel,
  yLabel,
  logY = false,
  xDomain,
  yDomain,
  underlay,
  footer,
  showLegend = true,
  variant = "panel",
}: LineChartProps) {
  const allX = series.flatMap((entry) => entry.points.map((point) => point.x));
  const allY = series.flatMap((entry) => entry.points.map((point) => point.y));
  const plotArea = areaFor(variant);

  const xScale = linearScale(xDomain ?? extent(allX), plotArea.x0, plotArea.x1);
  const makeY = logY ? logScale : linearScale;
  const yScale = makeY(yDomain ?? extent(allY), plotArea.y0, plotArea.y1);

  const coloured = series.map((entry, index) => ({
    ...entry,
    colour: entry.colour ?? seriesColour(index),
  }));

  return (
    <Frame
      xScale={xScale}
      yScale={yScale}
      xLabel={xLabel}
      yLabel={yLabel}
      label={label}
      variant={variant}
      footer={
        <div className="flex flex-col gap-1">
          {showLegend && (
            <Legend
              items={coloured.map((entry) => ({ label: entry.name, colour: entry.colour }))}
            />
          )}
          {footer}
        </div>
      }
    >
      {underlay?.(xScale, yScale)}
      {coloured.map((entry) =>
        entry.points.length === 1 ? (
          <circle
            key={entry.name}
            cx={xScale.project(entry.points[0]?.x ?? 0)}
            cy={yScale.project(entry.points[0]?.y ?? 0)}
            r={2.5}
            fill={entry.colour}
          />
        ) : (
          <path
            key={entry.name}
            d={linePath(entry.points, xScale, yScale)}
            fill="none"
            stroke={entry.colour}
            strokeWidth={1.5}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
        ),
      )}
    </Frame>
  );
}
