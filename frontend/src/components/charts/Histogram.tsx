/**
 * Score distribution by class, with the threshold drawn on it.
 *
 * This is the chart that makes a threshold decision legible: the confusion matrix says
 * how many were wrong, and this says *why* — two distributions that barely separate, or
 * one long normal tail crossing the line.
 *
 * Both classes share one set of bin edges, computed over the union of their scores. Binned
 * separately they would be drawn on different axes and the overlap — the only thing worth
 * looking at — would be an artefact of the binning.
 */

import { Frame, areaFor } from "./Frame";
import { extent, histogram, linearScale } from "./scale";

// The histogram is always given a whole panel, never a column of a grid.
const plotArea = areaFor("wide");

/** Emerald for normal, red for defect — the same language `ui.tsx` uses for tone. */
const NORMAL_FILL = "#10b981";
const DEFECT_FILL = "#ef4444";

export interface HistogramProps {
  normal: number[];
  defect: number[];
  threshold?: number;
  bins?: number;
  label: string;
}

export function ScoreHistogram({
  normal,
  defect,
  threshold,
  bins = 32,
  label,
}: HistogramProps) {
  const domain = extent([...normal, ...defect]);
  const normalBins = histogram(normal, domain, bins);
  const defectBins = histogram(defect, domain, bins);
  const tallest = Math.max(1, ...normalBins, ...defectBins);

  const xScale = linearScale(domain, plotArea.x0, plotArea.x1);
  const yScale = linearScale([0, tallest], plotArea.y0, plotArea.y1);
  const binWidth = (plotArea.x1 - plotArea.x0) / bins;

  const bars = (counts: number[], fill: string, opacity: number) =>
    counts.map((count, index) =>
      count === 0 ? null : (
        <rect
          key={`${fill}-${index}`}
          x={plotArea.x0 + index * binWidth}
          y={yScale.project(count)}
          width={Math.max(0.5, binWidth - 0.5)}
          height={plotArea.y0 - yScale.project(count)}
          fill={fill}
          opacity={opacity}
        />
      ),
    );

  return (
    <Frame
      xScale={xScale}
      yScale={yScale}
      xLabel="anomaly score"
      yLabel="samples"
      label={label}
      variant="wide"
      footer={
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <li className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-2 w-3 rounded-sm"
              style={{ backgroundColor: NORMAL_FILL, opacity: 0.75 }}
            />
            <span className="font-mono text-xs">normal ({normal.length})</span>
          </li>
          <li className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-2 w-3 rounded-sm"
              style={{ backgroundColor: DEFECT_FILL, opacity: 0.75 }}
            />
            <span className="font-mono text-xs">defect ({defect.length})</span>
          </li>
          {threshold !== undefined && (
            <li className="font-mono text-xs">threshold {threshold.toFixed(4)}</li>
          )}
        </ul>
      }
    >
      {/* Normals first and defects over them: the defect bars are the smaller population
          in every realistic split, so drawing them last keeps them from being buried. */}
      {bars(normalBins, NORMAL_FILL, 0.65)}
      {bars(defectBins, DEFECT_FILL, 0.65)}

      {threshold !== undefined && Number.isFinite(threshold) && (
        <line
          x1={xScale.project(threshold)}
          x2={xScale.project(threshold)}
          y1={plotArea.y0}
          y2={plotArea.y1}
          stroke="currentColor"
          strokeWidth={1.25}
          strokeDasharray="3 2"
        />
      )}
    </Frame>
  );
}
