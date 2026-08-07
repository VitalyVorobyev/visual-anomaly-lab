/**
 * The plot frame every chart here composes: axes, ticks, grid, labels.
 *
 * Charts are drawn into a fixed `viewBox` and scaled by CSS, so one component works at
 * any panel width without measuring the DOM. The consequence is that font sizes are in
 * viewBox units and therefore scale with the chart: the *same* 10-unit label renders at
 * 9 px in a half-width panel and at 21 px across a full-width one, which looks like two
 * different design systems on one screen.
 *
 * So the viewBox is chosen to roughly match the width the chart will actually be given.
 * `panel` is for the two-column grids, `wide` for a chart that spans a panel on its own.
 * That keeps the scale factor near 1 in both, and it is why callers pick a variant rather
 * than a pixel width — the variant says where the chart is going, not how big to draw it.
 */

import type { ReactNode } from "react";

import type { Scale } from "./scale";

export type Variant = "panel" | "wide";

const GEOMETRY = {
  panel: { width: 480, height: 280 },
  wide: { width: 960, height: 320 },
} as const;

const MARGIN = { left: 56, right: 12, top: 12, bottom: 34 } as const;

/** The drawable rectangle for one variant, in that variant's viewBox units. */
export function areaFor(variant: Variant) {
  const { width, height } = GEOMETRY[variant];
  return {
    x0: MARGIN.left,
    x1: width - MARGIN.right,
    y0: height - MARGIN.bottom,
    y1: MARGIN.top,
    width,
    height,
  };
}

/** The default geometry, for callers that do not care which variant they are in. */
export const plotArea = areaFor("panel");

/** Slate at the two ends of the neutral ramp, matching `ui.tsx`'s chrome colours. */
const AXIS = "currentColor";

export interface FrameProps {
  xScale: Scale;
  yScale: Scale;
  xLabel?: string;
  yLabel?: string;
  xTicks?: number;
  yTicks?: number;
  /** Drawn inside the plot area, above the grid and below nothing. */
  children?: ReactNode;
  /** Accessible name — every chart must have one; it is what the tests query on. */
  label: string;
  /** Rendered under the plot: a legend, a caption, a truncation note. */
  footer?: ReactNode;
  /** Where this chart is going, so its text renders at the same size as its neighbours. */
  variant?: Variant;
}

export function Frame({
  xScale,
  yScale,
  xLabel,
  yLabel,
  xTicks = 5,
  yTicks = 4,
  children,
  label,
  footer,
  variant = "panel",
}: FrameProps) {
  const horizontal = xScale.ticks(xTicks);
  const vertical = yScale.ticks(yTicks);
  const plotArea = areaFor(variant);
  const PLOT = GEOMETRY[variant];

  return (
    <figure className="flex flex-col gap-1 text-slate-500 dark:text-slate-400">
      <svg
        role="img"
        aria-label={label}
        viewBox={`0 0 ${PLOT.width} ${PLOT.height}`}
        className="w-full"
      >
        {vertical.map((tick) => {
          const y = yScale.project(tick.value);
          return (
            <g key={`y-${tick.value}`}>
              <line
                x1={plotArea.x0}
                x2={plotArea.x1}
                y1={y}
                y2={y}
                stroke={AXIS}
                strokeWidth={0.5}
                opacity={0.25}
              />
              <text x={plotArea.x0 - 6} y={y + 3.5} textAnchor="end" fontSize={10} fill={AXIS}>
                {tick.label}
              </text>
            </g>
          );
        })}

        {horizontal.map((tick) => {
          const x = xScale.project(tick.value);
          return (
            <g key={`x-${tick.value}`}>
              <line
                x1={x}
                x2={x}
                y1={plotArea.y0}
                y2={plotArea.y1}
                stroke={AXIS}
                strokeWidth={0.5}
                opacity={0.15}
              />
              <text x={x} y={plotArea.y0 + 14} textAnchor="middle" fontSize={10} fill={AXIS}>
                {tick.label}
              </text>
            </g>
          );
        })}

        <line
          x1={plotArea.x0}
          x2={plotArea.x1}
          y1={plotArea.y0}
          y2={plotArea.y0}
          stroke={AXIS}
          strokeWidth={1}
          opacity={0.6}
        />
        <line
          x1={plotArea.x0}
          x2={plotArea.x0}
          y1={plotArea.y0}
          y2={plotArea.y1}
          stroke={AXIS}
          strokeWidth={1}
          opacity={0.6}
        />

        {children}

        {xLabel && (
          <text
            x={(plotArea.x0 + plotArea.x1) / 2}
            y={PLOT.height - 2}
            textAnchor="middle"
            fontSize={10}
            fill={AXIS}
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            transform={`translate(10 ${(plotArea.y0 + plotArea.y1) / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={10}
            fill={AXIS}
          >
            {yLabel}
          </text>
        )}
      </svg>
      {footer && <figcaption className="text-xs">{footer}</figcaption>}
    </figure>
  );
}

/**
 * A legend row. Kept beside the frame rather than inside the SVG so the swatches use the
 * same Tailwind colours as the rest of the UI instead of a second palette in viewBox units.
 */
export function Legend({ items }: { items: { label: string; colour: string }[] }) {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-0.5 w-4 rounded"
            style={{ backgroundColor: item.colour }}
          />
          <span className="font-mono text-xs">{item.label}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The shared series palette.
 *
 * Ordered so the first three — the EfficientAD loss terms — are distinguishable to a
 * colourblind reader, and picked from the same families `ui.tsx` already uses for tone.
 */
export const SERIES_COLOURS = [
  "#0ea5e9",
  "#f59e0b",
  "#8b5cf6",
  "#10b981",
  "#ef4444",
  "#64748b",
] as const;

export function seriesColour(index: number): string {
  return SERIES_COLOURS[index % SERIES_COLOURS.length] ?? "#64748b";
}
