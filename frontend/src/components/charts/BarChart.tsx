/**
 * Horizontal stacked bars — the per-defect-type breakdown.
 *
 * Horizontal because the categories are free text an adapter recorded (`Nick`, `Scratch`,
 * a VisA defect label), and a vertical axis would either truncate them or turn them
 * sideways. Stacked because the question is "of this defect type, how many did the model
 * catch", which is a proportion of a known total rather than two independent counts.
 */

export interface BarRow {
  label: string;
  /** Segments in draw order, left to right. */
  segments: { name: string; value: number; colour: string }[];
}

export function StackedBars({ rows, label }: { rows: BarRow[]; label: string }) {
  const widest = Math.max(
    1,
    ...rows.map((row) => row.segments.reduce((total, segment) => total + segment.value, 0)),
  );

  return (
    <div className="flex flex-col gap-2" role="table" aria-label={label}>
      {rows.map((row) => {
        const total = row.segments.reduce((sum, segment) => sum + segment.value, 0);
        return (
          <div key={row.label} className="flex items-center gap-3 text-xs" role="row">
            <span className="w-28 shrink-0 truncate font-mono" role="rowheader" title={row.label}>
              {row.label}
            </span>
            <div
              className="flex h-4 flex-1 overflow-hidden rounded bg-slate-100 dark:bg-slate-800"
              role="cell"
            >
              {row.segments.map((segment) =>
                segment.value === 0 ? null : (
                  <div
                    key={segment.name}
                    className="h-full"
                    style={{
                      width: `${(segment.value / widest) * 100}%`,
                      backgroundColor: segment.colour,
                    }}
                    title={`${segment.name}: ${segment.value}`}
                  />
                ),
              )}
            </div>
            <span className="w-24 shrink-0 text-right font-mono text-slate-500" role="cell">
              {row.segments
                .filter((segment) => segment.value > 0)
                .map((segment) => `${segment.value} ${segment.name}`)
                .join(" · ") || `${total}`}
            </span>
          </div>
        );
      })}
    </div>
  );
}
