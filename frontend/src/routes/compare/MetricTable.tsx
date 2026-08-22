/**
 * N runs as N columns, one metric per row.
 *
 * Two tables, and the split between them is the point of ADR-0028. The first holds the
 * threshold-independent metrics, which are functions of the *ranking* and therefore
 * comparable across methods directly — that is why the literature reports them. The second
 * holds everything that depends on a threshold, and every one of its columns is at a
 * different cut, because a score has no meaning outside its own run.
 *
 * So the second table prints each run's own threshold and the sentence that produced it,
 * in the header, above the counts. Take those away and it becomes the misleading table the
 * record rejected: N true confusion matrices at operating points nobody chose, looking
 * exactly like a fair comparison.
 */

import type { ComparedRun, OperatingPoint } from "../../api/client";
import type { MetricValue } from "../../api/metrics";
import {
  comparisonRows,
  detectionRows,
  formatScore,
  groupingNote,
  pixelRows,
  timingRows,
} from "../../api/metrics";
import type { ComparisonRow } from "../../api/metrics";
import { InfoHint, Panel, cn } from "@vitavision/lab-ui";

export function MetricTable({ runs }: { runs: ComparedRun[] }) {
  const metrics = runs.map((run) => (run.metrics ?? {}) as MetricValue);
  const sections: { title: string; rows: ComparisonRow[] }[] = [
    { title: "Detection", rows: comparisonRows(metrics.map(detectionRows)) },
    { title: "Pixel level", rows: comparisonRows(metrics.map(pixelRows)) },
    { title: "Timing", rows: comparisonRows(metrics.map(timingRows)) },
  ].filter((section) => section.rows.length > 0);

  /* Stated rather than left as a gap. On a dataset of single-image samples the image-level
     rows are the sample-level rows redrawn, so they are dropped — and a reader who knows
     they exist would otherwise wonder which run failed to report them. Only when it holds
     for every run: one grouped run in the selection is a reason to show both levels. */
  const ungrouped = metrics.every((entry) => groupingNote(entry) !== null)
    ? groupingNote(metrics[0] ?? {})
    : null;

  return (
    <Panel title="Threshold-independent">
      <p className="mb-3 text-xs text-fg-muted">
        Functions of the ranking, not of the score values, so these compare across methods
        directly. Read from the metric sets the scoring runs stored — nothing here is
        recomputed.
      </p>
      <Grid runs={runs}>
        {sections.map((section) => (
          <SectionRows key={section.title} title={section.title} rows={section.rows} />
        ))}
      </Grid>
      {ungrouped !== null && <p className="mt-3 text-xs text-fg-subtle">{ungrouped}</p>}
    </Panel>
  );
}

export function OperatingTable({
  runs,
  operatingPoint,
  recallTarget,
}: {
  runs: ComparedRun[];
  operatingPoint: OperatingPoint;
  recallTarget: number;
}) {
  const rows: ComparisonRow[] = [
    {
      key: "threshold",
      label: "Threshold",
      hint: "In this run's own units. Never comparable with the column beside it.",
      values: runs.map((run) =>
        run.threshold === null || run.threshold === undefined ? null : run.threshold.toFixed(4),
      ),
    },
    counts("true_positive", "True positives"),
    counts("false_positive", "False positives"),
    counts("true_negative", "True negatives"),
    counts("false_negative", "False negatives"),
    rates("precision", "Precision"),
    rates("recall", "Recall"),
    rates("f1", "F1"),
    rates("accuracy", "Accuracy"),
  ];

  function counts(key: keyof NonNullable<ComparedRun["confusion"]>, label: string): ComparisonRow {
    return {
      key,
      label,
      values: runs.map((run) => (run.confusion ? String(run.confusion[key]) : null)),
    };
  }

  function rates(
    key: "precision" | "recall" | "f1" | "accuracy",
    label: string,
  ): ComparisonRow {
    return { key, label, values: runs.map((run) => formatScore(run[key])) };
  }

  return (
    <Panel title="At each run's own operating point">
      <p className="mb-3 text-xs text-fg-muted">
        {operatingPoint === "f1" ? (
          <>
            Every run at <strong className="font-medium text-fg">its own F1 optimum</strong> —
            each method at its best. Fitted with this subset&rsquo;s labels, so read it as an
            upper bound rather than as field performance.
          </>
        ) : (
          <>
            Every run at the highest cut still reaching{" "}
            <strong className="font-medium text-fg">
              {(recallTarget * 100).toFixed(0)}% recall
            </strong>{" "}
            — the same detection rate, so the comparison is of what that rate costs in false
            alarms.
          </>
        )}{" "}
        The thresholds below are in each run&rsquo;s own units and are not comparable with
        each other; that is why they are printed.
      </p>
      <Grid runs={runs} rationale>
        <SectionRows rows={rows} />
      </Grid>
    </Panel>
  );
}

/**
 * The shared frame: one header column of labels, one column per run.
 *
 * Hand-built rather than the `Table` primitive because the axes are transposed — here a
 * *row* is a metric and a *column* is a run, so the header carries run identity and every
 * cell in a row is the same quantity. `Table` builds the other way round.
 */
function Grid({
  runs,
  rationale = false,
  children,
}: {
  runs: ComparedRun[];
  /** Print each run's threshold rationale under its name. */
  rationale?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-line align-bottom">
            <th className="py-2 pr-4 text-xs font-medium text-fg-muted">Metric</th>
            {runs.map((run) => (
              // Right-aligned to match the cells beneath it. Left-aligned, a wide table
              // puts a run's name a column's width away from its own numbers.
              <th key={run.id} className="min-w-32 px-3 py-2 text-right">
                <span className="block truncate text-sm font-semibold text-fg">{run.name}</span>
                <span className="block font-mono text-[11px] text-fg-subtle">
                  {run.model_type}
                </span>
                {rationale && (
                  <span className="mt-1 block text-[11px] leading-snug font-normal text-fg-muted">
                    {run.threshold_rationale || "—"}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        {children}
      </table>
    </div>
  );
}

function SectionRows({ title, rows }: { title?: string; rows: ComparisonRow[] }) {
  return (
    <tbody>
      {title && (
        <tr>
          <th
            colSpan={1 + (rows[0]?.values.length ?? 0)}
            className="pt-4 pb-1 text-left text-xs font-semibold text-fg"
          >
            {title}
          </th>
        </tr>
      )}
      {rows.map((row) => (
        <MetricRowCells key={row.key} row={row} />
      ))}
    </tbody>
  );
}

function MetricRowCells({ row }: { row: ComparisonRow }) {
  // Highest wins for everything here, and there is nothing here where it does not:
  // rates, areas and counts of correct answers. Timing and the confusion's error counts
  // are deliberately left unmarked — "best" would be lowest there, and a table that marks
  // two directions with one colour teaches nothing.
  const best = bestIndex(row);

  return (
    <tr className="border-b border-line/60 last:border-0">
      <th className="py-1.5 pr-4 text-left text-xs font-normal text-fg-muted">
        {row.label}
        {row.hint && <InfoHint>{row.hint}</InfoHint>}
      </th>
      {row.values.map((value, index) => (
        <td
          key={index}
          className={cn(
            "px-3 py-1.5 text-right font-mono text-sm tabular-nums",
            index === best && "font-semibold text-signal",
          )}
        >
          {/* A metric that could not be computed stays a dash. Never 0.000. */}
          {value ?? <span className="text-fg-subtle">—</span>}
        </td>
      ))}
    </tr>
  );
}

/** Which column holds the leading value, or `-1` when the row is not a "higher is better". */
const HIGHER_IS_BETTER = new Set([
  "sample_roc_auc",
  "sample_average_precision",
  "image_roc_auc",
  "image_average_precision",
  "pixel_roc_auc",
  "au_pro",
  "precision",
  "recall",
  "f1",
  "accuracy",
]);

function bestIndex(row: ComparisonRow): number {
  if (!HIGHER_IS_BETTER.has(row.key)) return -1;
  let best = -1;
  let peak = -Infinity;
  let ties = 0;
  row.values.forEach((value, index) => {
    if (value === null) return;
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    if (numeric > peak) {
      peak = numeric;
      best = index;
      ties = 1;
    } else if (numeric === peak) {
      ties += 1;
    }
  });
  // Marking one of two identical numbers as the winner is a claim the data does not make.
  return ties > 1 ? -1 : best;
}
