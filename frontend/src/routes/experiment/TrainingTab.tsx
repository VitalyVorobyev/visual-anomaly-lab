/**
 * Watching a run, live and afterwards.
 *
 * The series come from `useJob`: a snapshot of everything the run has logged so far, then
 * the socket. Reload mid-run and the curves come back whole rather than restarting from
 * the moment the page opened — which is the difference between a chart and a decoration,
 * and the reason ADR-0020 exists.
 *
 * Losses go on a log axis by default. The three EfficientAD terms differ by more than an
 * order of magnitude for most of a run, and on a linear axis two of them are a flat line
 * along the bottom.
 */

import { useState } from "react";

import type { JobSummary } from "../../api/client";
import { LineChart } from "../../components/charts/LineChart";
import type { Series } from "../../hooks/useJob";
import { Empty, Panel } from "../../components/ui";

/** Anything not obviously a loss goes on its own linear chart — a rate, a count. */
function isLoss(name: string): boolean {
  return name.startsWith("loss");
}

/** Enough digits to be useful without pretending to a precision the number lacks. */
function formatScalar(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  return value.toPrecision(6).replace(/0+$/, "").replace(/\.$/, "");
}

export function TrainingTab({
  jobs,
  series,
  followingJobId,
}: {
  jobs: JobSummary[];
  series: Series[];
  followingJobId: number | undefined;
}) {
  const [logScale, setLogScale] = useState(true);

  const trainingJobs = jobs.filter((job) => job.kind === "train");
  if (trainingJobs.length === 0) {
    return (
      <Panel title="Training">
        <Empty>Nothing has trained yet.</Empty>
      </Panel>
    );
  }

  if (series.length === 0) {
    return (
      <Panel title="Training">
        <Empty>
          This run recorded no scalar series. A method reports them by calling{" "}
          <span className="font-mono">ctx.metric</span> during <span className="font-mono">fit</span>.
        </Empty>
      </Panel>
    );
  }

  // A series with one point is a *value*, not a curve. `pixel_reference` reports
  // `reference_images = 128` once and nothing else; plotting that as a scatter of one dot
  // against a step axis running -1 to 1 is a chart of nothing, and reads as a run that
  // failed to record anything. Printed as a number it is exactly the fact it was.
  const scalars = series.filter((entry) => entry.points.length === 1);
  const curves = series.filter((entry) => entry.points.length > 1);
  const losses = curves.filter((entry) => isLoss(entry.name));
  const others = curves.filter((entry) => !isLoss(entry.name));
  const truncated = series.filter((entry) => entry.dropped > 0);

  return (
    <div className="flex flex-col gap-6">
      {losses.length > 0 && (
        <Panel
          title="Losses"
          actions={
            <label className="flex items-center gap-2 text-xs text-slate-500">
              <input
                type="checkbox"
                aria-label="Logarithmic loss axis"
                checked={logScale}
                onChange={(event) => setLogScale(event.target.checked)}
              />
              log axis
            </label>
          }
        >
          <LineChart
            label="Training losses per step"
            xLabel="step"
            yLabel="loss"
            variant="wide"
            logY={logScale}
            series={losses.map((entry) => ({
              name: entry.name,
              points: entry.points.map((point) => ({ x: point.step, y: point.value })),
            }))}
          />
        </Panel>
      )}

      {others.map((entry) => (
        <Panel key={entry.name} title={entry.name}>
          <LineChart
            label={`${entry.name} per step`}
            xLabel="step"
            variant="wide"
            showLegend={false}
            series={[
              {
                name: entry.name,
                points: entry.points.map((point) => ({ x: point.step, y: point.value })),
              },
            ]}
          />
        </Panel>
      ))}

      {scalars.length > 0 && (
        <Panel title="Reported once">
          <dl className="grid gap-4 sm:grid-cols-3">
            {scalars.map((entry) => (
              <div key={entry.name} className="flex flex-col">
                <dt className="text-xs text-slate-500 dark:text-slate-400">{entry.name}</dt>
                <dd className="font-mono text-sm">{formatScalar(entry.points[0]?.value)}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}

      {truncated.length > 0 && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {/* A silent truncation would read as "this is all there was". */}
          Downsampled for drawing:{" "}
          {truncated
            .map((entry) => `${entry.name} (${entry.points.length} of ${entry.total})`)
            .join(", ")}
          .
        </p>
      )}

      {followingJobId === undefined && (
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Showing the most recent training run. Pick another from the Runs list on Overview.
        </p>
      )}
    </div>
  );
}
