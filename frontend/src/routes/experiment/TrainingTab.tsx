/**
 * Watching a run, live and afterwards.
 *
 * The series come from `useJob`: a snapshot of everything the run has logged so far, then
 * the socket. Reload mid-run and the curves come back whole rather than restarting from
 * the moment the page opened — which is the difference between a chart and a decoration,
 * and the reason ADR-0020 exists.
 *
 * The console lives here too, beside the chart it belongs to. It used to be on Overview,
 * one tab away from the curves describing the same run — so watching a training run meant
 * choosing between seeing the loss fall and seeing what the process was saying. Everything
 * arrives as props from the one `useJob` the route already holds: **this tab issues no
 * requests of its own**, which is what keeps a second socket from opening when it mounts.
 *
 * Losses go on a log axis by default. The three EfficientAD terms differ by more than an
 * order of magnitude for most of a run, and on a linear axis two of them are a flat line
 * along the bottom.
 */

import { useState } from "react";

import type { JobDetail, JobSummary } from "../../api/client";
import { LineChart } from "../../components/charts/LineChart";
import { JobProgress } from "../../components/JobProgress";
import type { Series } from "../../hooks/useJob";
import { Disclosure, Empty, Panel, Switch } from "../../components/ui";

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

export interface TrainingConsole {
  jobId: number | undefined;
  job: JobDetail | undefined;
  lines: string[];
  error: Error | null;
}

export function TrainingTab({
  jobs,
  series,
  followingJobId,
  console: live,
}: {
  jobs: JobSummary[];
  series: Series[];
  followingJobId: number | undefined;
  console: TrainingConsole;
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
      <div className="flex flex-col gap-6">
        <Panel title="Training">
          <Empty>
            This run recorded no scalar series. A method reports them by calling{" "}
            <span className="font-mono">ctx.metric</span> during{" "}
            <span className="font-mono">fit</span>.
          </Empty>
        </Panel>
        <ConsolePanel console={live} />
      </div>
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
            <Switch
              checked={logScale}
              onCheckedChange={setLogScale}
              label={<span className="text-xs text-fg-muted">log axis</span>}
            />
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
                <dt className="text-xs text-fg-muted">{entry.name}</dt>
                <dd className="font-mono text-sm">{formatScalar(entry.points[0]?.value)}</dd>
              </div>
            ))}
          </dl>
        </Panel>
      )}

      {truncated.length > 0 && (
        <p className="text-xs text-fg-muted">
          {/* A silent truncation would read as "this is all there was". */}
          Downsampled for drawing:{" "}
          {truncated
            .map((entry) => `${entry.name} (${entry.points.length} of ${entry.total})`)
            .join(", ")}
          .
        </p>
      )}

      {followingJobId === undefined && (
        <p className="text-xs text-fg-muted">
          Showing the most recent training run. Pick another from the Runs list on the Jobs
          &amp; files tab.
        </p>
      )}

      <ConsolePanel console={live} />
    </div>
  );
}

/**
 * The run's own output, folded away.
 *
 * Collapsed by default because the chart is the answer and the log is the follow-up
 * question — but it is on this tab, one click from the curve, rather than on the screen
 * that is supposed to be about results.
 */
function ConsolePanel({ console: live }: { console: TrainingConsole }) {
  if (live.jobId === undefined) return null;
  return (
    <Panel title={`Job #${live.jobId}`}>
      <Disclosure summary="Console">
        <JobProgress
          jobId={live.jobId}
          job={live.job}
          lines={live.lines}
          error={live.error}
        />
      </Disclosure>
    </Panel>
  );
}
