/**
 * The experiment's own state: what has run, what it was configured with, and the
 * threshold-free numbers it produced.
 *
 * This is M3's screen, moved intact. The M4 tabs sit beside it rather than replacing it —
 * a run's log and its metric table are what someone opens the page for most of the time,
 * and burying them under a tab that is not the default would be a regression dressed as a
 * feature.
 */

import type { MetricValue } from "../../api/metrics";
import { caveats, detectionRows, groupingNote, pixelRows, timingRows } from "../../api/metrics";
import type { MetricRow } from "../../api/metrics";
import type { JobSummary, Subset } from "../../api/client";
import { JobProgress } from "../../components/JobProgress";
import { Badge, Button, CountRun, Disclosure, Empty, ErrorBox, Panel } from "../../components/ui";
import type { Tone } from "../../components/ui";
import { useJob, isTerminal } from "../../hooks/useJob";
import { useReevaluate, useStartRun } from "../../hooks/useExperiments";

export function Runs({
  experimentId,
  jobs,
  onFollow,
  followingJobId,
}: {
  experimentId: number;
  jobs: JobSummary[];
  onFollow: (jobId: number) => void;
  followingJobId: number | undefined;
}) {
  const start = useStartRun(experimentId);
  const busy = jobs.some((job) => !isTerminal(job.status));

  const run = (kind: "train" | "infer") =>
    start.mutate({ kind }, { onSuccess: (job) => onFollow(job.id) });

  return (
    <Panel
      title="Runs"
      actions={
        <div className="flex gap-2">
          <Button variant="primary" disabled={busy || start.isPending} onClick={() => run("train")}>
            Train
          </Button>
          <Button disabled={busy || start.isPending} onClick={() => run("infer")}>
            Score &amp; evaluate
          </Button>
        </div>
      }
    >
      {start.error && <ErrorBox>{start.error.message}</ErrorBox>}
      {jobs.length === 0 && <Empty>Nothing has run yet. Train first, then score.</Empty>}

      {jobs.length > 0 && (
        <ul className="divide-y divide-line text-sm ">
          {jobs.map((job) => (
            <li key={job.id} className="flex items-center gap-3 py-2">
              <span className="font-mono text-xs text-fg-muted">#{job.id}</span>
              <span className="w-16">{job.kind}</span>
              <Badge tone={jobTone(job.status)}>{job.status}</Badge>
              <span className="truncate text-xs text-fg-muted">
                {job.message ?? job.error ?? ""}
              </span>
              {job.id !== followingJobId && (
                <button
                  type="button"
                  className="ml-auto text-xs text-fg-muted hover:underline"
                  onClick={() => onFollow(job.id)}
                >
                  show log
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

export function jobTone(status: string): Tone {
  if (status === "succeeded") return "normal";
  if (status === "failed") return "defect";
  if (status === "running") return "info";
  return "neutral";
}

export function Console({ jobId, experimentId }: { jobId: number; experimentId: number }) {
  // The experiment id is what makes the screen refresh itself when this job ends: the
  // console is the one thing following the socket, so its terminal frame is the only
  // moment anything here knows the run is over.
  const { job, lines, error } = useJob(jobId, { experimentId });
  return (
    <Panel title={`Job #${jobId}`}>
      <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
    </Panel>
  );
}

export function Configuration({
  detail,
}: {
  detail: { config: Record<string, unknown>; preprocessing: Record<string, unknown> };
}) {
  return (
    <Panel title="Configuration">
      {/* Frozen into the experiment at creation, so this is what the run *used* rather than
          what the form currently defaults to. Worth reading before comparing two runs. */}
      <Disclosure summary="What this run was configured with">
        <div className="grid gap-5 sm:grid-cols-2">
          <ConfigBlock title="Method" values={detail.config} />
          <ConfigBlock title="Preprocessing" values={detail.preprocessing} />
        </div>
      </Disclosure>
    </Panel>
  );
}

function ConfigBlock({ title, values }: { title: string; values: Record<string, unknown> }) {
  const entries = Object.entries(values);
  return (
    <div className="flex flex-col gap-1.5">
      <h3 className="text-xs font-semibold text-fg">{title}</h3>
      {entries.length === 0 ? (
        <p className="text-xs text-fg-muted">Every option left at its default.</p>
      ) : (
        <dl className="flex flex-col gap-1 font-mono text-xs">
          {entries.map(([key, value]) => (
            <div key={key} className="flex items-baseline justify-between gap-3 border-b border-line/60 pb-1 last:border-0">
              <dt className="text-fg-muted">{key}</dt>
              <dd className="text-fg tabular-nums">{JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

export function Metrics({
  experimentId,
  metrics,
  aggregation,
}: {
  experimentId: number;
  metrics: { subset: Subset; metrics: Record<string, unknown> }[];
  aggregation: string;
}) {
  const reevaluate = useReevaluate(experimentId);

  return (
    <Panel
      title="Metrics"
      actions={
        <Button disabled={reevaluate.isPending} onClick={() => reevaluate.mutate()}>
          Recompute
        </Button>
      }
    >
      <p className="mb-3 text-xs text-fg-muted">
        Threshold-independent, computed from stored scores. Channels aggregate to a part by{" "}
        <span className="font-mono">{aggregation}</span>; recomputing re-reads without
        re-running inference.
      </p>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {metrics.map((entry) => (
          <SubsetMetrics key={entry.subset} subset={entry.subset} metrics={entry.metrics} />
        ))}
      </div>
    </Panel>
  );
}

function SubsetMetrics({ subset, metrics }: { subset: Subset; metrics: MetricValue }) {
  const counts = (metrics.samples ?? {}) as Record<string, number>;
  const notes = caveats(metrics);
  const ungrouped = groupingNote(metrics);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        {subset}
        <CountRun
          counts={[
            ["normal", counts.normal ?? 0, "normal"],
            ["defect", counts.defect ?? 0, "defect"],
            ["unlabeled", counts.unlabeled ?? 0, "unlabeled"],
          ]}
        />
      </h3>

      <MetricList rows={detectionRows(metrics)} />
      {/* Not a warning — an explanation of an absence, so it reads in the muted chrome
          colour rather than the verdict palette (ADR-0021). */}
      {ungrouped !== null && <p className="text-xs text-fg-subtle">{ungrouped}</p>}
      {pixelRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-semibold text-fg">
            Pixel level
          </h4>
          <MetricList rows={pixelRows(metrics)} />
        </>
      )}
      {timingRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-semibold text-fg">
            Timing
          </h4>
          <MetricList rows={timingRows(metrics)} />
        </>
      )}

      {notes.length > 0 && (
        <ul className="mt-1 flex flex-col gap-1">
          {notes.map((note) => (
            <li key={note} className="text-xs text-warn">
              {note}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function MetricList({ rows }: { rows: MetricRow[] }) {
  return (
    <dl className="flex flex-col gap-1">
      {rows.map((row) => (
        <div key={row.key} className="flex items-baseline gap-2" title={row.hint}>
          <dt className="text-xs text-fg-muted">{row.label}</dt>
          <dd className="ml-auto font-mono text-sm">
            {/* A metric that could not be computed stays visibly absent. Rendering it as
                0.000 would answer a question that should be asked. */}
            {row.value ?? <span className="text-fg-subtle">—</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
