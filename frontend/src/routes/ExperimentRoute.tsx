/**
 * One experiment: run it, watch it, and read what it found.
 *
 * The training console is `JobProgress` and `useJob`, unchanged from the import screen —
 * putting import on the job system in M2 was precisely so that training would need no
 * second progress mechanism, and this screen is where that pays.
 *
 * The threshold slider recomputes on the server from persisted scores. Nothing is stored
 * per threshold (ADR-0011), so dragging it is a filter over a few hundred floats.
 */

import { useEffect, useState } from "react";
import { Link, useParams } from "react-router";

import type { MetricValue } from "../api/metrics";
import { caveats, detectionRows, pixelRows, timingRows } from "../api/metrics";
import type { MetricRow } from "../api/metrics";
import type { JobSummary, SampleVerdict, Subset } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { Badge, Button, CountRun, Empty, ErrorBox, Panel } from "../components/ui";
import type { Tone } from "../components/ui";
import { useJob, isTerminal } from "../hooks/useJob";
import {
  useExperiment,
  useReevaluate,
  useResults,
  useStartRun,
  useThreshold,
} from "../hooks/useExperiments";

const OUTCOME_TONE: Record<string, Tone> = {
  tp: "normal",
  tn: "normal",
  fp: "warning",
  fn: "defect",
  unlabeled: "unlabeled",
};

const OUTCOME_LABEL: Record<string, string> = {
  tp: "true positive",
  tn: "true negative",
  fp: "false positive",
  fn: "false negative",
  unlabeled: "unlabeled",
};

export function ExperimentRoute() {
  const { experimentId: raw } = useParams();
  const experimentId = raw === undefined ? undefined : Number(raw);
  const experiment = useExperiment(experimentId);
  const [followingJobId, setFollowingJobId] = useState<number | undefined>();

  // A job that is still running when the screen opens should be followed without anyone
  // having to press anything — reload during training and the console picks up again.
  useEffect(() => {
    if (followingJobId !== undefined || !experiment.data) return;
    const live = experiment.data.jobs.find((job) => !isTerminal(job.status));
    if (live) setFollowingJobId(live.id);
  }, [experiment.data, followingJobId]);

  if (experiment.isPending) return <p className="text-sm text-slate-500">Loading…</p>;
  if (experiment.error) return <ErrorBox>{experiment.error.message}</ErrorBox>;
  if (!experiment.data || experimentId === undefined) return <Empty>No such experiment.</Empty>;

  const detail = experiment.data;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold tracking-tight">{detail.name}</h2>
        <Badge tone={detail.status === "trained" ? "normal" : "unlabeled"}>{detail.status}</Badge>
        <span className="font-mono text-xs text-slate-500">{detail.model_type}</span>
        <span className="text-xs text-slate-500">
          {detail.dataset_name} · split {detail.split_name}
        </span>
        <Link to="/experiments" className="ml-auto text-sm text-slate-500 hover:underline">
          All experiments
        </Link>
      </header>

      <Runs
        experimentId={experimentId}
        jobs={detail.jobs}
        onFollow={setFollowingJobId}
        followingJobId={followingJobId}
      />

      {followingJobId !== undefined && <Console jobId={followingJobId} />}

      <Configuration detail={detail} />

      {detail.metrics.length > 0 && (
        <Metrics
          experimentId={experimentId}
          metrics={detail.metrics}
          aggregation={String(detail.evaluation.aggregation ?? "max")}
        />
      )}

      {detail.scored_subsets.length > 0 && (
        <Results experimentId={experimentId} subsets={detail.scored_subsets} />
      )}
    </div>
  );
}

function Runs({
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
        <ul className="divide-y divide-slate-200 text-sm dark:divide-slate-700">
          {jobs.map((job) => (
            <li key={job.id} className="flex items-center gap-3 py-2">
              <span className="font-mono text-xs text-slate-500">#{job.id}</span>
              <span className="w-16">{job.kind}</span>
              <Badge tone={jobTone(job.status)}>{job.status}</Badge>
              <span className="truncate text-xs text-slate-500">{job.message ?? job.error ?? ""}</span>
              {job.id !== followingJobId && (
                <button
                  type="button"
                  className="ml-auto text-xs text-slate-500 hover:underline"
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

function jobTone(status: string): Tone {
  if (status === "succeeded") return "normal";
  if (status === "failed") return "defect";
  if (status === "running") return "info";
  return "neutral";
}

function Console({ jobId }: { jobId: number }) {
  const { job, lines, error } = useJob(jobId);
  return (
    <Panel title={`Job #${jobId}`}>
      <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
    </Panel>
  );
}

function Configuration({
  detail,
}: {
  detail: { config: Record<string, unknown>; preprocessing: Record<string, unknown> };
}) {
  return (
    <details className="rounded-lg border border-slate-200 px-4 py-3 dark:border-slate-700">
      <summary className="cursor-pointer text-sm font-semibold tracking-tight">
        Configuration
      </summary>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <ConfigBlock title="Method" values={detail.config} />
        <ConfigBlock title="Preprocessing" values={detail.preprocessing} />
      </div>
    </details>
  );
}

function ConfigBlock({ title, values }: { title: string; values: Record<string, unknown> }) {
  return (
    <div>
      <h3 className="text-xs font-medium tracking-wide text-slate-500 uppercase">{title}</h3>
      <dl className="mt-1 font-mono text-xs">
        {Object.entries(values).map(([key, value]) => (
          <div key={key} className="flex gap-2">
            <dt className="text-slate-500">{key}</dt>
            <dd>{JSON.stringify(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Metrics({
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
      <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">
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
      {pixelRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
            Pixel level
          </h4>
          <MetricList rows={pixelRows(metrics)} />
        </>
      )}
      {timingRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-medium tracking-wide text-slate-500 uppercase">
            Timing
          </h4>
          <MetricList rows={timingRows(metrics)} />
        </>
      )}

      {notes.length > 0 && (
        <ul className="mt-1 flex flex-col gap-1">
          {notes.map((note) => (
            <li key={note} className="text-xs text-amber-700 dark:text-amber-300">
              {note}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function MetricList({ rows }: { rows: MetricRow[] }) {
  return (
    <dl className="flex flex-col gap-1">
      {rows.map((row) => (
        <div key={row.key} className="flex items-baseline gap-2" title={row.hint}>
          <dt className="text-xs text-slate-500 dark:text-slate-400">{row.label}</dt>
          <dd className="ml-auto font-mono text-sm">
            {/* A metric that could not be computed stays visibly absent. Rendering it as
                0.000 would answer a question that should be asked. */}
            {row.value ?? <span className="text-slate-400">—</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function Results({ experimentId, subsets }: { experimentId: number; subsets: Subset[] }) {
  const [subset, setSubset] = useState<Subset>(subsets[subsets.length - 1] ?? "test");
  const results = useResults(experimentId, subset);
  const [threshold, setThreshold] = useState<number | null>(null);

  // The suggested threshold is the starting position, and it moves when the subset does —
  // but an operator's own drag must survive a re-render, so it is only adopted when the
  // slider has never been touched for this subset.
  useEffect(() => setThreshold(null), [subset]);
  const active = threshold ?? results.data?.suggested_threshold ?? 0;
  const report = useThreshold(experimentId, subset, active);

  const span = (results.data?.score_max ?? 1) - (results.data?.score_min ?? 0);
  const step = span > 0 ? span / 500 : 0.001;

  return (
    <Panel
      title="Results"
      actions={
        <select
          className="rounded border border-slate-300 bg-white px-2 py-1 text-sm dark:border-slate-600 dark:bg-slate-800"
          aria-label="Subset"
          value={subset}
          onChange={(event) => setSubset(event.target.value as Subset)}
        >
          {subsets.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      }
    >
      {results.error && <ErrorBox>{results.error.message}</ErrorBox>}
      {results.data && results.data.samples.length === 0 && (
        <Empty>Nothing scored in this subset.</Empty>
      )}

      {results.data && results.data.samples.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-3">
              <label className="text-xs font-medium tracking-wide text-slate-500 uppercase">
                Threshold
              </label>
              <input
                type="range"
                aria-label="Threshold"
                className="flex-1"
                min={results.data.score_min}
                max={results.data.score_max}
                step={step}
                value={active}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
              <span className="w-24 text-right font-mono text-sm">{active.toFixed(4)}</span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Opens at {results.data.suggested_threshold.toFixed(4)} —{" "}
              {results.data.threshold_rationale}.
            </p>
          </div>

          {report.data && (
            <>
              <Confusion report={report.data} />
              <VerdictTable experimentId={experimentId} samples={report.data.samples} />
            </>
          )}
        </div>
      )}
    </Panel>
  );
}

function Confusion({
  report,
}: {
  report: {
    confusion: {
      true_positive: number;
      false_positive: number;
      true_negative: number;
      false_negative: number;
    };
    precision: number | null;
    recall: number | null;
    f1: number | null;
  };
}) {
  const { confusion } = report;
  return (
    <div className="flex flex-wrap items-center gap-6">
      <table className="text-sm">
        <thead>
          <tr className="text-xs text-slate-500">
            <th className="px-2" />
            <th className="px-2 font-medium">called defect</th>
            <th className="px-2 font-medium">called normal</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          <tr>
            <th className="px-2 text-xs font-medium text-slate-500">is defect</th>
            <td className="px-2 text-emerald-700 dark:text-emerald-300">
              {confusion.true_positive}
            </td>
            <td className="px-2 text-red-700 dark:text-red-300">{confusion.false_negative}</td>
          </tr>
          <tr>
            <th className="px-2 text-xs font-medium text-slate-500">is normal</th>
            <td className="px-2 text-amber-700 dark:text-amber-300">{confusion.false_positive}</td>
            <td className="px-2 text-emerald-700 dark:text-emerald-300">
              {confusion.true_negative}
            </td>
          </tr>
        </tbody>
      </table>

      <dl className="flex gap-4 text-sm">
        {(
          [
            ["precision", report.precision],
            ["recall", report.recall],
            ["F1", report.f1],
          ] as const
        ).map(([label, value]) => (
          <div key={label} className="flex flex-col">
            <dt className="text-xs text-slate-500">{label}</dt>
            <dd className="font-mono">
              {value === null ? <span className="text-slate-400">—</span> : value.toFixed(3)}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function VerdictTable({
  experimentId,
  samples,
}: {
  experimentId: number;
  samples: SampleVerdict[];
}) {
  const [filter, setFilter] = useState<string>("all");
  // Already classified by the server, at the threshold currently in force. The rule "a
  // score at or above the threshold is a defect" therefore exists in exactly one place.
  const classified = samples;
  const shown = filter === "all" ? classified : classified.filter((s) => s.outcome === filter);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-1.5">
        {["all", "tp", "fp", "tn", "fn", "unlabeled"].map((key) => {
          const count =
            key === "all" ? classified.length : classified.filter((s) => s.outcome === key).length;
          return (
            <button
              key={key}
              type="button"
              disabled={count === 0 && key !== "all"}
              onClick={() => setFilter(key)}
              className={`rounded px-2 py-1 text-xs ${
                filter === key
                  ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  : "bg-slate-100 text-slate-600 disabled:opacity-40 dark:bg-slate-800 dark:text-slate-300"
              }`}
            >
              {key === "all" ? "all" : OUTCOME_LABEL[key]} ({count})
            </button>
          );
        })}
      </div>

      <ul className="max-h-96 divide-y divide-slate-200 overflow-y-auto text-sm dark:divide-slate-700">
        {shown.map((sample) => (
          <li key={sample.sample_id} className="flex items-center gap-3 py-1.5">
            <Link
              to={`/experiments/${experimentId}/samples/${sample.sample_id}`}
              className="font-mono text-xs hover:underline"
            >
              {sample.group_key}/{sample.external_id}
            </Link>
            <Badge tone={sample.label === "defect" ? "defect" : "normal"}>{sample.label}</Badge>
            <Badge tone={OUTCOME_TONE[sample.outcome] ?? "neutral"}>
              {OUTCOME_LABEL[sample.outcome] ?? sample.outcome}
            </Badge>
            {sample.notes && <span className="text-xs text-slate-500">{sample.notes}</span>}
            <span className="ml-auto font-mono text-xs">{sample.score.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
