/**
 * What this experiment achieved.
 *
 * Overview used to open on a run list and a job log, with the metrics and the results
 * below the fold — the screen that should answer "did this work" answered "what did the
 * last process print". The runs, the logs and the artifact listing are now a tab of their
 * own; the run controls are in the bar above the tabs. What is left here is the headline
 * numbers, the results at a threshold, the full metric tables, and the frozen configuration
 * that makes all of it reproducible — in that order, because the reader wants the finding
 * before the record of how it was obtained.
 */

import type { MetricValue } from "../../api/metrics";
import { caveats, detectionRows, groupingNote, pixelRows, timingRows } from "../../api/metrics";
import type { MetricRow } from "../../api/metrics";
import type { MetricSummary, Subset, TrainingState } from "../../api/client";
import { Button, Callout, CountRun, Disclosure, Panel, type Tone } from "@vitavision/lab-ui";
import { useReevaluate } from "../../hooks/useExperiments";

export function jobTone(status: string): Tone {
  if (status === "succeeded") return "normal";
  if (status === "failed") return "defect";
  if (status === "running") return "info";
  return "neutral";
}

/**
 * The four numbers someone opens this screen for, above everything else.
 *
 * Deliberately a *subset* of the metric tables below rather than a different computation —
 * the same values, promoted. Absent ones render as a dash and keep their slot, because a
 * missing pixel ROC-AUC (no masks in this dataset) is a fact about the run and hiding the
 * row would make two experiments' headlines silently different shapes.
 */
const HEADLINE: { key: string; label: string }[] = [
  { key: "sample_roc_auc", label: "sample ROC-AUC" },
  { key: "sample_average_precision", label: "sample AP" },
  { key: "pixel_roc_auc", label: "pixel ROC-AUC" },
  { key: "pixel_au_pro", label: "AU-PRO" },
];

export function Headline({
  metrics,
  subset,
}: {
  metrics: MetricSummary[];
  subset: Subset | undefined;
}) {
  const entry = metrics.find((row) => row.subset === subset) ?? metrics[metrics.length - 1];
  if (!entry) return null;
  const values = (entry.metrics ?? {}) as Record<string, unknown>;

  return (
    <div className="flex flex-wrap gap-x-10 gap-y-4 rounded-lg border border-line bg-surface px-4 py-3">
      {HEADLINE.map(({ key, label }) => {
        const value = values[key];
        return (
          <div key={key} className="flex flex-col gap-0.5">
            <span className="text-xs text-fg-muted">{label}</span>
            <span className="font-mono text-xl tabular-nums">
              {typeof value === "number" ? (
                value.toFixed(4)
              ) : (
                <span className="text-fg-subtle">—</span>
              )}
            </span>
          </div>
        );
      })}
      <div className="flex flex-col gap-0.5">
        <span className="text-xs text-fg-muted">subset</span>
        <span className="font-mono text-xl">{entry.subset}</span>
        {entry.ground_truth_stale && (
          <span className="text-xs text-warn">truth changed</span>
        )}
      </div>
    </div>
  );
}

export function Configuration({
  detail,
}: {
  detail: {
    config: Record<string, unknown>;
    preprocessing: Record<string, unknown>;
    region_profile_id: number;
    region_profile_name?: string | null;
    region_manifest_sha256: string;
    training_state?: TrainingState | null;
  };
}) {
  const state = detail.training_state ?? null;
  return (
    <Panel title="Configuration">
      {/* Frozen into the experiment at creation, so this is what the run *used* rather than
          what the form currently defaults to. Worth reading before comparing two runs. */}

      {/* Above the fold of the disclosure, because without it the block below shows
          `max_steps: 4000` beside a model that has trained for 8000 — which is a lie in
          the one record that is supposed to make a run reproducible (handbook jobs.md). */}
      {state !== null && (
        <p className="mb-3 text-xs text-fg-muted">
          <span className="font-mono text-fg">{state.completed_steps}</span> steps completed
          over {state.runs} run{state.runs === 1 ? "" : "s"}
          {state.runs > 1 && `, the last of them ${state.last_run_steps}`}.{" "}
          <span className="font-mono">max_steps</span> below is this method&rsquo;s{" "}
          <em>per-run</em> budget, not the total.
        </p>
      )}

      <Disclosure summary="What this run was configured with">
        <div className="grid gap-5 sm:grid-cols-3">
          <ConfigBlock
            title="Prepared region"
            values={{
              profile: detail.region_profile_name ?? detail.region_profile_id,
              revision_id: detail.region_profile_id,
              manifest: detail.region_manifest_sha256,
            }}
          />
          <ConfigBlock title="Method" values={detail.config} />
          <ConfigBlock title="Model input" values={detail.preprocessing} />
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
              <dd className="break-all text-right text-fg tabular-nums">{JSON.stringify(value)}</dd>
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
  metrics: MetricSummary[];
  aggregation: string;
}) {
  const reevaluate = useReevaluate(experimentId);
  const stale = metrics.some((entry) => entry.ground_truth_stale);

  return (
    <Panel
      title="Metrics"
      actions={
        <Button disabled={reevaluate.isPending} onClick={() => reevaluate.mutate()}>
          Recompute
        </Button>
      }
    >
      {stale && (
        <Callout className="mb-4" tone="warning" title="Ground truth changed">
          These metrics describe an older set of labels or masks. Recompute them before
          using the numbers for a decision or comparison.
        </Callout>
      )}
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
