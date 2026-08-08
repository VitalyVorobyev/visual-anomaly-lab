/**
 * Which runs are being compared.
 *
 * On the screen rather than upstream of it, so `#/compare` is somewhere you can go and not
 * only somewhere you can be sent. The alternative — checkboxes on the experiment list and a
 * button that navigates here — makes the comparison unreachable from its own URL and puts
 * the selection out of reach the moment you want to change one column.
 *
 * The dataset-and-split constraint is enforced here *and* on the server. Not redundancy:
 * the server's is the one that makes the rule true, and this one is what makes it
 * legible — a disabled row that says why beats a 422 after the fact.
 */

import type { ExperimentSummary } from "../../api/client";
import { MAX_RUNS, refusalReason } from "../../api/compareState";
import { Badge, Checkbox, Empty, SkeletonRows, cn } from "../../components/ui";
import { useDatasets } from "../../hooks/useCatalog";
import { useExperiments } from "../../hooks/useExperiments";

export function RunPicker({
  selected,
  onToggle,
}: {
  selected: number[];
  onToggle: (id: number) => void;
}) {
  const experiments = useExperiments();
  const datasets = useDatasets();

  if (experiments.isPending) return <SkeletonRows rows={4} />;
  const rows = experiments.data ?? [];
  if (rows.length === 0) {
    return <Empty>No experiments yet. Train one first, then two of them can be compared.</Empty>;
  }

  const anchorId = selected[0];
  const anchor = rows.find((row) => row.id === anchorId);
  const names = new Map((datasets.data ?? []).map((dataset) => [dataset.id, dataset.name]));

  /* Grouped by dataset and split, because that is the boundary of what can be compared:
     every row inside one group is selectable together, and no row crosses a heading. */
  const groups = groupRuns(rows);

  return (
    <div className="flex flex-col gap-4">
      {groups.map((group) => (
        <div key={`${group.datasetId}/${group.splitId}`} className="flex flex-col gap-1.5">
          <h3 className="text-xs font-semibold text-fg-muted">
            {names.get(group.datasetId) ?? `dataset ${group.datasetId}`}
            <span className="ml-2 font-mono text-[11px] text-fg-subtle">
              split {group.splitId}
            </span>
          </h3>
          {group.runs.map((run) => (
            <RunRow
              key={run.id}
              run={run}
              checked={selected.includes(run.id)}
              refusal={refusalReason(run, anchor, selected)}
              onToggle={() => onToggle(run.id)}
            />
          ))}
        </div>
      ))}

      <p className="text-xs text-fg-muted">
        {selected.length < 2
          ? "Pick at least two runs of the same split."
          : `${selected.length} of ${MAX_RUNS} selected.`}
      </p>
    </div>
  );
}

interface RunGroup {
  datasetId: number;
  splitId: number;
  runs: ExperimentSummary[];
}

function groupRuns(rows: ExperimentSummary[]): RunGroup[] {
  const groups = new Map<string, RunGroup>();
  for (const run of rows) {
    const key = `${run.dataset_id}/${run.split_id}`;
    const found = groups.get(key);
    if (found) found.runs.push(run);
    else groups.set(key, { datasetId: run.dataset_id, splitId: run.split_id, runs: [run] });
  }
  return [...groups.values()];
}

function RunRow({
  run,
  checked,
  refusal,
  onToggle,
}: {
  run: ExperimentSummary;
  checked: boolean;
  refusal: string | null;
  onToggle: () => void;
}) {
  const disabled = refusal !== null;
  return (
    <div
      title={refusal ?? undefined}
      className={cn(
        "flex items-center gap-3 rounded-control border px-3 py-1.5 transition-colors",
        checked ? "border-signal bg-signal/5" : "border-line bg-raised/40",
        disabled ? "opacity-45" : "hover:border-line-strong",
      )}
    >
      <Checkbox
        checked={checked}
        disabled={disabled}
        onCheckedChange={onToggle}
        aria-label={run.name}
      />
      <span className="min-w-0 flex-1 truncate text-sm text-fg">{run.name}</span>
      <span className="font-mono text-[11px] text-fg-subtle">{run.model_type}</span>
      {run.status !== "trained" && <Badge tone="unlabeled">{run.status}</Badge>}
      <span className="w-14 text-right font-mono text-xs tabular-nums">
        {/* A metric that could not be computed is a dash, never a zero. */}
        {run.headline_roc_auc === null || run.headline_roc_auc === undefined ? (
          <span className="text-fg-subtle">—</span>
        ) : (
          run.headline_roc_auc.toFixed(3)
        )}
      </span>
    </div>
  );
}
