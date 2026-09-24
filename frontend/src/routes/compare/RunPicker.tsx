/**
 * Which runs are being compared.
 *
 * On the screen as well as upstream of it, so `#/compare` is somewhere you can go and not
 * only somewhere you can be sent. The catalogue's checkboxes and an experiment's "Compare
 * with…" both hand a selection here; with only those, changing one column would mean going
 * back to another screen, and the comparison would be unreachable from its own URL. Both
 * entry points and this picker apply the one `refusalReason`.
 *
 * The dataset-and-split constraint is enforced here *and* on the server. Not redundancy:
 * the server's is the one that makes the rule true, and this one is what makes it
 * legible — a disabled row that says why beats a 422 after the fact.
 */

import { formatHeadline } from "../../api/headline";
import type { ExperimentSummary } from "../../api/client";
import { MAX_RUNS, refusalReason } from "../../api/compareState";
import { Badge, Checkbox, Empty, SkeletonRows, cn } from "@vitavision/lab-ui";
import { useSplitNames } from "../../hooks/useComparison";
import { useDatasets } from "../../hooks/useCatalog";
import { useExperiments, useModelTypes } from "../../hooks/useExperiments";

export function RunPicker({
  selected,
  onToggle,
}: {
  selected: number[];
  onToggle: (id: number) => void;
}) {
  const experiments = useExperiments();
  const datasets = useDatasets();
  const methods = useModelTypes();
  const methodTitles = new Map((methods.data?.methods ?? []).map((m) => [m.key, m.title]));
  const rows = experiments.data ?? [];
  // Before the early returns: the hook count may not depend on whether the list arrived.
  const splitNames = useSplitNames([...new Set(rows.map((row) => row.dataset_id))]);

  if (experiments.isPending) return <SkeletonRows rows={4} />;
  if (rows.length === 0) {
    return <Empty>No experiments yet. Train one first, then two of them can be compared.</Empty>;
  }

  const anchorId = selected[0];
  const anchor = rows.find((row) => row.id === anchorId);
  const names = new Map((datasets.data ?? []).map((dataset) => [dataset.id, dataset.name]));

  /* Grouped by dataset and split, because that is the boundary of what can be compared:
     every row inside one group is selectable together, and no row crosses a heading. Once
     a first run is chosen, only its group is shown — every other row would be a disabled
     line whose reason lived in a tooltip, and the reason is one sentence for all of them. */
  const groups = groupRuns(rows);
  const visible =
    anchor === undefined
      ? groups
      : groups.filter(
          (group) => group.datasetId === anchor.dataset_id && group.splitId === anchor.split_id,
        );
  const hidden = groups
    .filter((group) => !visible.includes(group))
    .reduce((count, group) => count + group.runs.length, 0);

  return (
    <div className="flex flex-col gap-4">
      {visible.map((group) => (
        <div key={`${group.datasetId}/${group.splitId}`} className="flex flex-col gap-1.5">
          <h3 className="text-xs font-semibold text-fg-muted">
            {names.get(group.datasetId) ?? `dataset ${group.datasetId}`}
            <span className="ml-2 font-mono text-[11px] text-fg-subtle">
              {splitNames.get(group.splitId) ?? `split ${group.splitId}`}
            </span>
          </h3>
          {group.runs.map((run) => (
            <RunRow
              key={run.id}
              run={run}
              checked={selected.includes(run.id)}
              refusal={refusalReason(run, anchor, selected)}
              methodTitle={methodTitles.get(run.model_type) ?? run.model_type}
              onToggle={() => onToggle(run.id)}
            />
          ))}
        </div>
      ))}

      {hidden > 0 && (
        <p className="text-xs text-fg-subtle">
          {hidden === 1 ? "1 run" : `${hidden} runs`} on other datasets or splits not shown —
          their numbers cover different samples. Clear the selection to see every run.
        </p>
      )}

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
  methodTitle,
  onToggle,
}: {
  run: ExperimentSummary;
  checked: boolean;
  refusal: string | null;
  methodTitle: string;
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
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm text-fg">{run.name}</span>
        {/* On the row, not only in a tooltip: a reason you have to hover to find is one
            most readers never see. */}
        {refusal && <span className="truncate text-[11px] text-fg-subtle">{refusal}</span>}
      </span>
      <span className="text-xs text-fg-muted">{methodTitle}</span>
      {run.status !== "trained" && <Badge tone="unlabeled">{run.status}</Badge>}
      <span className="w-24 text-right font-mono text-xs tabular-nums">
        {/* A metric that could not be computed is a dash, never a zero. */}
        {formatHeadline(run) ?? <span className="text-fg-subtle">—</span>}
      </span>
    </div>
  );
}
