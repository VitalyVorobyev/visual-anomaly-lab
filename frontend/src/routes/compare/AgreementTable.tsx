/**
 * Every sample, as every run judged it — and the filter that makes it useful.
 *
 * This is the thing a comparison screen can do that two results screens side by side
 * cannot: *where do these methods disagree*. One row per sample, one verdict per run, at
 * each run's own threshold, filtered to the rows where the verdicts differ.
 *
 * Nothing here reapplies a threshold. The server has already tagged every cell, at N
 * thresholds it resolved under one rule, precisely so that "at or above the threshold is a
 * defect" exists once and in one language (ADR-0028).
 *
 * A disagreement on a *labelled* sample means exactly one run is wrong, which is the
 * strongest single signal on the screen. On an unlabeled one it means the two methods have
 * found something to argue about, which is the best reason to go and look at the picture —
 * so the row links to the side-by-side view rather than merely reporting.
 */

import { Link } from "react-router";

import type { ComparedRun, ComparedSample } from "../../api/client";
import type { CompareState } from "../../api/compareState";
import { writeCompareState } from "../../api/compareState";
import type { Outcome } from "../../api/resultsState";
import { OUTCOMES } from "../../api/resultsState";
import { Badge, Empty, Panel, Switch, Tabs } from "@vitavision/lab-ui";
import { OUTCOME_LABEL, OUTCOME_TONE } from "../experiment/ResultsPanel";

export function AgreementTable({
  runs,
  samples,
  state,
  onChange,
}: {
  runs: ComparedRun[];
  samples: ComparedSample[];
  state: CompareState;
  onChange: (next: Partial<CompareState>) => void;
}) {
  const disagreements = samples.filter((row) => !row.agree).length;
  const filtered = samples.filter((row) => {
    if (state.disagreeOnly && row.agree) return false;
    if (state.outcome !== undefined && !row.outcomes.includes(state.outcome)) return false;
    return true;
  });

  return (
    <Panel
      title="Per sample"
      actions={
        <Switch
          checked={state.disagreeOnly}
          onCheckedChange={(disagreeOnly) => onChange({ disagreeOnly })}
          label="Only disagreements"
        />
      }
    >
      <div className="mb-3 flex flex-col gap-2">
        <p className="text-xs text-fg-muted">
          {/* Stated as a fraction, because the number alone means nothing without the
              denominator: 12 disagreements is a lot out of 40 and nothing out of 4000. */}
          These runs disagree on{" "}
          <span className="font-mono text-fg">{disagreements}</span> of{" "}
          <span className="font-mono">{samples.length}</span> samples, each at its own
          threshold. A disagreement on a labelled sample means exactly one of them is wrong.
        </p>

        <Tabs
          label="Outcome filter"
          active={state.outcome ?? "all"}
          onSelect={(id) => onChange({ outcome: id === "all" ? undefined : (id as Outcome) })}
          items={[
            { id: "all", label: "every outcome" },
            ...OUTCOMES.map((outcome) => ({
              id: outcome,
              label: OUTCOME_LABEL[outcome] ?? outcome,
              /* Any run, not every run — an outcome belongs to a verdict, and the point of
                 the table is that the verdicts differ. */
              count: samples.filter((row) => row.outcomes.includes(outcome)).length,
            })),
          ]}
        />
      </div>

      {filtered.length === 0 ? (
        <Empty>
          {state.disagreeOnly
            ? "Every sample got the same verdict from every run, at each one's own operating point."
            : "No sample matches this filter."}
        </Empty>
      ) : (
        <div className="w-full overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-line">
                <th className="py-2 pr-3 text-xs font-medium text-fg-muted">Sample</th>
                <th className="px-2 py-2 text-xs font-medium text-fg-muted">Label</th>
                {runs.map((run) => (
                  <th key={run.id} className="min-w-36 px-2 py-2">
                    <span className="block truncate text-xs font-semibold text-fg">
                      {run.name}
                    </span>
                  </th>
                ))}
                <th className="w-14" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <Row key={row.sample_id} row={row} runs={runs} state={state} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function Row({
  row,
  runs,
  state,
}: {
  row: ComparedSample;
  runs: ComparedRun[];
  state: CompareState;
}) {
  return (
    <tr className={row.agree ? "border-b border-line/60" : "border-b border-warn/30 bg-warn/5"}>
      <th className="py-1.5 pr-3 text-left font-mono text-xs font-normal text-fg">
        {row.external_id}
        {row.group_key !== row.external_id && (
          <span className="ml-1.5 text-fg-subtle">{row.group_key}</span>
        )}
      </th>
      <td className="px-2 py-1.5">
        <Badge tone={row.label === "defect" ? "defect" : row.label === "normal" ? "normal" : "unlabeled"}>
          {row.label}
        </Badge>
      </td>
      {runs.map((run, index) => (
        <td key={run.id} className="px-2 py-1.5">
          <div className="flex items-baseline gap-2">
            {row.outcomes[index] ? (
              <Badge tone={OUTCOME_TONE[row.outcomes[index] as string] ?? "neutral"}>
                {row.outcomes[index]}
              </Badge>
            ) : (
              <span className="text-xs text-fg-subtle">—</span>
            )}
            <span className="font-mono text-[11px] tabular-nums text-fg-muted">
              {/* Each run's score in its own units — printed for the reader's judgement,
                  never compared with the cell beside it. */}
              {row.scores[index] === null || row.scores[index] === undefined
                ? "—"
                : (row.scores[index] as number).toFixed(3)}
            </span>
          </div>
        </td>
      ))}
      <td className="py-1.5 pl-2 text-right">
        <Link
          to={`/compare/samples/${row.sample_id}?${writeCompareState(state).toString()}`}
          className="text-xs text-signal underline-offset-2 hover:underline"
        >
          open
        </Link>
      </td>
    </tr>
  );
}
