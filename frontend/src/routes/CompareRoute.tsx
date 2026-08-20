/**
 * Several methods, one split, one evaluation protocol — the screen the workbench exists for.
 *
 * The design decision it rests on is ADR-0028, and it is worth restating here because every
 * layout choice below follows from it: **a score has no meaning outside its own run.** So
 * the threshold-independent metrics are one table, everything threshold-dependent is
 * another, the second one carries each run's own cut and the sentence that produced it, and
 * nowhere is a number from one run compared with a number from another in score units.
 *
 * N-way rather than two-way at every layer, including the ones where two would have been
 * simpler. M6 and M7 each add a method and neither may cost a line here — the same claim
 * ADR-0007 makes about the method picker, applied to the screen that reads the results.
 *
 * The selection lives on the screen rather than upstream of it, so `#/compare` is somewhere
 * you can go rather than only somewhere you can be sent, and changing one column does not
 * mean going back to another screen.
 */

import { useSearchParams } from "react-router";

import type { Subset } from "../api/client";
import type { CompareState, CompareView } from "../api/compareState";
import { readCompareState, toggleRun, writeCompareState } from "../api/compareState";
import { Callout, Disclosure, Empty, ErrorBox, NumberInput, PageHeader, Panel, ReadoutStrip, SegmentedControl, Select, SkeletonRows, Tabs } from "@vitavision/lab-ui";
import { useComparison } from "../hooks/useComparison";
import { AgreementTable } from "./compare/AgreementTable";
import { CompareCurves } from "./compare/CompareCurves";
import { ConfigDiff } from "./compare/ConfigDiff";
import { MetricTable, OperatingTable } from "./compare/MetricTable";
import { RunPicker } from "./compare/RunPicker";

export function CompareRoute() {
  const [params, setParams] = useSearchParams();
  const state = readCompareState(params);
  const update = (next: Partial<CompareState>) => {
    setParams(writeCompareState({ ...state, ...next }), { replace: true });
  };

  const comparison = useComparison({
    ids: state.ids,
    subset: state.subset,
    at: state.at,
    recallTarget: state.recallTarget,
  });
  const report = comparison.data;

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Compare"
        meta={
          report && (
            <ReadoutStrip
              items={[
                { label: "dataset", value: report.dataset_name },
                { label: "split", value: report.split_name },
                { label: "subset", value: report.subset },
              ]}
            />
          )
        }
      />

      <Panel title="Runs">
        {/* Folded once a comparison is on screen: the picker is how you arrive and the
            table is what you came for, and leaving a list of every experiment above the
            numbers pushes the finding below the fold on the screen whose whole job is to
            show it. */}
        <Disclosure
          summary={state.ids.length >= 2 ? "Change which runs are compared" : "Choose runs"}
          count={state.ids.length}
          defaultOpen={state.ids.length < 2}
        >
          <RunPicker selected={state.ids} onToggle={(id) => update({ ids: toggleRun(state.ids, id) })} />
        </Disclosure>
      </Panel>

      {state.ids.length < 2 && (
        <Empty>
          Pick at least two runs of the same dataset and split. Runs on different data are
          not comparable, so they cannot be selected together.
        </Empty>
      )}

      {comparison.isPending && state.ids.length >= 2 && <SkeletonRows rows={6} />}
      {comparison.error && <ErrorBox>{comparison.error.message}</ErrorBox>}

      {report && (
        <>
          <Protocol
            state={state}
            subsets={report.subsets}
            onChange={update}
          />

          {/* Above the numbers, never below them. Every one of these changes what the table
              beneath it means, and a caveat under a metric is a caveat nobody read. */}
          {report.warnings.length > 0 && (
            <Callout tone="warning" title="Read these numbers with care">
              <ul className="flex list-disc flex-col gap-1 pl-4">
                {report.warnings.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </Callout>
          )}

          <Tabs
            label="Comparison views"
            active={state.view}
            onSelect={(view: CompareView) => update({ view })}
            items={[
              { id: "metrics", label: "Metrics" },
              { id: "curves", label: "Curves" },
              { id: "config", label: "Configuration" },
              { id: "samples", label: "Per sample", count: report.samples.length },
            ]}
          />

          {state.view === "metrics" && (
            <>
              <MetricTable runs={report.runs} />
              <OperatingTable
                runs={report.runs}
                operatingPoint={report.operating_point}
                recallTarget={report.recall_target}
              />
            </>
          )}

          {state.view === "curves" && (
            <CompareCurves runs={report.runs} subset={report.subset ?? undefined} />
          )}

          {state.view === "config" && <ConfigDiff runs={report.runs} />}

          {state.view === "samples" && (
            <AgreementTable
              runs={report.runs}
              samples={report.samples}
              state={state}
              onChange={update}
            />
          )}
        </>
      )}
    </div>
  );
}

/**
 * The two choices that decide every number below: which subset, and which operating point.
 *
 * Given a bar of its own rather than tucked into the metric panel, because the operating
 * point is not a display preference. It resolves a different threshold in every column, and
 * a control that changes N confusion matrices belongs where it can be seen doing it.
 */
function Protocol({
  state,
  subsets,
  onChange,
}: {
  state: CompareState;
  subsets: Subset[];
  onChange: (next: Partial<CompareState>) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-panel border border-line bg-surface px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-fg-muted">Subset</span>
        <Select
          className="w-32"
          aria-label="Subset"
          value={state.subset ?? ""}
          placeholder="Automatic"
          options={subsets.map((name) => ({ value: name, label: name }))}
          onValueChange={(value) =>
            onChange({ subset: value === "" ? undefined : (value as Subset) })
          }
        />
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-fg-muted">Operating point</span>
        <SegmentedControl
          aria-label="Operating point"
          value={state.at}
          onValueChange={(value) => onChange({ at: value as CompareState["at"] })}
          options={[
            { value: "f1", label: "each run's F1 optimum" },
            { value: "recall", label: "a shared recall" },
          ]}
        />
      </div>

      {/* Only while it can do something — a recall target with the F1 rule selected is a
          control for an invisible effect. */}
      {state.at === "recall" && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-fg-muted">Target recall</span>
          <NumberInput
            className="w-24"
            aria-label="Target recall"
            min={0.01}
            max={1}
            step={0.01}
            value={state.recallTarget}
            onChange={(event) => {
              const value = Number(event.target.value);
              // A half-typed number must not become a request the server refuses; the
              // control keeps the last valid target until a whole one is entered.
              if (Number.isFinite(value) && value > 0 && value <= 1) {
                onChange({ recallTarget: value });
              }
            }}
          />
        </div>
      )}
    </div>
  );
}
