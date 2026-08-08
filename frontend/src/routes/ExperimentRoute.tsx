/**
 * One experiment: run it, watch it, and read what it found.
 *
 * The training console is `JobProgress` and `useJob`, unchanged from the import screen —
 * putting import on the job system in M2 was precisely so that training would need no
 * second progress mechanism, and this screen is where that pays.
 *
 * M4 turned this into a set of tabs. The Overview is M3's screen intact; Training,
 * Benchmark, Architecture and Inspector are the workbench. **The last three are driven
 * entirely by the diagnostics index and the evaluation layer, never by `model_type`** —
 * which is what makes `efficientad_custom` inherit all of them in M6 (ADR-0018).
 *
 * The active tab lives in the URL rather than in component state, following
 * `api/browseState.ts`: reloading during a long run puts you back on the chart you were
 * watching, not on the front page of the experiment.
 */

import { useEffect, useState } from "react";
import { useParams, useSearchParams } from "react-router";

import type { Subset } from "./../api/client";
import { modelScoped, ofKinds } from "../api/diagnostics";
import { Tabs } from "../components/Tabs";
import { Badge, Empty, ErrorBox, PageHeader, ReadoutStrip, SkeletonRows } from "../components/ui";
import { useJob, isTerminal } from "../hooks/useJob";
import { useDiagnostics, useExperiment } from "../hooks/useExperiments";
import { BenchmarkTab } from "./experiment/BenchmarkTab";
import { ArchitectureTab, InspectorTab, PICTURE_KINDS, STRUCTURE_KINDS } from "./experiment/DiagnosticsTabs";
import { Configuration, Console, Metrics, Runs } from "./experiment/OverviewTab";
import { Results } from "./experiment/ResultsPanel";
import { TrainingTab } from "./experiment/TrainingTab";

const TAB_IDS = ["overview", "training", "benchmark", "architecture", "inspector"] as const;
type TabId = (typeof TAB_IDS)[number];

function parseTab(raw: string | null): TabId {
  return TAB_IDS.includes(raw as TabId) ? (raw as TabId) : "overview";
}

export function ExperimentRoute() {
  const { experimentId: raw } = useParams();
  const experimentId = raw === undefined ? undefined : Number(raw);
  const experiment = useExperiment(experimentId);
  const diagnostics = useDiagnostics(experimentId);
  const [params, setParams] = useSearchParams();
  const [followingJobId, setFollowingJobId] = useState<number | undefined>();

  // A job that is still running when the screen opens should be followed without anyone
  // having to press anything — reload during training and the console picks up again.
  useEffect(() => {
    if (followingJobId !== undefined || !experiment.data) return;
    const live = experiment.data.jobs.find((job) => !isTerminal(job.status));
    if (live) setFollowingJobId(live.id);
  }, [experiment.data, followingJobId]);

  // The charts want the most recent training run whether or not anyone is following it,
  // so that opening the tab after a run finished shows the run rather than nothing.
  const lastTrainJob = experiment.data?.jobs.filter((job) => job.kind === "train").at(0);
  const chartedJobId = followingJobId ?? lastTrainJob?.id;
  const charted = useJob(chartedJobId, { experimentId });

  const tab = parseTab(params.get("tab"));
  const selectTab = (next: TabId) => {
    const updated = new URLSearchParams(params);
    if (next === "overview") updated.delete("tab");
    else updated.set("tab", next);
    setParams(updated, { replace: true });
  };

  if (experiment.isPending) return <SkeletonRows rows={5} />;
  if (experiment.error) return <ErrorBox>{experiment.error.message}</ErrorBox>;
  if (!experiment.data || experimentId === undefined) return <Empty>No such experiment.</Empty>;

  const detail = experiment.data;
  const runScoped = modelScoped(diagnostics.data);
  const hasStructure = ofKinds(runScoped, STRUCTURE_KINDS).length > 0;
  const hasPictures = ofKinds(runScoped, PICTURE_KINDS).length > 0;
  const hasScores = detail.scored_subsets.length > 0;
  const hasTrained = detail.jobs.some((job) => job.kind === "train");

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        back={{ to: "/experiments", label: "All experiments" }}
        title={detail.name}
        actions={
          <Badge tone={detail.status === "trained" ? "normal" : "unlabeled"}>{detail.status}</Badge>
        }
        meta={
          /* The readout: what this run is, in one line, in the same slot on every screen
             that has a run. The app has a real hierarchy and no breadcrumbs, so without
             this the only way to tell which dataset you are inside is to recognise the
             name in the heading. */
          <ReadoutStrip
            items={[
              { label: "dataset", value: detail.dataset_name },
              { label: "split", value: detail.split_name },
              { label: "method", value: detail.model_type },
            ]}
          />
        }
      />

      {/* A tab appears when there is something in it. Gated on what the run recorded, never
          on which method recorded it — a method that emits no diagnostics simply has fewer
          tabs, with no branch on its name anywhere. */}
      <Tabs
        label="Experiment views"
        active={tab}
        onSelect={selectTab}
        items={[
          { id: "overview", label: "Overview" },
          { id: "training", label: "Training", disabled: !hasTrained },
          { id: "benchmark", label: "Benchmark", disabled: !hasScores },
          { id: "architecture", label: "Architecture", disabled: !hasStructure },
          { id: "inspector", label: "Inspector", disabled: !hasPictures },
        ]}
      />

      {tab === "overview" && (
        <>
          <Runs
            experimentId={experimentId}
            jobs={detail.jobs}
            onFollow={setFollowingJobId}
            followingJobId={followingJobId}
          />

          {followingJobId !== undefined && (
            <Console jobId={followingJobId} experimentId={experimentId} />
          )}

          <Configuration detail={detail} />

          {detail.metrics.length > 0 && (
            <Metrics
              experimentId={experimentId}
              metrics={detail.metrics}
              aggregation={String(detail.evaluation.aggregation ?? "max")}
            />
          )}

          {hasScores && <OverviewResults experimentId={experimentId} subsets={detail.scored_subsets} />}
        </>
      )}

      {tab === "training" && (
        <TrainingTab
          jobs={detail.jobs}
          series={charted.series}
          followingJobId={followingJobId}
        />
      )}

      {tab === "benchmark" && (
        <BenchmarkTab
          experimentId={experimentId}
          subsets={detail.scored_subsets}
          metrics={detail.metrics}
        />
      )}

      {tab === "architecture" && (
        <ArchitectureTab experimentId={experimentId} index={diagnostics.data} />
      )}

      {tab === "inspector" && (
        <InspectorTab experimentId={experimentId} index={diagnostics.data} />
      )}
    </div>
  );
}

/** The results panel without the distribution charts — those live on Benchmark. */
function OverviewResults({
  experimentId,
  subsets,
}: {
  experimentId: number;
  subsets: Subset[];
}) {
  const [subset, setSubset] = useState<Subset>(subsets[subsets.length - 1] ?? "test");
  return (
    <Results
      experimentId={experimentId}
      subsets={subsets}
      subset={subset}
      onSubset={setSubset}
    />
  );
}
