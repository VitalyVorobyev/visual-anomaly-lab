/**
 * One experiment: run it, watch it, and read what it found.
 *
 * The training console is `JobProgress` and `useJob`, unchanged from the import screen —
 * putting import on the job system in M2 was precisely so that training would need no
 * second progress mechanism, and this screen is where that pays.
 *
 * M4 turned this into a set of tabs; M4.7 decided what belongs on which. Overview is what
 * the experiment *achieved* — headline numbers, results at a threshold, the metric tables,
 * and the frozen configuration behind them. The run controls are a bar above the tabs, so
 * a run can be started and watched from wherever the reader is standing. Runs, logs and
 * files are their own tab, because a job log is diagnostic material rather than a finding.
 * Training, Benchmark, Architecture and Inspector are the workbench, and **the last two are
 * driven entirely by the diagnostics index, never by `model_type`** — which is what makes
 * `efficientad_custom` inherit both in M6 (ADR-0018).
 *
 * The active tab lives in the URL rather than in component state, following
 * `api/browseState.ts`: reloading during a long run puts you back on the chart you were
 * watching, not on the front page of the experiment.
 */

import { useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";

import type { MetricSummary, Subset } from "./../api/client";
import { modelScoped, ofKinds } from "../api/diagnostics";
import type { TabId } from "../api/experimentTabs";
import { parseTab } from "../api/experimentTabs";
import type { ResultsState } from "../api/resultsState";
import { readResultsState, writeResultsState } from "../api/resultsState";
import { Badge, Empty, ErrorBox, PageHeader, ReadoutStrip, SkeletonRows, Tabs } from "@vitavision/lab-ui";
import { useJob, isTerminal } from "../hooks/useJob";
import { useDiagnostics, useExperiment } from "../hooks/useExperiments";
import { BenchmarkTab } from "./experiment/BenchmarkTab";
import { GalleryTab } from "./experiment/GalleryTab";
import { JobsFilesTab } from "./experiment/JobsFilesTab";
import { RunBar } from "./experiment/RunBar";
import { useVerdicts } from "./experiment/useVerdicts";
import { ArchitectureTab, InspectorTab, PICTURE_KINDS, STRUCTURE_KINDS } from "./experiment/DiagnosticsTabs";
import { Configuration, Headline, Metrics } from "./experiment/OverviewTab";
import { Results } from "./experiment/ResultsPanel";
import { TrainingTab } from "./experiment/TrainingTab";

/* A disabled tab used to give no reason at all, which reads as broken rather than as
   not-yet. Each says what would fill it. */
const NOT_SCORED = "Nothing has been scored yet — run 'Score & evaluate' first.";
const NO_STRUCTURE = "This method recorded no architecture or score-normalization table.";
const NO_PICTURES = "This method recorded no run-scoped pictures of itself.";

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

  /*
   * When a scoring run lands while its own screen is open, go and look at what it found.
   *
   * The point of a run is the pictures it produced, and the previous behaviour was to
   * leave the reader on a console showing three lines of `[done]` with the results one
   * undiscovered tab away. Only from the Overview tab and only once per arrival: a reader
   * who has chosen a tab has said where they want to be, and yanking them out of it would
   * be a worse bug than the one this fixes.
   */
  const watchedJob = charted.job;
  const landedOn = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!watchedJob || watchedJob.kind !== "infer" || watchedJob.status !== "succeeded") return;
    if (landedOn.current === watchedJob.id) return;
    landedOn.current = watchedJob.id;
    if (params.get("tab") !== null) return;
    const updated = new URLSearchParams(params);
    updated.set("tab", "samples");
    setParams(updated, { replace: true });
  }, [watchedJob, params, setParams]);

  const tab = parseTab(params.get("tab"));
  const selectTab = (next: TabId) => {
    const updated = new URLSearchParams(params);
    if (next === "overview") updated.delete("tab");
    else updated.set("tab", next);
    setParams(updated, { replace: true });
  };

  // The results view's own state — tab, subset, filter, layers — lives in the same query
  // string so the gallery hands it to the sample page and gets it back on the way out.
  // The tab is part of that state now, so nothing here has to re-attach it by hand.
  const results = readResultsState(params);
  const updateResults = (next: Partial<ResultsState>) => {
    setParams(writeResultsState({ ...results, ...next }), { replace: true });
  };
  const verdicts = useVerdicts(experimentId, results);

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
              { label: "input", value: detail.region_profile_name },
              { label: "method", value: detail.model_type },
            ]}
          />
        }
      />

      {/* Above the tabs, so a run can be started from anywhere and its progress is visible
          without leaving whatever is being read. */}
      <RunBar
        experimentId={experimentId}
        detail={detail}
        jobs={detail.jobs}
        hasTrained={hasTrained}
        onFollow={setFollowingJobId}
        onViewLog={() => selectTab("training")}
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
          {
            id: "samples",
            label: "Samples",
            disabled: !hasScores,
            title: hasScores ? undefined : NOT_SCORED,
          },
          {
            id: "training",
            label: "Training",
            disabled: !hasTrained,
            title: hasTrained ? undefined : "Nothing has been trained yet.",
          },
          {
            id: "benchmark",
            label: "Benchmark",
            disabled: !hasScores,
            title: hasScores ? undefined : NOT_SCORED,
          },
          {
            id: "architecture",
            label: "Architecture",
            disabled: !hasStructure,
            title: hasStructure ? undefined : NO_STRUCTURE,
          },
          {
            id: "inspector",
            label: "Inspector",
            disabled: !hasPictures,
            title: hasPictures ? undefined : NO_PICTURES,
          },
          { id: "jobs", label: "Jobs & files" },
        ]}
      />

      {/* Achievement first, then the record of how it was obtained. The configuration is
          what makes a number reproducible, which is a reason to keep it on this screen and
          a reason for it not to be the first thing on it. */}
      {tab === "overview" && (
        <>
          {hasScores ? (
            <OverviewResults
              experimentId={experimentId}
              subsets={detail.scored_subsets}
              metrics={detail.metrics}
            />
          ) : (
            <Empty>
              Nothing has been scored yet. Train, then run &lsquo;Score &amp; evaluate&rsquo;
              from the bar above.
            </Empty>
          )}

          {detail.metrics.length > 0 && (
            <Metrics
              experimentId={experimentId}
              metrics={detail.metrics}
              aggregation={String(detail.evaluation.aggregation ?? "max")}
            />
          )}

          <Configuration detail={detail} />
        </>
      )}

      {tab === "samples" && (
        <GalleryTab
          experimentId={experimentId}
          state={results}
          onChange={updateResults}
          verdicts={verdicts}
          range={detail.map_range}
        />
      )}

      {tab === "training" && (
        <TrainingTab
          jobs={detail.jobs}
          series={charted.series}
          followingJobId={followingJobId}
          console={{
            jobId: chartedJobId,
            job: charted.job,
            lines: charted.lines,
            error: charted.error,
          }}
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

      {tab === "jobs" && (
        <JobsFilesTab
          experimentId={experimentId}
          jobs={detail.jobs}
          followingJobId={followingJobId}
          onFollow={setFollowingJobId}
        />
      )}
    </div>
  );
}

/**
 * The headline numbers and the results panel, sharing one subset.
 *
 * They must: a headline reading `test` above a results table showing `val` is two answers
 * to one question. The distribution charts stay off — those are the Benchmark tab.
 */
function OverviewResults({
  experimentId,
  subsets,
  metrics,
}: {
  experimentId: number;
  subsets: Subset[];
  metrics: MetricSummary[];
}) {
  const [subset, setSubset] = useState<Subset>(subsets[subsets.length - 1] ?? "test");
  return (
    <>
      <Headline metrics={metrics} subset={subset} />
      <Results
        experimentId={experimentId}
        subsets={subsets}
        subset={subset}
        onSubset={setSubset}
      />
    </>
  );
}
