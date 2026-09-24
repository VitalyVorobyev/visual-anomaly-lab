/**
 * What a run's result screens say, by task (ADR-0039).
 *
 * The shell is shared — run bar, tab set, subset, `ResultsState` in the URL, the gallery's
 * grid and the one `SampleStage` — and so is the data flow: `useVerdicts` hands every screen
 * the same classified rows. What differs by task is *what* those rows and metrics mean, and
 * that is all this registry holds: the gallery's outcome filters and the words around them,
 * and the bodies of Overview and Benchmark. An anomaly run reads a threshold and a
 * confusion matrix; a few-shot segmentation run reads overlap against its class and what
 * happened on each image (ADR-0040); a supervised segmentation run reads the summary of its
 * per-class confusion matrix (ADR-0039). No body is reached except through here.
 */

import type { ReactNode } from "react";

import type { MetricSummary, SampleVerdict, Subset, Task } from "../../api/client";
import type { MetricValue } from "../../api/metrics";
import { segmentationRows, semanticRows, timingRows } from "../../api/metrics";
import type { Outcome, ResultsState } from "../../api/resultsState";
import { MISTAKE_OUTCOMES } from "../../api/resultsState";
import {
  CountRun,
  DEFECT_COLOUR,
  Empty,
  ErrorBox,
  NORMAL_COLOUR,
  Panel,
  SkeletonRows,
  StackedBars,
} from "@vitavision/lab-ui";
import { BenchmarkTab } from "./BenchmarkTab";
import { Headline, MetricList, Metrics } from "./OverviewTab";
import { OUTCOME_TONE, Results } from "./ResultsPanel";
import type { Verdicts } from "./useVerdicts";

export interface OutcomeFilter {
  id: string;
  label: string;
  /** `undefined` is every sample. */
  outcomes: readonly Outcome[] | undefined;
}

export interface ResultsBodyProps {
  experimentId: number;
  subsets: Subset[];
  metrics: MetricSummary[];
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  verdicts: Verdicts;
  aggregation: string;
  targetLabel: string | null;
}

export interface TaskView {
  /** The gallery's outcome strip, in order. */
  filters: OutcomeFilter[];
  /** What ranking by score means, highest first and lowest first. */
  rank: { desc: string; asc: string };
  /** The line above the gallery that says what the outcome badges are measured against. */
  outcomeNote: (verdicts: Verdicts, targetLabel: string | null) => ReactNode;
  /** Overview once something is scored. */
  Scored: (props: ResultsBodyProps) => ReactNode;
  /** The metric tables, once there are metric sets. */
  MetricTables: (props: ResultsBodyProps) => ReactNode;
  Benchmark: (props: ResultsBodyProps) => ReactNode;
}

const ANOMALY: TaskView = {
  filters: [
    { id: "all", label: "all", outcomes: undefined },
    { id: "mistakes", label: "mistakes", outcomes: ["fp", "fn"] },
    { id: "tp", label: "true positive", outcomes: ["tp"] },
    { id: "fn", label: "false negative", outcomes: ["fn"] },
    { id: "fp", label: "false positive", outcomes: ["fp"] },
    { id: "tn", label: "true negative", outcomes: ["tn"] },
    { id: "unlabeled", label: "unlabeled", outcomes: ["unlabeled"] },
  ],
  rank: { desc: "most anomalous", asc: "least" },
  // The badges are a verdict *at a threshold*, and the threshold is set on Overview — so say
  // which one, or a TP here reads as a fact about the sample.
  outcomeNote: (verdicts) => (
    <>
      Outcomes at threshold{" "}
      <span className="font-mono text-fg">{verdicts.threshold.toFixed(4)}</span>
      {verdicts.rationale === undefined
        ? " — chosen on Overview."
        : ` — suggested: ${verdicts.rationale}.`}
    </>
  ),
  Scored: ({ experimentId, subsets, metrics, state, onChange }) => (
    <>
      <Headline metrics={metrics} subset={state.subset} />
      <Results
        experimentId={experimentId}
        subsets={subsets}
        state={state}
        onChange={onChange}
        metrics={metrics}
      />
    </>
  ),
  MetricTables: ({ experimentId, metrics, aggregation }) => (
    <Metrics experimentId={experimentId} metrics={metrics} aggregation={aggregation} />
  ),
  Benchmark: ({ experimentId, subsets, metrics, state, onChange }) => (
    <BenchmarkTab
      experimentId={experimentId}
      subsets={subsets}
      metrics={metrics}
      state={state}
      onChange={onChange}
    />
  ),
};

const SEGMENTATION_OUTCOMES: Outcome[] = [
  "hit",
  "low_iou",
  "miss",
  "false_presence",
  "correct_absence",
  "unlabeled",
];

const SEGMENTATION_HEADLINE = [
  { key: "foreground_iou", label: "foreground IoU" },
  { key: "foreground_dice", label: "Dice" },
  { key: "pixel_average_precision", label: "pixel AP" },
  { key: "image_presence_roc_auc", label: "presence ROC-AUC" },
  { key: "image_absent_false_positive_rate", label: "flagged when absent" },
];

const FEW_SHOT: TaskView = {
  filters: [
    { id: "all", label: "all", outcomes: undefined },
    {
      id: "mistakes",
      label: "mistakes",
      outcomes: MISTAKE_OUTCOMES.filter((outcome) => SEGMENTATION_OUTCOMES.includes(outcome)),
    },
    { id: "hit", label: "hit", outcomes: ["hit"] },
    { id: "low_iou", label: "low IoU", outcomes: ["low_iou"] },
    { id: "miss", label: "miss", outcomes: ["miss"] },
    { id: "false_presence", label: "false presence", outcomes: ["false_presence"] },
    { id: "correct_absence", label: "correct absence", outcomes: ["correct_absence"] },
    { id: "unlabeled", label: "unlabeled", outcomes: ["unlabeled"] },
  ],
  rank: { desc: "most present", asc: "least" },
  outcomeNote: (verdicts, targetLabel) => (
    <>
      Outcomes against <span className="font-mono text-fg">{targetLabel ?? "the class"}</span>
      {verdicts.rationale ? ` — ${verdicts.rationale}` : ""}; a find below 0.5 IoU is low.
    </>
  ),
  Scored: ({ metrics, state, verdicts }) => (
    <>
      <Headline metrics={metrics} subset={state.subset} keys={SEGMENTATION_HEADLINE} />
      <OutcomeTally verdicts={verdicts} />
    </>
  ),
  MetricTables: ({ experimentId, metrics }) => (
    <Metrics
      experimentId={experimentId}
      metrics={metrics}
      note={
        <>
          Threshold-free except the mask itself, which is the method&apos;s own or its map at{" "}
          <span className="font-mono">foreground probability ≥ 0.5</span>. Counted per image;
          unlabelled images are left out and counted.
        </>
      }
      body={(subset, values) => <SegmentationSubset subset={subset} metrics={values} />}
    />
  ),
  Benchmark: ({ verdicts }) => <OverlapDistribution verdicts={verdicts} />,
};

const SEMANTIC_HEADLINE = [
  { key: "mean_iou", label: "mean IoU" },
  { key: "pixel_accuracy", label: "pixel accuracy" },
  { key: "mean_class_accuracy", label: "mean class accuracy" },
  { key: "frequency_weighted_iou", label: "frequency-weighted IoU" },
];

const SEMANTIC: TaskView = {
  filters: [{ id: "all", label: "all", outcomes: undefined }],
  rank: { desc: "most found", asc: "least" },
  outcomeNote: () => (
    <>Ranked by the share of each image given a class; there is no per-sample verdict.</>
  ),
  Scored: ({ metrics, state }) => (
    <Headline metrics={metrics} subset={state.subset} keys={SEMANTIC_HEADLINE} />
  ),
  MetricTables: ({ experimentId, metrics }) => (
    <Metrics
      experimentId={experimentId}
      metrics={metrics}
      note={
        <>
          Read off one confusion matrix per subset, from the label maps the method wrote —
          nothing is thresholded. Counted per image; unlabelled images are left out and
          counted.
        </>
      }
      body={(subset, values) => <SemanticSubset subset={subset} metrics={values} />}
    />
  ),
  Benchmark: () => (
    <Panel title="Per class">
      <Empty>Each class&apos;s IoU is in the metric tables on Overview.</Empty>
    </Panel>
  ),
};

const VIEWS: Partial<Record<Task, TaskView>> = {
  anomaly: ANOMALY,
  few_shot_segmentation: FEW_SHOT,
  semantic_segmentation: SEMANTIC,
};

/** The view for a run's task; an unknown or still-loading task reads as anomaly. */
export function taskView(task: Task | undefined): TaskView {
  return (task && VIEWS[task]) || ANOMALY;
}

/** The per-sample report is read from every stored map, so it arrives after the metrics. */
function Pending({ title, verdicts }: { title: string; verdicts: Verdicts }) {
  return (
    <Panel title={title}>
      {verdicts.error ? <ErrorBox>{verdicts.error.message}</ErrorBox> : <SkeletonRows rows={2} />}
    </Panel>
  );
}

function OutcomeTally({ verdicts: report }: { verdicts: Verdicts }) {
  const title = "What it did, per sample";
  if (report.isPending || report.error) return <Pending title={title} verdicts={report} />;
  const verdicts = report.all;
  if (verdicts.length === 0) return null;
  const count = (outcome: string) => verdicts.filter((entry) => entry.outcome === outcome).length;
  return (
    <Panel title={title}>
      <CountRun
        counts={SEGMENTATION_OUTCOMES.map((outcome) => [
          outcome.replace("_", " "),
          count(outcome),
          OUTCOME_TONE[outcome] ?? "unlabeled",
        ])}
      />
    </Panel>
  );
}

function SegmentationSubset({ subset, metrics }: { subset: Subset; metrics: MetricValue }) {
  const images = (metrics.images ?? {}) as Record<string, number>;
  return (
    <div className="flex flex-col gap-2">
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        {subset}
        <CountRun
          counts={[
            ["present", images.present ?? 0, "defect"],
            ["absent", images.absent ?? 0, "normal"],
            ["unlabeled", images.unlabeled ?? 0, "unlabeled"],
          ]}
        />
      </h3>
      <MetricList rows={segmentationRows(metrics)} />
      {(images.without_prediction ?? 0) > 0 && (
        <p className="text-xs text-warn">
          {images.without_prediction} scored images have neither a map nor a mask.
        </p>
      )}
      {timingRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-semibold text-fg">Timing</h4>
          <MetricList rows={timingRows(metrics)} />
        </>
      )}
    </div>
  );
}

function SemanticSubset({ subset, metrics }: { subset: Subset; metrics: MetricValue }) {
  const images = (metrics.images ?? {}) as Record<string, number>;
  return (
    <div className="flex flex-col gap-2">
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        {subset}
        <CountRun
          counts={[
            ["labelled", images.labelled ?? 0, "normal"],
            ["unlabeled", images.unlabeled ?? 0, "unlabeled"],
          ]}
        />
      </h3>
      <MetricList rows={semanticRows(metrics)} />
      {(images.without_prediction ?? 0) > 0 && (
        <p className="text-xs text-warn">
          {images.without_prediction} scored images have no label map.
        </p>
      )}
      {timingRows(metrics).length > 0 && (
        <>
          <h4 className="mt-1 text-xs font-semibold text-fg">Timing</h4>
          <MetricList rows={timingRows(metrics)} />
        </>
      )}
    </div>
  );
}

const BUCKETS = [
  { label: "≥ 0.75", low: 0.75, high: Infinity },
  { label: "0.5 – 0.75", low: 0.5, high: 0.75 },
  { label: "0.25 – 0.5", low: 0.25, high: 0.5 },
  { label: "< 0.25", low: -Infinity, high: 0.25 },
];

function OverlapDistribution({ verdicts: report }: { verdicts: Verdicts }) {
  if (report.isPending || report.error) {
    return <Pending title="Overlap and absence" verdicts={report} />;
  }
  const verdicts = report.all;
  const overlaps = verdicts
    .map((entry) => (entry as SampleVerdict & { iou?: number | null }).iou)
    .filter((value): value is number => typeof value === "number");
  const absent = verdicts.filter(
    (entry) => entry.outcome === "correct_absence" || entry.outcome === "false_presence",
  );
  if (overlaps.length === 0 && absent.length === 0) {
    return (
      <Panel title="Overlap">
        <Empty>No sample in this subset has truth for the class.</Empty>
      </Panel>
    );
  }
  return (
    <Panel title="Overlap and absence">
      <div className="grid gap-8 lg:grid-cols-2">
        <StackedBars
          label="Samples that show the class, by IoU"
          rows={BUCKETS.map((bucket) => ({
            label: bucket.label,
            segments: [
              {
                name: "samples",
                value: overlaps.filter((value) => value >= bucket.low && value < bucket.high)
                  .length,
                colour: bucket.low >= 0.5 ? NORMAL_COLOUR : DEFECT_COLOUR,
              },
            ],
          }))}
        />
        <StackedBars
          label="Samples without it"
          rows={[
            {
              label: "absent",
              segments: [
                {
                  name: "correct absence",
                  value: absent.filter((entry) => entry.outcome === "correct_absence").length,
                  colour: NORMAL_COLOUR,
                },
                {
                  name: "false presence",
                  value: absent.filter((entry) => entry.outcome === "false_presence").length,
                  colour: DEFECT_COLOUR,
                },
              ],
            },
          ]}
        />
      </div>
      <p className="mt-4 text-xs text-fg-muted">
        How the overlap moves with the number of references is a question across runs — draw
        references with several shot counts and seeds, and compare them.
      </p>
    </Panel>
  );
}
