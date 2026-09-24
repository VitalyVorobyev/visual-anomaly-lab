/**
 * The threshold slider, the confusion matrix and the classified rows.
 *
 * M3's results panel, moved, with two M4 additions that read the *same* server-classified
 * rows: the score histogram with the threshold drawn on it, and the per-defect-type
 * breakdown. Neither reapplies the threshold rule — the server has already tagged every
 * row with its outcome, and holding that rule in TypeScript as well as Python would let
 * the two drift (§12).
 */

import { Link } from "react-router";

import type { MetricSummary, SampleVerdict, Subset } from "../../api/client";
import type { MetricValue } from "../../api/metrics";
import { localizationTolerancePx } from "../../api/metrics";
import type { Outcome, ResultsState } from "../../api/resultsState";
import { MISTAKE_OUTCOMES, writeResultsState } from "../../api/resultsState";
import { ThresholdCurve } from "../../components/charts/ThresholdCurve";
import { DEFECT_COLOUR, Empty, ErrorBox, InfoHint, NORMAL_COLOUR, Panel, ScoreHistogram, Select, Slider, StackedBars, type Tone } from "@vitavision/lab-ui";
import { useCurves, useResults, useThreshold } from "../../hooks/useExperiments";

export const OUTCOME_TONE: Record<string, Tone> = {
  tp: "normal",
  tn: "normal",
  fp: "warning",
  fn: "defect",
  unlabeled: "unlabeled",
};

export const OUTCOME_LABEL: Record<string, string> = {
  tp: "true positive",
  tn: "true negative",
  fp: "false positive",
  fn: "false negative",
  unlabeled: "unlabeled",
};

/**
 * The localization verdict's own vocabulary, deliberately kept apart from the outcome's.
 *
 * It is **orthogonal to the outcome, not a fifth value of it**: a true positive that fired
 * off target is still a true positive, and one that is `null` here is not a miss — the
 * question does not apply to a normal part, to a defect nobody annotated, or to a method
 * that wrote no map. So this is a second badge beside the outcome rather than a widening of
 * it, and `null` draws nothing at all.
 */
export function localizationBadge(
  localized: boolean | null | undefined,
): { tone: Tone; label: string } | null {
  if (localized === null || localized === undefined) return null;
  return localized
    ? { tone: "normal", label: "localized" }
    : { tone: "warning", label: "off target" };
}

/**
 * How many of the rows that could be judged were localized.
 *
 * `judged` counts every row the question applies to — the ones carrying a verdict — and
 * never the whole subset, so the denominator is "annotated defects whose map produced a
 * peak" rather than "defects". Deliberately not filtered by outcome: the verdict compares
 * the map against the ground truth and never against a cut, so this pair is the same at
 * every position of the slider it is printed beside, and filtering to the detections would
 * silently make it move.
 */
export function localizationSummary(
  rows: readonly { localized?: boolean | null }[],
): { localized: number; judged: number } {
  let localized = 0;
  let judged = 0;
  for (const row of rows) {
    if (row.localized === null || row.localized === undefined) continue;
    judged += 1;
    if (row.localized) localized += 1;
  }
  return { localized, judged };
}

export function Results({
  experimentId,
  subsets,
  state,
  onChange,
  metrics,
  charts = false,
}: {
  experimentId: number;
  subsets: Subset[];
  /**
   * The experiment's shared results state — the same object the Samples tab filters by.
   *
   * The subset and the threshold used to be this panel's own `useState`, one copy on
   * Overview and another on Benchmark, while the gallery read them from the URL that
   * nothing wrote. A cut chosen here therefore never reached the TP/FP badges one tab
   * over. Reading and writing the URL state is what makes it one threshold.
   */
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  /**
   * The stored metric sets, read for one thing only: the tolerance the localization
   * verdicts were decided against, so the strip can name the radius rather than leave the
   * rule abstract. Absent is fine — the counts stand without it.
   */
  metrics?: MetricSummary[];
  /** The benchmark tab wants the distribution charts; the overview does not. */
  charts?: boolean;
}) {
  const subset = state.subset ?? subsets.at(-1) ?? "test";
  const results = useResults(experimentId, subset);
  // `undefined` is the server's suggestion, which moves with the subset; a drag is written
  // to the URL and survives changing tab, reloading and the round trip to a sample.
  const active = state.threshold ?? results.data?.suggested_threshold ?? 0;
  const report = useThreshold(experimentId, subset, active);
  // The same curve the Benchmark tab draws, read here against the threshold axis instead
  // of against recall. One request, cached across both tabs.
  const curves = useCurves(experimentId, subset);

  const span = (results.data?.score_max ?? 1) - (results.data?.score_min ?? 0);
  const step = span > 0 ? span / 500 : 0.001;

  // This subset's stored metrics, read for the tolerance alone. `null` for a run evaluated
  // before the verdicts existed, which is why the strip has a phrasing that works without it.
  const stored = (metrics?.find((entry) => entry.subset === subset)?.metrics ?? {}) as MetricValue;

  return (
    <Panel
      title="Results"
      actions={
        <Select
          className="w-40"
          aria-label="Subset"
          value={subset}
          options={subsets.map((name) => ({ value: name, label: name }))}
          // A chosen cut is a position on *this* subset's score range, so it does not
          // survive a change of subset; the next one opens on its own suggestion.
          onValueChange={(value) => onChange({ subset: value as Subset, threshold: undefined })}
        />
      }
    >
      {results.error && <ErrorBox>{results.error.message}</ErrorBox>}
      {results.data && results.data.samples.length === 0 && (
        <Empty>Nothing scored in this subset.</Empty>
      )}

      {results.data && results.data.samples.length > 0 && (
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-3">
              <span className="shrink-0 text-xs font-medium text-fg">Threshold</span>
              <Slider
                aria-label="Threshold"
                min={results.data.score_min}
                max={results.data.score_max}
                step={step}
                value={active}
                onValueChange={(value) => onChange({ threshold: value })}
                readout={<span className="inline-block w-16 text-right">{active.toFixed(4)}</span>}
              />
            </div>
            <p className="text-xs text-fg-muted">
              {state.threshold === undefined ? "Opens" : "Opened"} at{" "}
              {results.data.suggested_threshold.toFixed(4)} — {results.data.threshold_rationale}.
              {state.threshold !== undefined && (
                <>
                  {" "}
                  <button
                    type="button"
                    className="text-signal underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-signal"
                    onClick={() => onChange({ threshold: undefined })}
                  >
                    Back to the suggestion
                  </button>
                </>
              )}
            </p>
          </div>

          {/* Directly under the slider, sharing its x axis: what every other cut would
              have produced, so moving the control is a choice rather than a search. */}
          <ThresholdCurve
            curve={curves.data?.sample_pr}
            active={active}
            suggested={results.data.suggested_threshold}
            domain={[results.data.score_min, results.data.score_max]}
          />

          {report.data && (
            <>
              {charts && <Distributions samples={report.data.samples} threshold={active} />}
              <Confusion
                report={report.data}
                samples={report.data.samples}
                tolerancePx={localizationTolerancePx(stored)}
              />
              {charts && <DefectTypes samples={report.data.samples} />}
              <OutcomeLinks state={{ ...state, subset }} samples={report.data.samples} />
            </>
          )}
        </div>
      )}
    </Panel>
  );
}

/**
 * Where the two classes actually sit, with the threshold on top.
 *
 * The confusion matrix says how many were wrong; this says why — two distributions that
 * barely separate, or one long normal tail crossing the line.
 */
function Distributions({
  samples,
  threshold,
}: {
  samples: SampleVerdict[];
  threshold: number;
}) {
  const normal = samples.filter((s) => s.label === "normal").map((s) => s.score);
  const defect = samples.filter((s) => s.label === "defect").map((s) => s.score);

  if (normal.length === 0 && defect.length === 0) return null;

  return (
    <ScoreHistogram
      label="Score distribution by class"
      normal={normal}
      defect={defect}
      threshold={threshold}
    />
  );
}

/**
 * Caught and missed, per defect type.
 *
 * The type is whatever an adapter recorded in the sample's notes — `Nick`, `Scratch`, a
 * VisA label. It is grouped rather than parsed: this layer has no vocabulary of defect
 * types and should not acquire one.
 */
function DefectTypes({ samples }: { samples: SampleVerdict[] }) {
  const byType = new Map<string, { caught: number; missed: number }>();
  for (const sample of samples) {
    if (sample.label !== "defect") continue;
    const key = sample.notes?.trim() || "unrecorded";
    const row = byType.get(key) ?? { caught: 0, missed: 0 };
    if (sample.outcome === "tp") row.caught += 1;
    else row.missed += 1;
    byType.set(key, row);
  }

  // One bucket called "unrecorded" is a breakdown of nothing — the adapter recorded no
  // types, and the confusion matrix above already says the same thing.
  if (byType.size === 0 || (byType.size === 1 && byType.has("unrecorded"))) return null;

  const rows = [...byType.entries()]
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([label, counts]) => ({
      label,
      segments: [
        { name: "caught", value: counts.caught, colour: NORMAL_COLOUR },
        { name: "missed", value: counts.missed, colour: DEFECT_COLOUR },
      ],
    }));

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold text-fg">
        By defect type, at this threshold
      </h3>
      <StackedBars rows={rows} label="Defects caught and missed by type" />
    </div>
  );
}

export function Confusion({
  report,
  samples = [],
  tolerancePx = null,
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
  /** The classified rows, read for their localization verdicts alone. */
  samples?: readonly { localized?: boolean | null }[];
  /** The resolved tolerance radius, when the run's metrics named one. */
  tolerancePx?: number | null;
}) {
  const { confusion } = report;
  const localization = localizationSummary(samples);
  return (
    <div className="flex flex-wrap items-center gap-6">
      <table className="text-sm">
        <thead>
          <tr className="text-xs text-fg-muted">
            <th className="px-2" />
            <th className="px-2 font-medium">called defect</th>
            <th className="px-2 font-medium">called normal</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          <tr>
            <th className="px-2 text-xs font-medium text-fg-muted">is defect</th>
            <td className="px-2 text-normal">
              {confusion.true_positive}
            </td>
            <td className="px-2 text-defect">{confusion.false_negative}</td>
          </tr>
          <tr>
            <th className="px-2 text-xs font-medium text-fg-muted">is normal</th>
            <td className="px-2 text-warn">{confusion.false_positive}</td>
            <td className="px-2 text-normal">
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
            <dt className="text-xs text-fg-muted">{label}</dt>
            <dd className="font-mono">
              {value === null ? <span className="text-fg-subtle">—</span> : value.toFixed(3)}
            </dd>
          </div>
        ))}

        {/* The one figure in this strip that does not move with the slider above it, which
            is exactly why it needs the hint: read as a neighbour of precision and recall it
            would look like another quantity at this threshold. */}
        <div className="flex flex-col">
          <dt className="flex items-center gap-1 text-xs text-fg-muted">
            localized
            <InfoHint label="What localized means">
              Threshold-free, and unchanged as the slider moves: it asks whether the map&rsquo;s
              peak landed within{" "}
              {tolerancePx === null ? "the resolved tolerance" : `${tolerancePx} px`} of the
              annotated region, never whether a score crossed a cut. Counted over the
              annotated defects whose map produced a peak — a normal part, an unannotated
              defect and a run with no map have no verdict rather than a failed one.
            </InfoHint>
          </dt>
          <dd className="font-mono">
            {localization.judged === 0 ? (
              <span className="text-fg-subtle">—</span>
            ) : (
              `${localization.localized} of ${localization.judged}`
            )}
          </dd>
        </div>
      </dl>
    </div>
  );
}

/**
 * Where each cell of the matrix goes to be looked at.
 *
 * This used to be a second verdict browser — its own outcome filter in local state, its
 * own scrolling list inside the page's scroller, and links that dropped the threshold, so
 * prev/next on the sample page walked a different set from the one listed. The Samples
 * tab is that browser; this hands it the filter, at the threshold in force here.
 */
function OutcomeLinks({
  state,
  samples,
}: {
  state: ResultsState;
  samples: readonly SampleVerdict[];
}) {
  const count = (outcomes: readonly Outcome[]) =>
    samples.filter((sample) => outcomes.includes(sample.outcome as Outcome)).length;
  const targets: { key: string; label: string; count: number; next: Partial<ResultsState> }[] = [
    {
      key: "mistakes",
      label: "mistakes",
      count: count(MISTAKE_OUTCOMES),
      next: { mistakesOnly: true, outcome: undefined },
    },
    ...(["fn", "fp", "tp", "tn"] as const).map((outcome) => ({
      key: outcome,
      label: OUTCOME_PLURAL[outcome],
      count: count([outcome]),
      next: { mistakesOnly: false, outcome },
    })),
  ];

  return (
    <nav aria-label="Open in Samples" className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
      <span className="text-xs text-fg-muted">Open in Samples</span>
      {targets.map((target) =>
        target.count === 0 ? (
          <span key={target.key} className="text-fg-subtle">
            {target.label} <span className="font-mono">0</span>
          </span>
        ) : (
          <Link
            key={target.key}
            to={{ search: writeResultsState({ ...state, ...target.next, tab: "samples" }).toString() }}
            className="text-signal underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-signal"
          >
            {target.label} <span className="font-mono">{target.count}</span>
          </Link>
        ),
      )}
    </nav>
  );
}

const OUTCOME_PLURAL: Record<"tp" | "tn" | "fp" | "fn", string> = {
  tp: "true positives",
  tn: "true negatives",
  fp: "false positives",
  fn: "false negatives",
};
