/**
 * The threshold slider, the confusion matrix and the classified rows.
 *
 * M3's results panel, moved, with two M4 additions that read the *same* server-classified
 * rows: the score histogram with the threshold drawn on it, and the per-defect-type
 * breakdown. Neither reapplies the threshold rule — the server has already tagged every
 * row with its outcome, and holding that rule in TypeScript as well as Python would let
 * the two drift (§12).
 */

import { useEffect, useState } from "react";
import { Link } from "react-router";

import type { MetricSummary, SampleVerdict, Subset } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";
import type { MetricValue } from "../../api/metrics";
import { localizationTolerancePx } from "../../api/metrics";
import { ThresholdCurve } from "../../components/charts/ThresholdCurve";
import { Badge, DEFECT_COLOUR, Empty, ErrorBox, InfoHint, NORMAL_COLOUR, Panel, ScoreHistogram, Select, Slider, StackedBars, Tabs, type Tone } from "@vitavision/lab-ui";
import {
  useCurves,
  useResults,
  useSamplePreviews,
  useThreshold,
} from "../../hooks/useExperiments";

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
  subset,
  onSubset,
  metrics,
  charts = false,
}: {
  experimentId: number;
  subsets: Subset[];
  subset: Subset;
  onSubset: (subset: Subset) => void;
  /**
   * The stored metric sets, read for one thing only: the tolerance the localization
   * verdicts were decided against, so the strip can name the radius rather than leave the
   * rule abstract. Absent is fine — the counts stand without it.
   */
  metrics?: MetricSummary[];
  /** The benchmark tab wants the distribution charts; the overview does not. */
  charts?: boolean;
}) {
  const results = useResults(experimentId, subset);
  const [threshold, setThreshold] = useState<number | null>(null);

  // The suggested threshold is the starting position, and it moves when the subset does —
  // but an operator's own drag must survive a re-render, so it is only adopted when the
  // slider has never been touched for this subset.
  useEffect(() => setThreshold(null), [subset]);
  const active = threshold ?? results.data?.suggested_threshold ?? 0;
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
          onValueChange={(value) => onSubset(value as Subset)}
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
                onValueChange={setThreshold}
                readout={<span className="inline-block w-16 text-right">{active.toFixed(4)}</span>}
              />
            </div>
            <p className="text-xs text-fg-muted">
              Opens at {results.data.suggested_threshold.toFixed(4)} —{" "}
              {results.data.threshold_rationale}.
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
              <VerdictTable
                experimentId={experimentId}
                subset={subset}
                samples={report.data.samples}
              />
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

function VerdictTable({
  experimentId,
  subset,
  samples,
}: {
  experimentId: number;
  subset: Subset;
  samples: SampleVerdict[];
}) {
  const [filter, setFilter] = useState<string>("all");
  /*
   * One image standing for each sample, so a row is a picture rather than a path.
   *
   * A list of `candle/Data/Images/Anomaly/097` tells a reader nothing they can act on; the
   * question a results table exists to answer — *what does the model think this looks
   * like* — needs the thing itself. The previews endpoint already exists for the gallery
   * and is keyed by sample, so this costs one cached request and no backend change; the
   * path stays beside the thumbnail, because it is what identifies the sample to anyone
   * going back to the source tree.
   */
  const previews = useSamplePreviews(experimentId, subset);
  const imageBySample = new Map(
    (previews.data ?? []).map((preview) => [preview.sample_id, preview.image_id]),
  );
  // Already classified by the server, at the threshold currently in force. The rule "a
  // score at or above the threshold is a defect" therefore exists in exactly one place.
  const classified = samples;
  const shown = filter === "all" ? classified : classified.filter((s) => s.outcome === filter);

  const keys = ["all", "tp", "fp", "tn", "fn", "unlabeled"];

  return (
    <div className="flex flex-col gap-2">
      <Tabs
        label="Outcome filter"
        active={filter}
        onSelect={setFilter}
        items={keys.map((key) => {
          const count =
            key === "all" ? classified.length : classified.filter((s) => s.outcome === key).length;
          return {
            id: key,
            label: key === "all" ? "all" : (OUTCOME_LABEL[key] ?? key),
            count,
            disabled: count === 0 && key !== "all",
          };
        })}
      />

      <ul className="max-h-[32rem] divide-y divide-line overflow-y-auto text-sm">
        {shown.map((sample) => {
          const path = `${sample.group_key}/${sample.external_id}`;
          const imageId = imageBySample.get(sample.sample_id);
          // Only the failure is marked. A badge on every localized row would put a second
          // column of green beside the outcome and bury the handful of rows that got the
          // right answer from the wrong pixels, which is the only thing to scan for here.
          const offTarget = sample.localized === false;
          return (
            <li key={sample.sample_id} className="flex items-center gap-3 py-1.5">
              {/* A fixed width so the badges line up into columns down the list. Ragged
                  columns turn "scan for the false positives" into reading every row. */}
              <Link
                to={`/experiments/${experimentId}/samples/${sample.sample_id}`}
                className="flex w-72 shrink-0 items-center gap-2.5 hover:underline"
                title={path}
              >
                {/* A fixed box whatever the source aspect ratio, so the rows stay a list
                    rather than a ragged column. `object-cover` crops; the sample page is
                    one click away for the whole picture. */}
                <span className="size-14 shrink-0 overflow-hidden rounded border border-line bg-raised">
                  {imageId !== undefined && (
                    <img
                      src={imageUrl(imageId, "thumb")}
                      alt=""
                      loading="lazy"
                      className="size-full object-cover"
                    />
                  )}
                </span>
                <span className="truncate font-mono text-[0.6875rem] text-fg-muted">{path}</span>
              </Link>
              <Badge tone={sample.label === "defect" ? "defect" : "normal"}>{sample.label}</Badge>
              <Badge tone={OUTCOME_TONE[sample.outcome] ?? "neutral"}>
                {OUTCOME_LABEL[sample.outcome] ?? sample.outcome}
              </Badge>
              {offTarget && <Badge tone="warning">off target</Badge>}
              {sample.notes && <span className="text-xs text-fg-muted">{sample.notes}</span>}
              <span className="ml-auto shrink-0 font-mono text-xs">{sample.score.toFixed(4)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
