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

import type { SampleVerdict, Subset } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";
import { StackedBars } from "../../components/charts/BarChart";
import { DEFECT_COLOUR, NORMAL_COLOUR } from "../../components/charts/Frame";
import { ScoreHistogram } from "../../components/charts/Histogram";
import { ThresholdCurve } from "../../components/charts/ThresholdCurve";
import { Tabs } from "../../components/Tabs";
import { Badge, Empty, ErrorBox, Panel, Select, Slider } from "../../components/ui";
import type { Tone } from "../../components/ui";
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

export function Results({
  experimentId,
  subsets,
  subset,
  onSubset,
  charts = false,
}: {
  experimentId: number;
  subsets: Subset[];
  subset: Subset;
  onSubset: (subset: Subset) => void;
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
              <Confusion report={report.data} />
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
}) {
  const { confusion } = report;
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
              {sample.notes && <span className="text-xs text-fg-muted">{sample.notes}</span>}
              <span className="ml-auto shrink-0 font-mono text-xs">{sample.score.toFixed(4)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
