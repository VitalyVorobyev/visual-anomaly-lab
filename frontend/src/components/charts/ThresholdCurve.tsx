/**
 * Precision, recall and F1 as functions of the threshold.
 *
 * The confusion matrix says what one cut produces. This says what *every* cut would have
 * produced, which is the question a slider makes someone ask and the one the screen could
 * not previously answer: a matrix at 0.065 does not tell you whether recall falls off a
 * cliff at 0.07 or drifts down over the whole range.
 *
 * Derived from the PR curve the server already computes, which is why it costs one array
 * on the wire rather than an endpoint: precision and recall at each distinct score are the
 * same numbers `average_precision` sums, and F1 is their harmonic mean. One implementation
 * for the chart and the scalar, so they cannot disagree (§8) — and because the server cuts
 * at the same rule the threshold report classifies by, the point under the active rule is
 * the confusion matrix drawn beside it.
 *
 * The suggested threshold maximizes F1, so its dashed rule lands on the F1 peak. That is
 * not decoration: it is the chart checking the server's arithmetic in front of the reader.
 */

import type { Curve } from "../../api/client";
import { Empty, LineChart, seriesColour, type Scale } from "@vitavision/lab-ui";

export interface ThresholdCurveProps {
  /** The PR curve, whose `t` carries the score at each point. */
  curve: Curve | null | undefined;
  /** Where the slider currently sits, drawn as a solid rule. */
  active: number;
  /** The server's opening position, drawn as a dashed rule. */
  suggested?: number | null;
  /** The slider's own domain, so the chart and the control share an x axis. */
  domain: [number, number];
  label?: string;
}

interface Point {
  x: number;
  y: number;
}

export function ThresholdCurve({
  curve,
  active,
  suggested,
  domain,
  label = "Precision, recall and F1 against the threshold",
}: ThresholdCurveProps) {
  // `t` is empty for a ROC curve and for any index written before thresholds were carried.
  // Saying so beats drawing precision against recall under an axis labelled "threshold".
  if (!curve || curve.t.length === 0) {
    return (
      <Empty>
        This subset has only one class in it, so there is no curve to draw against the
        threshold.
      </Empty>
    );
  }

  const recall: Point[] = [];
  const precision: Point[] = [];
  const f1: Point[] = [];

  for (const [index, t] of curve.t.entries()) {
    const r = curve.x[index] ?? 0;
    const p = curve.y[index] ?? 0;
    recall.push({ x: t, y: r });
    precision.push({ x: t, y: p });
    // Both zero is a real state — a cut above every score catches nothing — and the
    // harmonic mean is 0 there rather than NaN.
    f1.push({ x: t, y: p + r === 0 ? 0 : (2 * p * r) / (p + r) });
  }

  const rules = (x: Scale, y: Scale) => (
    <>
      {suggested !== null && suggested !== undefined && (
        <line
          x1={x.project(suggested)}
          x2={x.project(suggested)}
          y1={y.project(0)}
          y2={y.project(1)}
          stroke="currentColor"
          strokeWidth={0.75}
          strokeDasharray="4 3"
          opacity={0.5}
        />
      )}
      <line
        x1={x.project(active)}
        x2={x.project(active)}
        y1={y.project(0)}
        y2={y.project(1)}
        stroke="currentColor"
        strokeWidth={1}
        opacity={0.8}
      />
    </>
  );

  return (
    <LineChart
      label={label}
      variant="wide"
      xLabel="threshold"
      xDomain={domain}
      yDomain={[0, 1]}
      underlay={rules}
      series={[
        { name: "precision", points: precision, colour: seriesColour(0) },
        { name: "recall", points: recall, colour: seriesColour(1) },
        { name: "F1", points: f1, colour: seriesColour(2) },
      ]}
      footer={
        <span className="text-fg-subtle">
          The solid rule is the threshold in force; the dashed one is the opening position,
          which maximizes F1.
          {curve.dropped > 0 &&
            ` ${curve.total} cuts, downsampled to ${curve.t.length} for drawing.`}
        </span>
      }
    />
  );
}
