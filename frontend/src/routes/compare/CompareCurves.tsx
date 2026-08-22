/**
 * Every run's ROC and PR curve on one pair of axes.
 *
 * Composition over the existing chart rather than a comparison-specific one: `useCurves`
 * already serves a run's arrays, both axes are rates so both domains are fixed at `[0, 1]`,
 * and a curve is therefore the one thing on this screen that *is* directly comparable
 * without any rule. Overlaying them is the whole feature.
 *
 * One query per run, which is N small requests rather than one large one. That is the right
 * shape here: each is independently cached under the key the single-run Benchmark tab
 * already uses, so a run looked at there is drawn without a request, and adding a column
 * costs exactly one.
 *
 * Pixel-level curves are absent, as they are on the single-run tab: the accumulator streams
 * its histograms and discards them (handbook evaluation.md), so drawing one would mean re-reading every
 * map — the expensive pass this whole layer exists to avoid.
 */

import type { ComparedRun, Subset } from "../../api/client";
import { Callout, Empty, LineChart, Panel, SkeletonRows, seriesColour, type Scale, type Series } from "@vitavision/lab-ui";
import { useCurveSets } from "../../hooks/useComparison";

export function CompareCurves({
  runs,
  subset,
}: {
  runs: ComparedRun[];
  subset: Subset | undefined;
}) {
  const stale = runs.some((run) => run.ground_truth_stale);
  const curves = useCurveSets(
    runs.map((run) => run.id),
    subset,
    !stale,
  );
  const pending = curves.some((query) => query.isPending);

  const roc: Series[] = [];
  const pr: Series[] = [];
  runs.forEach((run, index) => {
    const data = curves[index]?.data;
    const colour = seriesColour(index);
    if (data?.sample_roc) roc.push({ name: run.name, colour, points: pointsOf(data.sample_roc) });
    if (data?.sample_pr) pr.push({ name: run.name, colour, points: pointsOf(data.sample_pr) });
  });

  return (
    <Panel title={`Curves — ${subset ?? "every scored subset"}`}>
      {stale && (
        <Callout tone="warning" title="Recompute before comparing curves">
          At least one run was measured against older labels or masks. The curves stay
          hidden until every run is reevaluated against the same current ground truth.
        </Callout>
      )}
      {!stale && pending && <SkeletonRows rows={3} />}
      {!stale && !pending && roc.length === 0 && pr.length === 0 && (
        <Empty>
          None of these runs has both classes in this subset, so there is no curve to draw.
        </Empty>
      )}
      {!stale && !pending && (roc.length > 0 || pr.length > 0) && (
        <div className="grid gap-8 lg:grid-cols-2">
          {roc.length > 0 && (
            <LineChart
              label="Sample-level ROC"
              series={roc}
              xLabel="false-positive rate"
              yLabel="true-positive rate"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
              underlay={chanceDiagonal}
            />
          )}
          {pr.length > 0 && (
            <LineChart
              label="Sample-level PR"
              series={pr}
              xLabel="recall"
              yLabel="precision"
              xDomain={[0, 1]}
              yDomain={[0, 1]}
            />
          )}
        </div>
      )}
      {!stale && <p className="mt-4 text-xs text-fg-muted">
        The areas under these curves are the ROC-AUC and AP in the table above, computed by
        the same implementation — a curve here cannot disagree with a number there.
      </p>}
    </Panel>
  );
}

function pointsOf(curve: { x: number[]; y: number[] }): { x: number; y: number }[] {
  return curve.x.map((x, index) => ({ x, y: curve.y[index] ?? 0 }));
}

/** The line a coin-flip classifier would draw, for the eye to measure every curve against. */
function chanceDiagonal(x: Scale, y: Scale) {
  return (
    <line
      x1={x.project(0)}
      y1={y.project(0)}
      x2={x.project(1)}
      y2={y.project(1)}
      stroke="currentColor"
      strokeWidth={0.75}
      strokeDasharray="4 3"
      opacity={0.4}
    />
  );
}
