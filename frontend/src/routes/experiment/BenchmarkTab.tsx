/**
 * How well it did, drawn rather than tabulated.
 *
 * Every curve here integrates to a number already on the Overview tab, and both come from
 * one implementation on the server — so a chart cannot disagree with the metric printed
 * beside it. A subset that cannot support a curve says so instead of drawing a chance
 * diagonal, for the same reason a metric that could not be computed renders as a dash.
 */

import { useState } from "react";

import type { MetricSummary, Subset } from "../../api/client";
import { groupingNote } from "../../api/metrics";
import { CurveChart } from "../../components/charts/CurveChart";
import { Callout, Empty, Panel, SkeletonRows } from "@vitavision/lab-ui";
import { useCurves } from "../../hooks/useExperiments";
import { Results } from "./ResultsPanel";

function scalar(metrics: MetricSummary | undefined, key: string): number | null {
  const value = metrics?.metrics?.[key];
  return typeof value === "number" ? value : null;
}

export function BenchmarkTab({
  experimentId,
  subsets,
  metrics,
}: {
  experimentId: number;
  subsets: Subset[];
  metrics: MetricSummary[];
}) {
  const [subset, setSubset] = useState<Subset>(subsets[subsets.length - 1] ?? "test");
  const forSubset = metrics.find((entry) => entry.subset === subset);
  const stale = forSubset?.ground_truth_stale ?? false;
  const curves = useCurves(experimentId, subset, !stale);

  const absent = "This subset has only one class in it, so there is no curve to draw.";
  /* One image per sample means `max` over a single value, so the image-level curve is the
     sample-level curve redrawn — four charts, two findings. Derived from the counts the
     evaluation layer recorded, so a grouped dataset still gets both pairs. */
  const ungrouped = groupingNote(forSubset?.metrics ?? {});

  return (
    <div className="flex flex-col gap-6">
      <Panel title={`Curves — ${subset}`}>
        {stale && (
          <Callout tone="warning" title="Recompute before reading these curves">
            Labels or masks changed after the stored metrics were computed. The charts
            stay hidden so a current curve is never paired with an older area value.
          </Callout>
        )}
        {!stale && curves.isPending && <SkeletonRows rows={3} />}
        {!stale && curves.data && (
          <div className="grid gap-8 lg:grid-cols-2">
            <CurveChart
              curve={curves.data.sample_roc}
              kind="roc"
              label={ungrouped ? "ROC curve" : "Sample-level ROC curve"}
              area={scalar(forSubset, "sample_roc_auc")}
              areaLabel={ungrouped ? "ROC-AUC" : "sample ROC-AUC"}
              absent={absent}
            />
            <CurveChart
              curve={curves.data.sample_pr}
              kind="pr"
              label={ungrouped ? "PR curve" : "Sample-level PR curve"}
              area={scalar(forSubset, "sample_average_precision")}
              areaLabel={ungrouped ? "AP" : "sample AP"}
              absent={absent}
            />
            {ungrouped === null && (
              <>
                <CurveChart
                  curve={curves.data.image_roc}
                  kind="roc"
                  label="Image-level ROC curve"
                  area={scalar(forSubset, "image_roc_auc")}
                  areaLabel="image ROC-AUC"
                  absent={absent}
                />
                <CurveChart
                  curve={curves.data.image_pr}
                  kind="pr"
                  label="Image-level PR curve"
                  area={scalar(forSubset, "image_average_precision")}
                  areaLabel="image AP"
                  absent={absent}
                />
              </>
            )}
          </div>
        )}
        {!stale && ungrouped !== null && (
          <p className="mt-4 text-xs text-fg-muted">{ungrouped}</p>
        )}
        {!stale && (
          <p className="mt-4 text-xs text-fg-muted">
            {/* Stating the gap rather than leaving a reader to wonder where it went. */}
            Pixel-level curves are not drawn. The pixel metrics stream their histograms
            and discard them so memory stays constant in the number of test images
            (ADR-0017); reconstructing the curve would mean re-reading every anomaly map.
            The pixel ROC-AUC and AU-PRO themselves are on the Overview tab.
          </p>
        )}
      </Panel>

      {subsets.length === 0 ? (
        <Panel title="Results">
          <Empty>Nothing has been scored yet.</Empty>
        </Panel>
      ) : (
        <Results
          experimentId={experimentId}
          subsets={subsets}
          subset={subset}
          onSubset={setSubset}
          charts
        />
      )}
    </div>
  );
}
