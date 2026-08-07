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
import { CurveChart } from "../../components/charts/CurveChart";
import { Empty, Panel, SkeletonRows } from "../../components/ui";
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
  const curves = useCurves(experimentId, subset);
  const forSubset = metrics.find((entry) => entry.subset === subset);

  const absent = "This subset has only one class in it, so there is no curve to draw.";

  return (
    <div className="flex flex-col gap-6">
      <Panel title={`Curves — ${subset}`}>
        {curves.isPending && <SkeletonRows rows={3} />}
        {curves.data && (
          <div className="grid gap-8 lg:grid-cols-2">
            <CurveChart
              curve={curves.data.sample_roc}
              kind="roc"
              label="Sample-level ROC curve"
              area={scalar(forSubset, "sample_roc_auc")}
              areaLabel="sample ROC-AUC"
              absent={absent}
            />
            <CurveChart
              curve={curves.data.sample_pr}
              kind="pr"
              label="Sample-level PR curve"
              area={scalar(forSubset, "sample_average_precision")}
              areaLabel="sample AP"
              absent={absent}
            />
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
          </div>
        )}
        <p className="mt-4 text-xs text-fg-muted">
          {/* Stating the gap rather than leaving a reader to wonder where it went. */}
          Pixel-level curves are not drawn. The pixel metrics stream their histograms and
          discard them so memory stays constant in the number of test images (ADR-0017);
          reconstructing the curve would mean re-reading every anomaly map. The pixel
          ROC-AUC and AU-PRO themselves are on the Overview tab.
        </p>
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
