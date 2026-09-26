/**
 * Object detection runs of one split and one class list, side by side (ADR-0039).
 *
 * Only what needs no cut crosses a column: COCO's AP family and recall, per class, and the
 * timing. Each run's verdicts are drawn at a confidence cut resolved from its own detections,
 * and a confidence means nothing outside its run, so the cut, and the F1 it reaches, stay on
 * that run's Overview and the server leaves them out of this report (ADR-0028).
 */

import type { DetectionComparison, Subset } from "../../api/client";
import { comparisonRows, objectDetectionRows, timingRows } from "../../api/metrics";
import { Callout, Panel, Select } from "@vitavision/lab-ui";
import { Grid, SectionRows } from "./MetricTable";

export function DetectionCompare({
  report,
  onSubset,
}: {
  report: DetectionComparison;
  onSubset: (subset: Subset) => void;
}) {
  const metrics = report.runs.map((run) => (run.metrics ?? {}));
  const grid = report.runs.map((run) => ({
    id: run.id,
    name: run.name,
    model_type: run.model_type,
    note: run.scored ? undefined : "not scored on this subset",
  }));
  return (
    <>
      {report.warnings.length > 0 && (
        <Callout tone="warning" title="Read these numbers with care">
          <ul className="flex list-disc flex-col gap-1 pl-4">
            {report.warnings.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </Callout>
      )}
      <Panel
        title="Every threshold-free metric, per run"
        actions={
          report.subsets.length > 1 ? (
            <Select
              className="w-32"
              aria-label="Subset"
              value={report.subset ?? ""}
              options={report.subsets.map((name) => ({ value: name, label: name }))}
              onValueChange={(value) => onSubset(value as Subset)}
            />
          ) : undefined
        }
      >
        <p className="mb-3 text-xs text-fg-muted">
          COCO&apos;s AP and recall over {report.classes.join(", ")}, from the metric sets each
          run stored. Each run&apos;s confidence cut is resolved from its own detections and is
          not comparable across runs, so it and the F1 it reaches are on the run&apos;s own
          Overview, not here.
        </p>
        <Grid runs={grid}>
          <SectionRows title="Detection" rows={comparisonRows(metrics.map(objectDetectionRows))} />
          <SectionRows title="Timing" rows={comparisonRows(metrics.map(timingRows))} />
        </Grid>
      </Panel>
    </>
  );
}
