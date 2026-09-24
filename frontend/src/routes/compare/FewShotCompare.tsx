/**
 * Few-shot segmentation runs of one class, side by side (ADR-0040).
 *
 * The anomaly comparison fixes the split and varies the method. A few-shot question varies
 * the *references* too — how many, and which draw — so here the split may differ and the
 * class may not. Three readings, in the order they answer "which method, and how many
 * references does it need":
 *
 * 1. **Overlap against references.** Foreground IoU by method and shot count, the mean over
 *    the draws with their spread — the curve the protocol in `measurements.md` is about,
 *    as a table because each cell is a handful of runs, not a function.
 * 2. **Every metric, per run**, on the same grid the anomaly table uses.
 * 3. **Where they disagree**: the queries every run scored, where the outcomes differ.
 *
 * Nothing is compared in score units; every number here is threshold-free or the run's own
 * mask against truth (ADR-0028).
 */

import { useMemo, useState } from "react";
import { Link } from "react-router";

import type { FewShotComparison, FewShotRun } from "../../api/client";
import type { MetricValue } from "../../api/metrics";
import { comparisonRows, formatScore, segmentationRows, timingRows } from "../../api/metrics";
import { Badge, Callout, Empty, Panel, SegmentedControl, Table, type Column } from "@vitavision/lab-ui";
import { useModelTypes } from "../../hooks/useExperiments";
import { OUTCOME_LABEL, OUTCOME_TONE } from "../experiment/ResultsPanel";
import { Grid, SectionRows } from "./MetricTable";

export function FewShotCompare({ report }: { report: FewShotComparison }) {
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
      <ReferenceTable runs={report.runs} />
      <RunMetrics runs={report.runs} />
      <Disagreements report={report} />
    </>
  );
}

function iouOf(run: FewShotRun): number | null {
  const value = (run.metrics as MetricValue).foreground_iou;
  return typeof value === "number" ? value : null;
}

function ReferenceTable({ runs }: { runs: FewShotRun[] }) {
  const methods = useModelTypes();
  const titles = new Map((methods.data?.methods ?? []).map((method) => [method.key, method.title]));
  const shots = [...new Set(runs.map((run) => run.references))].sort((a, b) => a - b);
  const keys = [...new Set(runs.map((run) => run.model_type))];

  const cell = (method: string, count: number) => {
    const values = runs
      .filter((run) => run.model_type === method && run.references === count)
      .map(iouOf)
      .filter((value): value is number => value !== null);
    if (values.length === 0) return <span className="text-fg-subtle">—</span>;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const spread =
      values.length > 1
        ? Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length)
        : null;
    return (
      <span className="font-mono tabular-nums">
        {formatScore(mean)}
        {spread !== null && <span className="text-fg-subtle"> ±{spread.toFixed(3)}</span>}
        <span className="ml-1 text-[11px] text-fg-subtle">n={values.length}</span>
      </span>
    );
  };

  const columns: Column<string>[] = [
    {
      key: "method",
      header: "Method",
      cell: (method) => titles.get(method) ?? method,
    },
    ...shots.map((count) => ({
      key: `shots-${count}`,
      header: count === 1 ? "1 reference" : `${count} references`,
      numeric: true,
      cell: (method: string) => cell(method, count),
    })),
  ];

  return (
    <Panel title="Foreground IoU against the number of references">
      <p className="mb-3 text-xs text-fg-muted">
        The mean over every compared run with that many references, ± its spread across the
        reference draws — which is how much the answer depends on which references were
        chosen.
      </p>
      <Table
        caption="Foreground IoU by method and shot count"
        rows={keys}
        rowKey={(key) => key}
        columns={columns}
      />
    </Panel>
  );
}

function RunMetrics({ runs }: { runs: FewShotRun[] }) {
  const metrics = runs.map((run) => (run.metrics ?? {}) as MetricValue);
  const grid = runs.map((run) => ({
    id: run.id,
    name: run.name,
    model_type: run.model_type,
    note: `${run.references} refs${run.seed === null || run.seed === undefined ? "" : ` · seed ${run.seed}`}`,
  }));
  return (
    <Panel title="Every metric, per run">
      <p className="mb-3 text-xs text-fg-muted">
        On each run&apos;s test queries — every sample but its own references. Read from the
        metric sets the scoring runs stored.
      </p>
      <Grid runs={grid}>
        <SectionRows title="Segmentation" rows={comparisonRows(metrics.map(segmentationRows))} />
        <SectionRows title="Timing" rows={comparisonRows(metrics.map(timingRows))} />
      </Grid>
    </Panel>
  );
}

type Row = FewShotComparison["samples"][number];

function Disagreements({ report }: { report: FewShotComparison }) {
  const [only, setOnly] = useState<"disagree" | "all">("disagree");
  const rows = useMemo(
    () => report.samples.filter((sample) => only === "all" || !sample.agree),
    [report.samples, only],
  );
  const disagreeing = report.samples.filter((sample) => !sample.agree).length;

  const columns: Column<Row>[] = [
    {
      key: "sample",
      header: "Sample",
      cell: (sample) => (
        <span className="font-mono text-xs">
          {sample.group_key}/{sample.external_id}
        </span>
      ),
    },
    ...report.runs.map((run, index) => ({
      key: `run-${run.id}`,
      header: run.name,
      cell: (sample: Row) => {
        const outcome = sample.outcomes[index] ?? "unlabeled";
        const iou = sample.ious[index];
        return (
          <Link
            to={`/experiments/${run.id}/samples/${sample.sample_id}`}
            className="inline-flex items-center gap-1.5"
          >
            <Badge tone={OUTCOME_TONE[outcome] ?? "unlabeled"}>
              {OUTCOME_LABEL[outcome] ?? outcome}
            </Badge>
            {typeof iou === "number" && (
              <span className="font-mono text-[11px] text-fg-muted">{iou.toFixed(2)}</span>
            )}
          </Link>
        );
      },
    })),
  ];

  return (
    <Panel
      title="Where they disagree"
      actions={
        <SegmentedControl
          aria-label="Which queries"
          value={only}
          onValueChange={(value) => setOnly(value === "all" ? "all" : "disagree")}
          options={[
            { value: "disagree", label: `disagreeing · ${disagreeing}` },
            { value: "all", label: `all shared · ${report.samples.length}` },
          ]}
        />
      }
    >
      {report.samples.length === 0 ? (
        <Empty>These runs share no scored query.</Empty>
      ) : rows.length === 0 ? (
        <Empty>Every shared query has the same outcome under every run.</Empty>
      ) : (
        <Table
          caption="Shared queries and each run's outcome"
          rows={rows}
          rowKey={(sample) => sample.sample_id}
          columns={columns}
        />
      )}
    </Panel>
  );
}
