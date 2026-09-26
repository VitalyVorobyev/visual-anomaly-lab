/**
 * What a split contains, as one bar and one line of counts per subset.
 *
 * The bar is divided by subset in proportion to its samples, and each subset by what its
 * task reads (ADR-0041): an anomaly split by verdict — normal, defect, unlabelled — and a
 * split for a class task by class. A few-shot split names one class, so it is divided into
 * the samples that show it and the ones that do not; a split by class is divided by the
 * dataset's own class colours. The same component reads a stored split and a dry run, so
 * what a preset promises and what it creates are drawn the same way.
 */

import type { CSSProperties, ReactNode } from "react";

import { cn, CountRun, type Tone } from "@vitavision/lab-ui";

import type { SplitParams, SubsetComposition } from "../api/client";

export interface ClassInfo {
  key: string;
  name: string;
  color: string;
}

export type CompositionMode =
  | { kind: "labels" }
  | { kind: "class"; focus: string }
  | { kind: "classes" };

/**
 * What a split's bar is divided by: verdicts for the strategies that train on normals, the
 * one class a few-shot draw names, and every class otherwise — a hand-picked split of a
 * dataset with class truth is read by class too.
 */
export function compositionMode(
  strategy: string,
  params: { label_key?: SplitParams["label_key"] | undefined },
  composition: readonly SubsetComposition[],
): CompositionMode {
  if (strategy === "few_shot" && params.label_key) {
    return { kind: "class", focus: params.label_key };
  }
  if (strategy === "normal_only_train" || strategy === "imported") return { kind: "labels" };
  const byClass = composition.some((row) => (row.classes ?? []).length > 0);
  return byClass ? { kind: "classes" } : { kind: "labels" };
}

interface Segment {
  label: string;
  count: number;
  className?: string;
  style?: CSSProperties;
  tone: Tone;
}

function segments(
  row: SubsetComposition,
  mode: CompositionMode,
  classes: readonly ClassInfo[],
): Segment[] {
  if (mode.kind === "labels") {
    return [
      { label: "normal", count: row.normal, className: "bg-normal", tone: "normal" },
      { label: "defect", count: row.defect, className: "bg-defect", tone: "defect" },
      { label: "unlabeled", count: row.unlabeled, className: "bg-line-strong", tone: "unlabeled" },
    ];
  }
  const shown = new Map((row.classes ?? []).map((entry) => [entry.key, entry.samples]));
  const names = new Map(classes.map((entry) => [entry.key, entry]));
  if (mode.kind === "class") {
    const showing = shown.get(mode.focus) ?? 0;
    return [
      {
        label: `show ${names.get(mode.focus)?.name ?? mode.focus}`,
        count: showing,
        className: "bg-signal",
        tone: "info",
      },
      {
        label: "without it",
        count: row.total - showing,
        className: "bg-line-strong",
        tone: "neutral",
      },
    ];
  }
  return [...shown.entries()].map(([key, count]) => {
    const info = names.get(key);
    return {
      label: info?.name ?? key,
      count,
      // A class's colour is data the dataset chose, not a theme decision, so it is a style.
      style: info ? { backgroundColor: info.color } : undefined,
      className: info ? undefined : "bg-line-strong",
      tone: "neutral",
    };
  });
}

/** Past this many classes a line per class stops being readable, and a range says more. */
const MAX_LISTED = 8;

function Counts({ parts }: { parts: Segment[] }) {
  const shown = parts.filter((part) => part.count > 0);
  if (shown.length > MAX_LISTED) {
    const counts = shown.map((part) => part.count);
    const low = Math.min(...counts);
    const high = Math.max(...counts);
    return (
      <span>
        {shown.length} classes · {low === high ? low : `${low}–${high}`} samples each
      </span>
    );
  }
  return (
    <CountRun
      counts={parts.map((part): [string, number, Tone] => [part.label, part.count, part.tone])}
    />
  );
}

export function SplitComposition({
  composition,
  mode,
  classes = [],
  normalsOnlyTrain = false,
  rowAction,
  className,
}: {
  composition: readonly SubsetComposition[];
  mode: CompositionMode;
  classes?: readonly ClassInfo[];
  /** Mark a train subset with no defect, the promise a split for anomaly detection makes. */
  normalsOnlyTrain?: boolean;
  /** A trailing cell per subset, such as a link into it. */
  rowAction?: (row: SubsetComposition) => ReactNode;
  className?: string;
}) {
  const filled = composition.filter((row) => row.total > 0);
  const summary = filled.map((row) => `${row.subset} ${row.total}`).join(", ");
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div
        role="img"
        aria-label={`Composition: ${summary || "empty"}`}
        className="flex h-2 w-full gap-0.5 overflow-hidden rounded-full bg-raised"
      >
        {filled.map((row) => {
          const parts = segments(row, mode, classes).filter((part) => part.count > 0);
          return (
            <div
              key={row.subset}
              className="flex h-full min-w-1 overflow-hidden"
              style={{ flexGrow: row.total, flexBasis: 0 }}
            >
              {parts.length === 0 ? (
                <div className="h-full w-full bg-line-strong" />
              ) : (
                parts.map((part) => (
                  <div
                    key={part.label}
                    className={cn("h-full", part.className)}
                    style={{ ...part.style, flexGrow: part.count, flexBasis: 0 }}
                  />
                ))
              )}
            </div>
          );
        })}
      </div>
      <dl
        className={cn(
          "grid items-baseline gap-x-3 gap-y-0.5 text-xs",
          rowAction ? "grid-cols-[auto_auto_1fr_auto]" : "grid-cols-[auto_auto_1fr]",
        )}
      >
        {composition.map((row) => (
          <div key={row.subset} className="contents">
            <dt className="text-fg-muted">{row.subset}</dt>
            <dd className="text-right font-mono tabular-nums text-fg">{row.total}</dd>
            <dd className="flex min-w-0 flex-wrap items-baseline gap-x-2 text-fg-muted">
              {row.total === 0 ? (
                <span className="text-fg-subtle">empty</span>
              ) : (
                <>
                  <Counts parts={segments(row, mode, classes)} />
                  {/* A defect here would teach the model that defects are normal. */}
                  {normalsOnlyTrain && row.subset === "train" && row.defect === 0 && (
                    <span className="text-normal">no defect ✓</span>
                  )}
                </>
              )}
            </dd>
            {rowAction && <dd className="text-right">{rowAction(row)}</dd>}
          </div>
        ))}
      </dl>
    </div>
  );
}
