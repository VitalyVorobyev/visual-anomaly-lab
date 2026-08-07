/**
 * Status colour, in one place.
 *
 * The tone vocabulary is deliberately small and deliberately *about verdicts*. Under
 * ADR-0021 the chrome carries no saturation, so a colour on screen means something: green
 * is a normal sample, red is a defect, amber is a caveat. `info` is the one tone that
 * borrows the interaction accent, and it is reserved for statements of fact about a method
 * -- "produces anomaly maps" -- rather than for anything a reader must act on.
 */

import type { ReactNode } from "react";

import { cn } from "./cn";

export type Tone = "neutral" | "normal" | "defect" | "unlabeled" | "warning" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-raised text-fg-muted ring-line",
  normal: "bg-normal/12 text-normal ring-normal/25",
  defect: "bg-defect/12 text-defect ring-defect/25",
  unlabeled: "bg-raised text-fg-subtle ring-line",
  warning: "bg-warn/12 text-warn ring-warn/25",
  info: "bg-signal/12 text-signal ring-signal/25",
};

const DOT_CLASSES: Record<Tone, string> = {
  neutral: "bg-fg-subtle",
  normal: "bg-normal",
  defect: "bg-defect",
  unlabeled: "bg-line-strong",
  warning: "bg-warn",
  info: "bg-signal",
};

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-control px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * A status as a dot and a word.
 *
 * Four files had grown their own near-identical `status → tone` map. The mapping still
 * belongs to each caller -- only it knows whether "running" is good news -- but the
 * *rendering* of the answer does not.
 */
export function StatusDot({ tone, children }: { tone: Tone; children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-fg-muted">
      <span className={cn("size-1.5 shrink-0 rounded-full", DOT_CLASSES[tone])} aria-hidden />
      {children}
    </span>
  );
}

/** Counts as a compact `12 normal · 9 defect` run, skipping the zeroes. */
export function CountRun({ counts }: { counts: [string, number, Tone][] }) {
  const shown = counts.filter(([, value]) => value > 0);
  if (shown.length === 0) return <span className="text-xs text-fg-subtle">empty</span>;
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {shown.map(([name, value, tone]) => (
        <Badge key={name} tone={tone}>
          <span className="font-mono">{value}</span> {name}
        </Badge>
      ))}
    </span>
  );
}
