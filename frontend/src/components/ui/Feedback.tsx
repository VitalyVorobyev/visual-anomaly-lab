/**
 * What the screen says when it has something other than data to show.
 *
 * An error explains what happened and what to do about it; it does not apologise and it is
 * never vague. An empty state is an invitation to act, so it carries the action rather than
 * describing it. And loading is a shape, not the word "Loading…" -- which is what eight
 * screens rendered, so a slow request looked identical to a broken one.
 */

import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "./cn";

type CalloutTone = "info" | "warning" | "error" | "success";

const CALLOUT: Record<CalloutTone, { classes: string; Icon: typeof Info }> = {
  info: { classes: "border-signal/30 bg-signal/8 text-fg", Icon: Info },
  warning: { classes: "border-warn/30 bg-warn/8 text-fg", Icon: AlertTriangle },
  error: { classes: "border-defect/30 bg-defect/8 text-fg", Icon: XCircle },
  success: { classes: "border-normal/30 bg-normal/8 text-fg", Icon: CheckCircle2 },
};

const ICON_TONE: Record<CalloutTone, string> = {
  info: "text-signal",
  warning: "text-warn",
  error: "text-defect",
  success: "text-normal",
};

export function Callout({
  tone = "info",
  title,
  actions,
  className,
  children,
}: {
  tone?: CalloutTone;
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  const { classes, Icon } = CALLOUT[tone];
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "flex items-start gap-2.5 rounded-control border px-3 py-2.5 text-sm leading-snug",
        classes,
        className,
      )}
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", ICON_TONE[tone])} aria-hidden />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        {title && <p className="font-medium">{title}</p>}
        <div className="min-w-0 text-fg-muted">{children}</div>
        {actions && <div className="mt-1 flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}

/** The error shape the screens already used, now on the shared callout. */
export function ErrorBox({ children }: { children: ReactNode }) {
  return <Callout tone="error">{children}</Callout>;
}

export function Empty({ action, children }: { action?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 text-center">
      <p className="text-sm text-fg-muted">{children}</p>
      {action}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn("animate-pulse rounded-control bg-line", className)}
    />
  );
}

/** A stand-in for a panel's worth of content, so the layout does not jump when it lands. */
export function SkeletonRows({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <div role="status" aria-label="Loading" className={cn("flex flex-col gap-2", className)}>
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} className={cn("h-4", index === rows - 1 ? "w-2/3" : "w-full")} />
      ))}
    </div>
  );
}

export function ProgressBar({ fraction, label }: { fraction: number; label?: string }) {
  const percent = Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
  return (
    <div className="flex flex-col gap-1.5">
      <div
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-1.5 overflow-hidden rounded-full bg-line"
      >
        <div
          className="h-full rounded-full bg-signal transition-[width] duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      {label && (
        <span className="font-mono text-xs text-fg-muted tabular-nums">
          {percent}% · {label}
        </span>
      )}
    </div>
  );
}
