/**
 * The small set of primitives the M2 screens actually repeat.
 *
 * Extracted from real duplication rather than written up front. ADR-0012 rules out a
 * component library on the grounds that nothing yet needs one and the distinctive parts
 * of this UI are custom anyway; that stays true, and this file is the cheap alternative —
 * a status colour, a panel border and a button shape, in one place, so four screens agree
 * on them without importing a framework.
 */

import type { ReactNode } from "react";

export type Tone = "neutral" | "normal" | "defect" | "unlabeled" | "warning" | "info";

const TONE_CLASSES: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  normal: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-200",
  defect: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-200",
  unlabeled: "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
  warning: "bg-amber-100 text-amber-900 dark:bg-amber-900/40 dark:text-amber-100",
  info: "bg-blue-100 text-blue-900 dark:bg-blue-900/40 dark:text-blue-100",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium ${TONE_CLASSES[tone]}`}
    >
      {children}
    </span>
  );
}

export function Panel({ title, actions, children }: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-900">
      {(title || actions) && (
        <header className="flex items-center justify-between gap-4 border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

type ButtonVariant = "primary" | "secondary" | "danger";

const BUTTON_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-slate-900 text-white hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white",
  secondary:
    "border border-slate-300 text-slate-700 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800",
  danger: "bg-red-600 text-white hover:bg-red-500",
};

export function Button({
  variant = "secondary",
  type = "button",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      type={type}
      {...rest}
      className={`rounded px-3 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${BUTTON_CLASSES[variant]} ${rest.className ?? ""}`}
    />
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium tracking-wide text-slate-500 uppercase dark:text-slate-400">
        {label}
      </span>
      {children}
    </label>
  );
}

export const inputClasses =
  "rounded border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 " +
  "focus:border-slate-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100";

export function ErrorBox({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="rounded border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
    >
      {children}
    </p>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-8 text-center text-sm text-slate-500 dark:text-slate-400">{children}</p>;
}

export function ProgressBar({ fraction, label }: { fraction: number; label?: string }) {
  const percent = Math.round(Math.min(Math.max(fraction, 0), 1) * 100);
  return (
    <div className="flex flex-col gap-1">
      <div
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-2 overflow-hidden rounded bg-slate-200 dark:bg-slate-700"
      >
        <div
          className="h-full bg-slate-900 transition-[width] dark:bg-slate-100"
          style={{ width: `${percent}%` }}
        />
      </div>
      {label && (
        <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
          {percent}% · {label}
        </span>
      )}
    </div>
  );
}

/** Counts as a compact `12 normal · 9 defect` run, skipping the zeroes. */
export function CountRun({ counts }: { counts: [string, number, Tone][] }) {
  const shown = counts.filter(([, value]) => value > 0);
  if (shown.length === 0) return <span className="text-xs text-slate-500">empty</span>;
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {shown.map(([name, value, tone]) => (
        <Badge key={name} tone={tone}>
          {value} {name}
        </Badge>
      ))}
    </span>
  );
}
