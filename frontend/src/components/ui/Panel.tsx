/**
 * The page furniture: what holds a screen together above the level of a single control.
 *
 * Every route used to open with its own hand-written `<h2 className="text-lg …">` inside its
 * own wrapper, and three of them used `space-y-*` where the rest used `flex flex-col gap-*`.
 * That is one heading size and one gap rule, restated eleven times and agreeing by luck.
 */

import type { ReactNode } from "react";
import { Link } from "react-router";

import { cn } from "./cn";

export function Panel({
  title,
  actions,
  className,
  bodyClassName,
  children,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  bodyClassName?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("rounded-panel border border-line bg-surface", className)}>
      {(title || actions) && (
        <header className="flex min-h-11 items-center justify-between gap-4 border-b border-line px-4 py-2.5">
          <h2 className="text-sm font-semibold tracking-tight text-fg">{title}</h2>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

/**
 * A numbered step inside a form.
 *
 * The number is not decoration: it marks a genuine ordering, where each choice narrows the
 * next. Do not reach for this where the parts are merely adjacent.
 */
export function Section({
  step,
  title,
  hint,
  actions,
  children,
}: {
  step?: number;
  title: string;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline gap-2.5">
        {step !== undefined && (
          <span
            className="font-mono text-xs text-fg-subtle tabular-nums"
            aria-hidden
          >
            {String(step).padStart(2, "0")}
          </span>
        )}
        <h3 className="text-sm font-semibold tracking-tight text-fg">{title}</h3>
        {hint && <p className="text-xs text-fg-muted">{hint}</p>}
        {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

export function PageHeader({
  title,
  meta,
  actions,
  back,
}: {
  title: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  back?: { to: string; label: string };
}) {
  return (
    <header className="flex flex-col gap-1.5">
      {back && (
        <Link
          to={back.to}
          className="w-fit text-xs text-fg-muted transition-colors hover:text-signal"
        >
          ← {back.label}
        </Link>
      )}
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {meta && <div className="flex flex-wrap items-center gap-2 text-xs text-fg-muted">{meta}</div>}
    </header>
  );
}

export type ReadoutItem = {
  /** A quiet prefix naming what the value is, where the value alone is ambiguous. */
  label?: string;
  value: ReactNode;
  to?: string;
};

/**
 * The instrument's display line: the facts about what is on screen, in one fixed slot.
 *
 * The application has a real hierarchy -- dataset, split, experiment, sample -- and no
 * breadcrumbs, so until now the only way to tell which dataset you were inside was to
 * recognise the name in the heading. This states it in the same place on every screen, in
 * mono, the way a piece of measuring equipment has one readout rather than a label per
 * panel. Rendered as a list because it is one.
 */
export function ReadoutStrip({
  items,
  className,
}: {
  items: ReadoutItem[];
  className?: string;
}) {
  const shown = items.filter((item) => item.value !== null && item.value !== undefined);
  if (shown.length === 0) return null;

  return (
    <ol
      className={cn(
        "flex flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-fg-muted",
        className,
      )}
    >
      {shown.map((item, index) => (
        <li key={index} className="flex items-center gap-2">
          {index > 0 && (
            <span className="text-fg-subtle" aria-hidden>
              ·
            </span>
          )}
          {item.label && <span className="text-fg-subtle">{item.label}</span>}
          {item.to ? (
            <Link to={item.to} className="text-fg transition-colors hover:text-signal">
              {item.value}
            </Link>
          ) : (
            <span className="text-fg">{item.value}</span>
          )}
        </li>
      ))}
    </ol>
  );
}
