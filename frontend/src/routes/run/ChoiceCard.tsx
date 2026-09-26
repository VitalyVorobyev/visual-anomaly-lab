/**
 * One option of a guided step, as a card you press — the method card's shape, for any choice.
 *
 * A native radio underneath, so Tab reaches the group and the arrow keys move inside it, and
 * the whole card is the hit target. What makes the option worth choosing — its pictures, its
 * composition — goes in the body, where the reader weighs it.
 */

import { cn } from "@vitavision/lab-ui";
import type { ReactNode } from "react";

export function ChoiceCard({
  name,
  value,
  selected,
  onSelect,
  title,
  badges,
  description,
  children,
  className,
}: {
  /** The radio group this card belongs to. */
  name: string;
  value: string;
  selected: boolean;
  onSelect: () => void;
  title: ReactNode;
  badges?: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <label
      className={cn(
        "relative flex cursor-pointer flex-col gap-2.5 rounded-panel border p-3.5 transition-colors",
        "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-signal",
        selected
          ? "border-signal bg-signal/5 ring-1 ring-signal"
          : "border-line bg-surface hover:border-line-strong",
        className,
      )}
    >
      <input
        type="radio"
        name={name}
        value={value}
        checked={selected}
        onChange={onSelect}
        className="absolute inset-0 cursor-pointer opacity-0"
      />
      <span className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-semibold tracking-tight text-fg">{title}</span>
        {badges && <span className="flex flex-wrap gap-1">{badges}</span>}
      </span>
      {description && <span className="text-xs leading-snug text-fg-muted">{description}</span>}
      {children}
    </label>
  );
}

/** A step's own heading and the one sentence that says what is being decided. */
export function StepIntro({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="flex flex-col gap-1">
      <h2 className="text-base font-semibold tracking-tight text-fg">{title}</h2>
      {children && <p className="text-sm text-fg-muted">{children}</p>}
    </header>
  );
}
