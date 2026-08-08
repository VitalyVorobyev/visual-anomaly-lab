/**
 * A dense on/off control for a thing that is drawn on screen.
 *
 * `Switch` is a settings row — a stacked label, a description, generous padding — and
 * three of them side by side over an image canvas is a panel, not a toolbar. This is the
 * same binary decision in the space a toolbar has: one line, one target, and an optional
 * swatch carrying the colour the layer is actually drawn in, so the legend and the control
 * are the same object rather than two things to reconcile.
 *
 * Hand-built on a native `button` with `role="switch"`, per ADR-0021: the four Radix
 * primitives are the four listed there, and a chip is not one of them.
 */

import type { ReactNode } from "react";

import { cn, focusRing } from "./cn";

export function ToggleChip({
  checked,
  onCheckedChange,
  children,
  swatch,
  disabled = false,
  title,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  children: ReactNode;
  /** A CSS colour: the colour this layer is drawn in, shown as a dot. */
  swatch?: string;
  disabled?: boolean;
  /** Why the control is disabled, or what the layer is. Never the only explanation. */
  title?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      title={title}
      onClick={() => onCheckedChange(!checked)}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-control border px-2 py-1 text-xs whitespace-nowrap transition-colors",
        checked
          ? "border-line-strong bg-raised text-fg"
          : "border-line bg-transparent text-fg-muted hover:text-fg",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-fg-muted",
        focusRing,
      )}
    >
      {swatch !== undefined && (
        <span
          aria-hidden
          className={cn(
            "size-2 shrink-0 rounded-full border transition-opacity",
            checked ? "opacity-100" : "opacity-30",
          )}
          style={{ backgroundColor: swatch, borderColor: swatch }}
        />
      )}
      {children}
    </button>
  );
}
