/**
 * The two binary controls, and the difference between them.
 *
 * A `Switch` changes how the thing behaves from now on -- "allow downloads", "show the
 * mask" -- and reads as a setting. A `Checkbox` asserts something about a state you are
 * about to commit -- "I have read the warnings", "keep this channel" -- and reads as a
 * statement. Both were bare browser checkboxes before, styled by nobody, so the two jobs
 * were indistinguishable on screen.
 */

import * as RadixCheckbox from "@radix-ui/react-checkbox";
import * as RadixSwitch from "@radix-ui/react-switch";
import { Check, Minus } from "lucide-react";
import type { ReactNode } from "react";

import { cn, focusRing } from "./cn";

export function Switch({
  checked,
  onCheckedChange,
  label,
  description,
  disabled = false,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className="flex min-w-0 items-start gap-2.5">
      <RadixSwitch.Root
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
        className={cn(
          "mt-0.5 h-4 w-7 shrink-0 rounded-full border border-line-strong bg-raised transition-colors",
          "data-[state=checked]:border-signal data-[state=checked]:bg-signal",
          "disabled:cursor-not-allowed disabled:opacity-50",
          focusRing,
        )}
      >
        <RadixSwitch.Thumb
          className={cn(
            "block size-3 rounded-full bg-fg-subtle transition-transform will-change-transform",
            "translate-x-0.5 data-[state=checked]:translate-x-3.5 data-[state=checked]:bg-signal-fg",
          )}
        />
      </RadixSwitch.Root>
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm leading-tight text-fg">{label}</span>
        {description && (
          <span className="text-xs leading-snug text-fg-muted">{description}</span>
        )}
      </span>
    </label>
  );
}

export function Checkbox({
  checked,
  onCheckedChange,
  label,
  description,
  disabled = false,
  "aria-label": ariaLabel,
}: {
  /** `"indeterminate"` for a partial selection over a set. */
  checked: boolean | "indeterminate";
  onCheckedChange: (checked: boolean) => void;
  label?: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
  /** For a checkbox in a table cell, where the row is the label. */
  "aria-label"?: string;
}) {
  const box = (
    <RadixCheckbox.Root
      checked={checked}
      onCheckedChange={(next) => onCheckedChange(next === true)}
      disabled={disabled}
      aria-label={ariaLabel}
      className={cn(
        "flex size-4 shrink-0 items-center justify-center rounded-[0.25rem] border border-line-strong bg-raised transition-colors",
        "data-[state=checked]:border-signal data-[state=checked]:bg-signal",
        "data-[state=indeterminate]:border-signal data-[state=indeterminate]:bg-signal",
        "disabled:cursor-not-allowed disabled:opacity-50",
        focusRing,
      )}
    >
      <RadixCheckbox.Indicator className="text-signal-fg">
        {checked === "indeterminate" ? (
          <Minus className="size-3" strokeWidth={3} />
        ) : (
          <Check className="size-3" strokeWidth={3} />
        )}
      </RadixCheckbox.Indicator>
    </RadixCheckbox.Root>
  );

  if (label === undefined) return box;

  return (
    <label className="flex min-w-0 items-start gap-2.5">
      <span className="mt-0.5">{box}</span>
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm leading-tight text-fg">{label}</span>
        {description && (
          <span className="text-xs leading-snug text-fg-muted">{description}</span>
        )}
      </span>
    </label>
  );
}
