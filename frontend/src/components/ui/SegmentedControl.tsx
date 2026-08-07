/**
 * A choice among two or three, shown as two or three things.
 *
 * This is the control that was missing when `model_size` -- a `Literal["small", "medium"]`
 * -- rendered as a free text box with `small` as its placeholder. Every option visible at
 * once, no popup, no typing, and no way to enter a value the backend will reject.
 *
 * Built on native radios rather than a headless package: a radio group in one `name` gets
 * roving focus and arrow-key navigation from the platform, which is the entire reason to
 * reach for a library here.
 *
 * **`""` means unset**, and this control keeps that true without showing it. The segment
 * highlighted is always the *effective* value, so the reader sees what will happen;
 * choosing the segment that matches the schema default stores `""` rather than the value,
 * so the default is still sent by nobody and defined only in Python. See
 * `api/schemaForm.ts` for why that rule is worth this much care.
 */

import { useId } from "react";

import { cn } from "./cn";

export function SegmentedControl({
  value,
  defaultValue,
  options,
  onValueChange,
  disabled = false,
  "aria-label": ariaLabel,
  className,
}: {
  /** `""` when the field is untouched. */
  value: string;
  /** What the backend applies when nothing is sent. Highlighted while `value` is `""`. */
  defaultValue?: string;
  options: { value: string; label: string }[];
  onValueChange: (value: string) => void;
  disabled?: boolean;
  "aria-label"?: string;
  className?: string;
}) {
  const name = useId();
  const effective = value === "" ? defaultValue : value;

  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex w-fit max-w-full items-center gap-0.5 rounded-control border border-line-strong bg-raised p-0.5",
        disabled && "opacity-50",
        className,
      )}
    >
      {options.map((option) => {
        const checked = option.value === effective;
        return (
          <label
            key={option.value}
            className={cn(
              "relative cursor-pointer rounded-[0.3rem] px-2.5 py-1 text-xs font-medium transition-colors",
              "has-focus-visible:outline-2 has-focus-visible:outline-offset-1 has-focus-visible:outline-signal",
              checked
                ? "bg-surface text-fg shadow-sm"
                : "text-fg-muted hover:text-fg",
              disabled && "cursor-not-allowed",
            )}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={checked}
              disabled={disabled}
              onChange={() =>
                // Selecting the default returns the field to unset: same effective value,
                // nothing sent, and the default keeps living in exactly one place.
                onValueChange(option.value === defaultValue ? "" : option.value)
              }
              className="absolute inset-0 cursor-pointer opacity-0"
            />
            {option.label}
          </label>
        );
      })}
    </div>
  );
}
