/**
 * A picker with a list you can actually see.
 *
 * ADR-0012 declined a component library and set the condition for revisiting: "if a
 * dialog/slider/tab set is genuinely wanted". This is the control that met it. A native
 * `<select>` cannot style its option list on macOS, so the popup arrived in the OS palette
 * in the middle of a dark instrument panel, and it cannot render a two-line option or a
 * disabled one with a reason.
 *
 * The one contract worth reading before changing this: **`""` means unset.** The schema
 * form sends nothing for a field left alone so the backend default stays the single
 * definition of it (see `api/schemaForm.ts`). Radix reserves the empty string internally --
 * an `Item` may not carry it -- so an explicit "use the default" entry travels under a
 * sentinel and is mapped back at the boundary. Callers only ever see `""`.
 */

import * as RadixSelect from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

import { cn, focusRingInset } from "./cn";

const UNSET_SENTINEL = "__unset__";

export type SelectOption = {
  value: string;
  label: string;
  /** Shown quietly after the label -- a strategy, a count, a reason it is unavailable. */
  note?: string;
  disabled?: boolean;
};

export function Select({
  value,
  onValueChange,
  options,
  placeholder = "Choose…",
  /** When given, adds a leading entry that returns the field to unset. */
  unsetLabel,
  disabled = false,
  "aria-label": ariaLabel,
  className,
}: {
  value: string;
  onValueChange: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  unsetLabel?: string;
  disabled?: boolean;
  "aria-label"?: string;
  className?: string;
}) {
  return (
    <RadixSelect.Root
      value={value === "" ? undefined : value}
      onValueChange={(next) => onValueChange(next === UNSET_SENTINEL ? "" : next)}
      disabled={disabled}
    >
      <RadixSelect.Trigger
        aria-label={ariaLabel}
        className={cn(
          "flex h-8 w-full items-center justify-between gap-2 rounded-control border border-line-strong bg-raised px-2.5 text-sm text-fg",
          "transition-colors hover:border-fg-subtle",
          "disabled:cursor-not-allowed disabled:opacity-50",
          "data-[placeholder]:text-fg-subtle",
          focusRingInset,
          className,
        )}
      >
        <RadixSelect.Value placeholder={placeholder} />
        <RadixSelect.Icon className="shrink-0 text-fg-subtle">
          <ChevronDown className="size-3.5" />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>

      <RadixSelect.Portal>
        <RadixSelect.Content
          position="popper"
          sideOffset={4}
          className={cn(
            "z-50 max-h-72 min-w-(--radix-select-trigger-width) overflow-hidden",
            "rounded-panel border border-line bg-overlay shadow-lg shadow-black/25",
          )}
        >
          <RadixSelect.Viewport className="p-1">
            {unsetLabel !== undefined && (
              <SelectItem value={UNSET_SENTINEL} label={unsetLabel} muted />
            )}
            {options.map((option) => (
              <SelectItem
                key={option.value}
                value={option.value}
                label={option.label}
                note={option.note}
                disabled={option.disabled}
              />
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  );
}

function SelectItem({
  value,
  label,
  note,
  disabled,
  muted = false,
}: {
  value: string;
  label: string;
  note?: string;
  disabled?: boolean;
  muted?: boolean;
}) {
  return (
    <RadixSelect.Item
      value={value}
      disabled={disabled}
      className={cn(
        "relative flex cursor-default items-center gap-2 rounded-control py-1.5 pr-2 pl-7 text-sm outline-none select-none",
        "data-highlighted:bg-raised data-highlighted:text-fg",
        "data-disabled:pointer-events-none data-disabled:opacity-45",
        muted ? "text-fg-muted" : "text-fg",
      )}
    >
      <RadixSelect.ItemIndicator className="absolute left-2 text-signal">
        <Check className="size-3.5" />
      </RadixSelect.ItemIndicator>
      <RadixSelect.ItemText>{label}</RadixSelect.ItemText>
      {note && <span className="ml-auto font-mono text-xs text-fg-subtle">{note}</span>}
    </RadixSelect.Item>
  );
}
