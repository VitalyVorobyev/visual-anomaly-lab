/**
 * Text entry.
 *
 * `inputClasses` used to be an exported *string*, pasted onto every native `<input>`,
 * `<select>` and `<textarea>` in the application -- which is why a select and a text box
 * were visually indistinguishable, and why nothing could be varied without a find-and-
 * replace. It is kept as an alias below only so screens still on the old idiom pick up the
 * new palette before they are rewritten.
 */

import type { InputHTMLAttributes, TextareaHTMLAttributes } from "react";

import { cn, focusRingInset } from "./cn";

export const controlClasses = cn(
  "w-full rounded-control border border-line-strong bg-raised px-2.5 text-sm text-fg",
  "placeholder:text-fg-subtle",
  "transition-colors hover:border-fg-subtle",
  "disabled:cursor-not-allowed disabled:opacity-50",
  focusRingInset,
);

/** @deprecated Use `Input`, `NumberInput`, `Textarea` or `Select`. */
export const inputClasses = controlClasses;

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...rest} className={cn(controlClasses, "h-8", className)} />;
}

/**
 * A number, with the schema's own bounds attached.
 *
 * The bounds were in the JSON Schema all along -- `max_steps` is 10..200000 -- and were
 * dropped on the way to the control, so the form accepted values the backend would reject
 * and reported it as a 422 half a screen away from the field that caused it.
 *
 * `font-mono` because these are read as quantities and compared down a column.
 */
export function NumberInput({
  className,
  ...rest
}: InputHTMLAttributes<HTMLInputElement> & { min?: number; max?: number }) {
  return (
    <input
      {...rest}
      type="number"
      inputMode="decimal"
      className={cn(controlClasses, "h-8 font-mono tabular-nums", className)}
    />
  );
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea {...rest} className={cn(controlClasses, "py-1.5 font-mono", className)} />;
}
