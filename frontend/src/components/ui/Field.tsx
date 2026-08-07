/**
 * A labelled control.
 *
 * Two things the old version could not express, both of which the schema form needed and
 * faked. A required field had `" (required)"` concatenated onto its label string, so the
 * marker was part of the text and could not be styled, read out as a marker, or dropped.
 * And a field's description was rendered by the *caller*, below the whole thing, so nothing
 * connected the two for a screen reader.
 *
 * `as="group"` exists because a `<label>` cannot label a set of radios or a segmented
 * control; that case needs a labelled group instead, and getting it wrong is the difference
 * between a control that announces itself and one that announces nothing.
 */

import { useId, type ReactNode } from "react";

import { cn } from "./cn";

export function Field({
  label,
  description,
  error,
  required = false,
  /** A compact fact about the accepted values -- a range, a unit -- kept out of the prose. */
  annotation,
  as = "label",
  className,
  children,
}: {
  label: string;
  description?: ReactNode;
  error?: ReactNode;
  required?: boolean;
  annotation?: ReactNode;
  as?: "label" | "group";
  className?: string;
  children: ReactNode;
}) {
  const describedBy = useId();
  const Wrapper = as === "label" ? "label" : "div";

  return (
    <Wrapper
      className={cn("flex min-w-0 flex-col gap-1.5", className)}
      {...(as === "group" ? { role: "group", "aria-label": label } : {})}
    >
      <span className="flex items-baseline gap-2">
        <span className="text-xs font-medium text-fg">
          {label}
          {required && (
            <span className="ml-0.5 text-defect" title="Required">
              *
            </span>
          )}
        </span>
        {annotation && (
          <span className="ml-auto font-mono text-[11px] text-fg-subtle">{annotation}</span>
        )}
      </span>

      {children}

      {description && (
        <span id={describedBy} className="text-xs leading-snug text-fg-muted">
          {description}
        </span>
      )}
      {error && (
        <span role="alert" className="text-xs leading-snug text-defect">
          {error}
        </span>
      )}
    </Wrapper>
  );
}
