/**
 * Content folded away, with a caret we draw ourselves.
 *
 * Seven `<details>` elements across the app each carried the same hand-written summary
 * class string, and all seven showed the UA triangle in the OS palette, at whatever size
 * the platform picked. The base layer removes the native marker; this supplies one that
 * rotates with the element's own open state, so no JavaScript is involved in the animation.
 *
 * `<details>` rather than a headless package on purpose: the element already has the
 * semantics, the keyboard behaviour and the find-in-page integration.
 */

import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "./cn";

export function Disclosure({
  summary,
  count,
  defaultOpen = false,
  className,
  children,
}: {
  summary: ReactNode;
  /** How many things are inside, where the number is the reason to open it. */
  count?: number;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details open={defaultOpen} className={cn("group", className)}>
      <summary
        className={cn(
          "flex cursor-pointer items-center gap-1.5 rounded-control py-1 text-xs font-medium text-fg-muted",
          "transition-colors hover:text-fg",
          "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal",
        )}
      >
        <ChevronRight
          className="size-3.5 shrink-0 transition-transform group-open:rotate-90"
          aria-hidden
        />
        {summary}
        {count !== undefined && (
          <span className="font-mono text-fg-subtle tabular-nums">{count}</span>
        )}
      </summary>
      <div className="pt-3">{children}</div>
    </details>
  );
}
