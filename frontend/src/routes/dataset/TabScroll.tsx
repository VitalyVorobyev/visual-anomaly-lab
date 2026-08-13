/**
 * The one scroll region a dataset tab is allowed to own.
 *
 * `DatasetLayout` above it is `overflow-hidden` and hands down whatever height the band
 * left; this is where that height is spent. The scroller is full width and the measure is
 * a centred box inside it -- the same two-box shape as `ReadingLayout` -- so the scrollbar
 * sits at the window edge on every tab and does not move when the measure does.
 *
 * `scrollbar-gutter: stable` reserves the gutter whether or not the content overflows.
 * Without it, moving from a tab that overflows to one that does not reclaims the
 * scrollbar's width and nudges the whole column sideways, which is half of what the tab
 * strip was doing.
 *
 * A tab whose content is full-bleed -- the browser's virtual grid -- does not use this and
 * declares its own scroller carrying the same `data-scroll="tab"` marker. There is always
 * exactly one per tab, and nothing inside it may scroll on the same axis.
 */

import type { ReactNode } from "react";

import { cn } from "../../components/ui";

export function TabScroll({
  measure = "read",
  className,
  children,
}: {
  /** `read` caps at a readable 72rem; `wide` gives a data surface the workspace's 100rem. */
  measure?: "read" | "wide";
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      data-scroll="tab"
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]"
    >
      <div
        className={cn(
          "mx-auto w-full px-5 py-5",
          measure === "read" ? "max-w-6xl" : "max-w-[100rem]",
          className,
        )}
      >
        {children}
      </div>
    </div>
  );
}
