/** Previous / next in the queue, blocked while there is unsaved work to navigate away from. */

import { ArrowLeft, ArrowRight } from "lucide-react";

import { Button } from "@vitavision/lab-ui";
import type { QueueNavigation } from "./useQueueNavigation";

export function QueueFooter({
  queue,
  dirty,
  multiChannel,
}: {
  queue: QueueNavigation;
  dirty: boolean;
  multiChannel: boolean;
}) {
  return (
    <footer className="flex items-center justify-between border-t border-line px-3 py-2">
      <Button
        icon={<ArrowLeft />}
        disabled={!queue.hasPrevious || dirty}
        onClick={queue.openPrevious}
        aria-label="Previous image"
      />
      {/* The few keys a labelling pass leans on; `?` has the rest, from the one list. */}
      {/* One line, or none: at the rail's width the full set wrapped into two cramped lines,
          so the narrow case keeps only the pointer to the full sheet. */}
      <span className="min-w-0 truncate text-center font-mono text-[10px] text-fg-subtle">
        <span className="hidden xl:inline">
          J/K after save · C completes{multiChannel ? " · [ ] channel" : ""} ·{" "}
        </span>
        ? all keys
      </span>
      <Button
        icon={<ArrowRight />}
        disabled={!queue.hasNext || dirty}
        onClick={queue.openNext}
        aria-label="Next image"
      />
    </footer>
  );
}
