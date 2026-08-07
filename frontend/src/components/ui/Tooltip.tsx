/**
 * A short explanation, on demand.
 *
 * Replaces the native `title` attribute, which appears after an unpredictable delay, in the
 * OS palette, at a size nobody chose, and never at all for a keyboard user.
 *
 * A tooltip is for text that helps but is not needed to operate the control. Anything a
 * reader must have in order to answer the question belongs in the field's description,
 * where it is always visible.
 */

import * as RadixTooltip from "@radix-ui/react-tooltip";
import { HelpCircle } from "lucide-react";
import type { ReactNode } from "react";

import { cn, focusRing } from "./cn";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <RadixTooltip.Provider delayDuration={200}>{children}</RadixTooltip.Provider>;
}

export function Tooltip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RadixTooltip.Root>
      <RadixTooltip.Trigger asChild>{children}</RadixTooltip.Trigger>
      <RadixTooltip.Portal>
        <RadixTooltip.Content
          sideOffset={6}
          collisionPadding={8}
          className={cn(
            "z-50 max-w-xs rounded-control border border-line bg-overlay px-2.5 py-1.5",
            "text-xs leading-snug text-fg shadow-lg shadow-black/25",
          )}
        >
          {content}
          <RadixTooltip.Arrow className="fill-overlay" />
        </RadixTooltip.Content>
      </RadixTooltip.Portal>
    </RadixTooltip.Root>
  );
}

/** The standard affordance: a quiet mark that yields the explanation on hover or focus. */
export function InfoHint({ children, label = "More information" }: { children: ReactNode; label?: string }) {
  return (
    <Tooltip content={children}>
      <button
        type="button"
        aria-label={label}
        className={cn("rounded-full text-fg-subtle transition-colors hover:text-fg", focusRing)}
      >
        <HelpCircle className="size-3.5" />
      </button>
    </Tooltip>
  );
}
