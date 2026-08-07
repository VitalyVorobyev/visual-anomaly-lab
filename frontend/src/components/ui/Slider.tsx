/**
 * A continuous value, with its number always on screen.
 *
 * Three screens drive a slider that changes what you are looking at -- the results
 * threshold, the overlay opacity, the split fractions -- and all three were a raw
 * `<input type="range">` in the browser's own colours. The threshold one in particular is a
 * measurement instrument: the number matters as much as the position, so it is rendered
 * beside the track rather than left to a separate line of prose.
 */

import * as RadixSlider from "@radix-ui/react-slider";
import type { ReactNode } from "react";

import { cn, focusRing } from "./cn";

export function Slider({
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 1,
  step = 0.01,
  /** Rendered to the right of the track. Give it a fixed-width mono span for stability. */
  readout,
  disabled = false,
  "aria-label": ariaLabel,
  className,
}: {
  value: number;
  onValueChange: (value: number) => void;
  onValueCommit?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  readout?: ReactNode;
  disabled?: boolean;
  "aria-label"?: string;
  className?: string;
}) {
  return (
    // Grows by default: a slider that does not fill the space it is given collapses to a
    // few pixels of track, which is unusable and reads as a rendering fault.
    <div className={cn("flex min-w-0 flex-1 items-center gap-3", className)}>
      <RadixSlider.Root
        value={[value]}
        onValueChange={([next]) => next !== undefined && onValueChange(next)}
        onValueCommit={([next]) => next !== undefined && onValueCommit?.(next)}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        aria-label={ariaLabel}
        className={cn(
          "relative flex h-5 min-w-0 flex-1 touch-none items-center select-none",
          disabled && "opacity-50",
        )}
      >
        <RadixSlider.Track className="relative h-1 grow overflow-hidden rounded-full bg-line">
          <RadixSlider.Range className="absolute h-full bg-signal" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          className={cn(
            "block size-3.5 rounded-full border-2 border-signal bg-surface transition-shadow",
            "hover:shadow-[0_0_0_4px] hover:shadow-signal/20",
            "disabled:cursor-not-allowed",
            focusRing,
          )}
        />
      </RadixSlider.Root>
      {readout && (
        <span className="shrink-0 font-mono text-xs text-fg tabular-nums">{readout}</span>
      )}
    </div>
  );
}
