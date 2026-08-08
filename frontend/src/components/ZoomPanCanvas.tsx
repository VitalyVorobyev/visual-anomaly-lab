/**
 * A pannable, zoomable frame that transforms **everything stacked inside it together**.
 *
 * Extracted from the dataset browser, where it wrapped exactly one image. The experiment's
 * result viewer needs the same gesture over a stack — the photograph, the anomaly map, the
 * model's segmentation, the ground-truth outline — and the layers only stay registered
 * with each other if one transform moves all of them. Transforming each layer separately
 * would drift them apart at the exact zoom level where a reader is trying to judge whether
 * the prediction lands on the defect.
 *
 * Zoom bottoms out at 1x and snaps the pan back to centre there, so there is always a
 * gesture that gets you unlost without a reset button.
 */

import { useRef } from "react";
import type { CSSProperties, ReactNode } from "react";

import { cn } from "./ui";

export interface View {
  zoom: number;
  x: number;
  y: number;
}

export const RESET_VIEW: View = { zoom: 1, x: 0, y: 0 };

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 12;
const ZOOM_SENSITIVITY = 0.0015;

/** The zoom past which a preview tier stops being enough and real pixels are wanted. */
export const FULL_TIER_ZOOM = 2;

export function ZoomPanCanvas({
  view,
  onView,
  children,
  className,
  style,
  label,
}: {
  view: View;
  onView: (view: View) => void;
  children: ReactNode;
  /** Height and any other framing; the transform and overflow are handled here. */
  className?: string;
  /** For `aspect-ratio`, which has to be a computed value rather than a utility class. */
  style?: CSSProperties;
  /** Drawn bottom-left over the image, where it cannot push the canvas down. */
  label?: ReactNode;
}) {
  const dragging = useRef<{ x: number; y: number } | null>(null);

  return (
    <div
      style={style}
      className={cn(
        "relative cursor-grab overflow-hidden rounded border border-line bg-[#08090a] select-none",
        className,
      )}
      onWheel={(event) => {
        const next = Math.min(
          MAX_ZOOM,
          Math.max(MIN_ZOOM, view.zoom * (1 - event.deltaY * ZOOM_SENSITIVITY)),
        );
        onView(next === MIN_ZOOM ? RESET_VIEW : { ...view, zoom: next });
      }}
      onPointerDown={(event) => {
        dragging.current = { x: event.clientX - view.x, y: event.clientY - view.y };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const origin = dragging.current;
        if (!origin) return;
        onView({ ...view, x: event.clientX - origin.x, y: event.clientY - origin.y });
      }}
      onPointerUp={() => {
        dragging.current = null;
      }}
    >
      {/* One transformed box holding every layer, so they cannot drift apart. */}
      <div
        className="h-full w-full"
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transformOrigin: "center",
        }}
      >
        {children}
      </div>
      {label && (
        <span className="pointer-events-none absolute bottom-1 left-1 rounded bg-black/60 px-1 font-mono text-[10px] text-white">
          {label}
        </span>
      )}
    </div>
  );
}
