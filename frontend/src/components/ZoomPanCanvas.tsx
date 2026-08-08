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
 * gesture that gets you unlost — and zoom 1 *is* fit-to-window, because the canvas is laid
 * out inside `max-h-full max-w-full` with the source aspect ratio. That is why the
 * double-click gesture toggles fit against **1:1 native pixels** rather than "fitting":
 * fitting is where you already are, and the thing a reader actually wants at a defect is
 * the sensor's own resolution with no resampling between them and it.
 */

import { useEffect, useRef, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

import { cn } from "./ui";

export interface View {
  zoom: number;
  x: number;
  y: number;
}

/** Zoom 1, centred — which is fit-to-window, given how the canvas is laid out. */
export const RESET_VIEW: View = { zoom: 1, x: 0, y: 0 };

export const MIN_ZOOM = 1;
export const MAX_ZOOM = 12;
const ZOOM_SENSITIVITY = 0.0015;

/** The zoom past which a preview tier stops being enough and real pixels are wanted. */
export const FULL_TIER_ZOOM = 2;

/**
 * Zoom by `factor` about a point, keeping whatever is under that point where it is.
 *
 * Pure, and exported, because it is the one piece of arithmetic here that can be silently
 * wrong: a centre-anchored zoom looks correct until someone tries to magnify a defect in a
 * corner, at which point every wheel notch has to be undone with a drag.
 *
 * The transform is `translate(t) scale(z)` about the box's centre, so a content point `c`
 * lands at `p = c·z + t`. Holding `c` fixed while `z` becomes `z'` gives
 * `t' = p − (p − t)·z'/z`, with `p` measured from the centre.
 */
export function zoomAt(
  view: View,
  pointer: { x: number; y: number },
  rect: { left: number; top: number; width: number; height: number },
  factor: number,
): View {
  const zoom = clamp(view.zoom * factor, MIN_ZOOM, MAX_ZOOM);
  // Bottoming out is a reset: at 1x the content fills the frame, so any offset is a way to
  // be lost with no visible handle to get back.
  if (zoom === MIN_ZOOM) return RESET_VIEW;

  const px = pointer.x - rect.left - rect.width / 2;
  const py = pointer.y - rect.top - rect.height / 2;
  const ratio = zoom / view.zoom;
  return {
    zoom,
    x: px - (px - view.x) * ratio,
    y: py - (py - view.y) * ratio,
  };
}

/**
 * The zoom at which one source pixel covers one CSS pixel.
 *
 * Clamped, so an image smaller than its frame reports 1: there are no further pixels to
 * reveal, and pretending otherwise would magnify the resampler rather than the sensor.
 */
export function nativeZoomFor(sourceWidth: number, boxWidth: number): number {
  if (!(sourceWidth > 0) || !(boxWidth > 0)) return FULL_TIER_ZOOM;
  return clamp(sourceWidth / boxWidth, MIN_ZOOM, MAX_ZOOM);
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

export function ZoomPanCanvas({
  view,
  onView,
  children,
  className,
  style,
  label,
  nativeWidth,
  fitLabel = "Fit",
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
  /**
   * The source image's pixel width, so double-click can reach true 1:1. Without it the
   * gesture falls back to `FULL_TIER_ZOOM`, which is at least the tier boundary.
   */
  nativeWidth?: number;
  /** Hidden by passing `null` — the dataset browser has its own button beside the canvas. */
  fitLabel?: ReactNode;
}) {
  const dragging = useRef<{ x: number; y: number } | null>(null);
  // Mirrored into state only so the cursor can change — the ref is what the move handler
  // reads, because a re-render per pointermove would be a cost for nothing.
  const [grabbing, setGrabbing] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const latest = useRef(view);
  latest.current = view;

  /*
   * The wheel listener is attached by hand because React registers `wheel` at the root as
   * **passive**, which makes `preventDefault` in a synthetic handler a no-op. Without it
   * the page scrolls behind the canvas while you zoom, so a gesture aimed at the image
   * moves the document instead — which reads as the zoom being broken.
   */
  useEffect(() => {
    const element = box.current;
    if (!element) return;

    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      const factor = 1 - event.deltaY * ZOOM_SENSITIVITY;
      onView(zoomAt(latest.current, { x: event.clientX, y: event.clientY }, rect, factor));
    };

    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [onView]);

  const toggleNative = (event: { clientX: number; clientY: number }) => {
    const element = box.current;
    if (!element) return;
    const rect = element.getBoundingClientRect();
    const native = nativeZoomFor(nativeWidth ?? 0, rect.width);
    // Anywhere but fit goes back to fit; fit goes to 1:1 under the cursor.
    const target = view.zoom > MIN_ZOOM ? MIN_ZOOM : native;
    onView(zoomAt(view, { x: event.clientX, y: event.clientY }, rect, target / view.zoom));
  };

  const endDrag = () => {
    dragging.current = null;
    setGrabbing(false);
  };

  return (
    <div
      ref={box}
      style={style}
      className={cn(
        "relative overflow-hidden rounded border border-line bg-[#08090a] select-none",
        grabbing ? "cursor-grabbing" : "cursor-grab",
        className,
      )}
      onPointerDown={(event) => {
        dragging.current = { x: event.clientX - view.x, y: event.clientY - view.y };
        setGrabbing(true);
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const origin = dragging.current;
        if (!origin) return;
        onView({ ...view, x: event.clientX - origin.x, y: event.clientY - origin.y });
      }}
      onPointerUp={endDrag}
      // A drag interrupted by a lost capture — a context menu, a browser gesture, the
      // pointer leaving the window — used to leave `dragging` set, so the next hover
      // panned the image without a button held down.
      onPointerCancel={endDrag}
      onLostPointerCapture={endDrag}
      onDoubleClick={toggleNative}
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
      {fitLabel !== null && view.zoom > MIN_ZOOM && (
        <button
          type="button"
          className="absolute right-1 bottom-1 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] text-white hover:bg-black/80"
          // Stopping `pointerdown` is what makes the button work at all: without it the
          // press bubbles to the canvas, which calls `setPointerCapture` on itself and
          // takes the click with it, so the button rendered and did nothing.
          onPointerDown={(event) => event.stopPropagation()}
          onClick={(event) => {
            event.stopPropagation();
            onView(RESET_VIEW);
          }}
          // The canvas swallows double-clicks to toggle 1:1; without this the second click
          // of a fast double-tap on the button reaches the canvas and zooms straight back.
          onDoubleClick={(event) => event.stopPropagation()}
        >
          {fitLabel}
        </button>
      )}
    </div>
  );
}
