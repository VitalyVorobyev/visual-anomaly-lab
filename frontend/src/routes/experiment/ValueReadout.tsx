/**
 * What the numbers are, at the pixel under the cursor.
 *
 * You asked for the colour under the cursor. In a measurement tool the useful readout is
 * the model's own number, not the CSS colour a colormap chose for it — so this reports the
 * anomaly value in map units with its position in the run-wide range, and the preprocessed
 * model-input values the method actually read. They are projected back through the pinned
 * transform so the pointer stays in source coordinates; outside the prepared crop there
 * is deliberately no value. For a multi-channel dataset every plane remains available.
 *
 * Both come from float32 planes fetched once per image and indexed locally (ADR-0023). No
 * colour is ever inverted: the colormap clips and quantizes, so the inverse is multi-valued
 * and would report a *display* quantity rather than a measurement.
 */

import type { MapScale } from "../../api/client";
import { fractionOf, valueAt, valuesAt, type ValuePlane } from "@vitavision/lab-ui";

export interface HoverPosition {
  u: number;
  v: number;
}

export function ValueReadout({
  position,
  map,
  source,
  range,
}: {
  position: HoverPosition | null;
  map: ValuePlane | undefined;
  /** Every colour plane the experiment preprocesses to, in one payload. */
  source: ValuePlane | undefined;
  range: MapScale | null | undefined;
}) {
  // A fixed-height line whether or not the pointer is over the canvas: a readout that
  // appears and disappears reflows everything under it on every mouse movement.
  if (position === null) {
    return (
      <span className="font-mono text-xs text-fg-subtle">
        hover the image for the values under the cursor
      </span>
    );
  }

  const anomaly = map ? valueAt(map, position.u, position.v) : null;
  const fraction = anomaly === null ? null : fractionOf(anomaly, range);
  const channels = source ? valuesAt(source, position.u, position.v) : [];
  const uncovered = source !== undefined && channels.length === 0;

  // A plane the server decimated is an approximate readout, and saying so is the whole
  // reason the stride rides in the header.
  const sampled = [map, source].some((plane) => plane !== undefined && plane.stride > 1);

  return (
    <span className="flex flex-wrap items-baseline gap-x-4 font-mono text-xs text-fg-muted">
      {anomaly !== null && (
        <span>
          map <span className="text-fg tabular-nums">{format(anomaly)}</span>
          {fraction !== null && (
            <span className="text-fg-subtle"> · {(fraction * 100).toFixed(0)}% of run range</span>
          )}
        </span>
      )}
      {channels.length > 0 && (
        <span>
          input{" "}
          <span className="text-fg tabular-nums">
            {channels.map((value) => value.toFixed(3)).join(", ")}
          </span>
        </span>
      )}
      {uncovered && <span className="text-fg-subtle">outside prepared region</span>}
      {!uncovered && anomaly === null && channels.length === 0 && (
        <span className="text-fg-subtle">no values recorded for this image</span>
      )}
      {sampled && <span className="text-fg-subtle">sampled</span>}
    </span>
  );
}

/** Enough digits for a small anomaly value without claiming precision it lacks. */
function format(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}
