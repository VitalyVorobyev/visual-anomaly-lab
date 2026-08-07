/**
 * One renderer per `kind`, and a placeholder for a kind that does not exist yet.
 *
 * The placeholder is not defensive clutter: ADR-0018 says the index is self-describing and
 * the UI renders by kind, which means a future plugin can legitimately write a kind this
 * build has never heard of. Crashing the tab would make the contract a lie; saying so
 * plainly keeps everything else on the screen.
 */

import { useState } from "react";

import type { DiagnosticEntry } from "../../api/client";
import { diagnosticPayloadUrl, gridFrameCount } from "../../api/diagnostics";
import { Empty } from "../ui";

/** A `map` or an `image`: one server-rendered PNG, on a black field so it reads as data. */
export function DiagnosticImage({
  experimentId,
  entry,
  frame = 0,
  caption,
}: {
  experimentId: number;
  entry: DiagnosticEntry;
  frame?: number;
  caption?: string;
}) {
  return (
    <div className="overflow-hidden rounded border border-line bg-[#08090a] ">
      <img
        src={diagnosticPayloadUrl(experimentId, entry, frame)}
        alt={caption ?? entry.title}
        className="block w-full"
        loading="lazy"
      />
    </div>
  );
}

/**
 * A `grid`: N small multiples, one request per cell.
 *
 * Capped on first paint, because a grid can hold as many frames as a model felt like
 * emitting and each is a request. The cap is a disclosure with a button, not a silent
 * truncation.
 */
export function DiagnosticGrid({
  experimentId,
  entry,
  initial = 16,
}: {
  experimentId: number;
  entry: DiagnosticEntry;
  initial?: number;
}) {
  const total = gridFrameCount(entry);
  const [shown, setShown] = useState(Math.min(initial, total));

  if (total === 0) return <Empty>This grid recorded no cells.</Empty>;

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-4 gap-1.5 sm:grid-cols-6 lg:grid-cols-8">
        {Array.from({ length: shown }, (_, frame) => (
          <figure key={frame} className="flex flex-col gap-0.5">
            <DiagnosticImage
              experimentId={experimentId}
              entry={entry}
              frame={frame}
              caption={`${entry.title}, channel ${frame}`}
            />
            <figcaption className="text-center font-mono text-[10px] text-fg-subtle">
              {frame}
            </figcaption>
          </figure>
        ))}
      </div>
      {shown < total && (
        <button
          type="button"
          onClick={() => setShown(total)}
          className="self-start text-xs text-fg-muted hover:underline"
        >
          Showing {shown} of {total} — show all
        </button>
      )}
    </div>
  );
}

/** A `table`: whatever columns and rows the model declared. */
export function DiagnosticTable({ payload }: { payload: Record<string, unknown> }) {
  const columns = Array.isArray(payload.columns) ? (payload.columns as unknown[]) : [];
  const rows = Array.isArray(payload.rows) ? (payload.rows as unknown[][]) : [];

  if (rows.length === 0) return <Empty>This table recorded no rows.</Empty>;

  return (
    <div className="overflow-x-auto">
      <table className="text-sm">
        {columns.length > 0 && (
          <thead>
            <tr className="text-xs text-fg-muted">
              {columns.map((column, index) => (
                <th key={index} className="px-2 py-1 text-left font-medium">
                  {String(column)}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody className="font-mono">
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-t border-line">
              {(Array.isArray(row) ? row : [row]).map((cell, cellIndex) => (
                <td key={cellIndex} className="px-2 py-1">
                  {String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A kind this build does not know how to draw. Named, not swallowed. */
export function UnknownKind({ entry }: { entry: DiagnosticEntry }) {
  return (
    <p className="rounded border border-dashed border-line-strong px-3 py-2 text-xs text-fg-muted ">
      This run recorded a diagnostic of kind <span className="font-mono">{entry.kind}</span>,
      which this version of the workbench cannot draw. The data is on disk under the
      experiment's <span className="font-mono">diagnostics/</span> directory.
    </p>
  );
}
