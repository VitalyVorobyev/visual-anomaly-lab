/**
 * A table.
 *
 * Five routes each hand-rolled `<table className="w-full text-left text-sm">` over a
 * copy-pasted `<thead>` and `border-t border-slate-100` rows, which is five chances for the
 * header treatment to drift apart -- and it had.
 *
 * `numeric` is the column property that matters here. Quantities are set in mono with
 * tabular figures and aligned right, so a column of scores can be compared by eye down its
 * decimal point rather than read one row at a time.
 */

import type { ReactNode } from "react";

import { cn } from "./cn";

export type Column<Row> = {
  key: string;
  header: ReactNode;
  /** Right-aligned, mono, tabular. Use for anything that is a quantity. */
  numeric?: boolean;
  width?: string;
  cell: (row: Row) => ReactNode;
};

export function Table<Row>({
  columns,
  rows,
  rowKey,
  empty = "Nothing here.",
  caption,
  className,
}: {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row, index: number) => string | number;
  empty?: ReactNode;
  caption?: string;
  className?: string;
}) {
  if (rows.length === 0) {
    return <p className="py-6 text-center text-sm text-fg-muted">{empty}</p>;
  }

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      <table className="w-full text-left text-sm">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line">
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                style={column.width ? { width: column.width } : undefined}
                className={cn(
                  "pb-2 text-xs font-medium text-fg-muted",
                  column.numeric && "text-right",
                )}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey(row, index)} className="border-b border-line/60 last:border-0">
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={cn(
                    "py-2 pr-3 align-middle text-fg last:pr-0",
                    column.numeric && "text-right font-mono tabular-nums",
                  )}
                >
                  {column.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
