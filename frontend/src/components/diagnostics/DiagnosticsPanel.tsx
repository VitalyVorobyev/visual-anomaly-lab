/**
 * The render-by-kind switch — the one place in the frontend that decides how a diagnostic
 * is drawn, and the reason M4's views cost M6 nothing (ADR-0018).
 *
 * The switch below is over `entry.kind` and nothing else. It does not read `entry.key`, it
 * does not read the experiment's `model_type`, and it must not start: a method that emits
 * the same kinds gets every view here for free, and a method that emits something new
 * gets a named placeholder rather than a broken page.
 */

import type { DiagnosticEntry } from "../../api/client";
import { ArchitectureGraph } from "./ArchitectureGraph";
import {
  DiagnosticGrid,
  DiagnosticImage,
  DiagnosticTable,
  UnknownKind,
} from "./DiagnosticViews";

export function DiagnosticBody({
  experimentId,
  entry,
}: {
  experimentId: number;
  entry: DiagnosticEntry;
}) {
  switch (entry.kind) {
    case "graph":
      return <ArchitectureGraph payload={entry.payload ?? {}} />;
    case "table":
      return <DiagnosticTable payload={entry.payload ?? {}} />;
    case "grid":
      return <DiagnosticGrid experimentId={experimentId} entry={entry} />;
    case "map":
    case "image":
      return <DiagnosticImage experimentId={experimentId} entry={entry} />;
    default:
      return <UnknownKind entry={entry} />;
  }
}

/**
 * One diagnostic with its title and the model's own description of it.
 *
 * The description is the plugin's, verbatim. A weakly-typed contract means the caption is
 * the only thing standing between a reader and a plausible, wrong interpretation of a
 * picture — ADR-0018 says so outright — so it is shown, never summarized.
 */
export function DiagnosticCard({
  experimentId,
  entry,
  wide = false,
}: {
  experimentId: number;
  entry: DiagnosticEntry;
  wide?: boolean;
}) {
  return (
    <figure className={`flex flex-col gap-2 ${wide ? "sm:col-span-2" : ""}`}>
      <figcaption className="flex flex-col gap-0.5">
        <span className="text-sm font-medium">{entry.title}</span>
        {entry.description && (
          <span className="text-xs text-fg-muted">{entry.description}</span>
        )}
      </figcaption>
      <DiagnosticBody experimentId={experimentId} entry={entry} />
    </figure>
  );
}

/**
 * A set of diagnostics laid out together.
 *
 * `graph` and `table` get the full width because they are read left to right; the picture
 * kinds tile.
 */
export function DiagnosticsPanel({
  experimentId,
  entries,
}: {
  experimentId: number;
  entries: DiagnosticEntry[];
}) {
  return (
    <div className="grid gap-6 sm:grid-cols-2">
      {entries.map((entry) => (
        <DiagnosticCard
          key={`${entry.key}-${entry.image_id ?? "run"}`}
          experimentId={experimentId}
          entry={entry}
          wide={entry.kind === "graph" || entry.kind === "table" || entry.kind === "grid"}
        />
      ))}
    </div>
  );
}
