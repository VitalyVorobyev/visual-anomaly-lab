/**
 * The architecture view and the teacher inspector.
 *
 * Both are the same three lines: take the run-scoped entries, narrow them by `kind`, hand
 * them to the panel. Neither knows a method's name, and neither knows a key — the split
 * between the two tabs is "structure the model declared" versus "pictures the model
 * produced", which is a kind distinction and nothing else.
 *
 * That is what M6 inherits: `efficientad_custom` emits a `graph` and some `image`s and
 * appears in both tabs with no code written here.
 */

import type { DiagnosticIndex } from "../../api/client";
import { modelScoped, ofKinds } from "../../api/diagnostics";
import { DiagnosticsPanel } from "../../components/diagnostics/DiagnosticsPanel";
import { Empty, Panel } from "@vitavision/lab-ui";

/** Kinds that describe the model's own structure. */
export const STRUCTURE_KINDS = ["graph", "table"];

/** Kinds that are a picture of something the model computed. */
export const PICTURE_KINDS = ["map", "image", "grid"];

const NOTHING_YET =
  "Nothing recorded yet. Diagnostics are written during training — run it, or re-run with " +
  "diagnostics enabled.";

export function ArchitectureTab({
  experimentId,
  index,
}: {
  experimentId: number;
  index: DiagnosticIndex | undefined;
}) {
  const entries = ofKinds(modelScoped(index), STRUCTURE_KINDS);

  return (
    <Panel title="Architecture">
      {entries.length === 0 ? (
        <Empty>{NOTHING_YET}</Empty>
      ) : (
        <div className="flex flex-col gap-6">
          <p className="text-xs text-fg-muted">
            Read from a real forward pass when the model was trained, not drawn by hand — so
            these shapes are the shapes this experiment actually ran, and change with its
            configuration.
          </p>
          <DiagnosticsPanel experimentId={experimentId} entries={entries} />
        </div>
      )}
    </Panel>
  );
}

export function InspectorTab({
  experimentId,
  index,
}: {
  experimentId: number;
  index: DiagnosticIndex | undefined;
}) {
  const entries = ofKinds(modelScoped(index), PICTURE_KINDS);

  return (
    <Panel title="Inspector">
      {entries.length === 0 ? (
        <Empty>{NOTHING_YET}</Empty>
      ) : (
        <div className="flex flex-col gap-6">
          <p className="text-xs text-fg-muted">
            What this method recorded about itself during training. Per-image diagnostics live
            on each sample's page, beside its anomaly map.
          </p>
          <DiagnosticsPanel experimentId={experimentId} entries={entries} />
        </div>
      )}
    </Panel>
  );
}
