/** The selected region: its class, its operation, delete, and tracing a mask into contours. */

import { Trash2, WandSparkles } from "lucide-react";

import type { AnnotationLabel } from "../../api/client";
import { Button, Select } from "@vitavision/lab-ui";
import type { DocumentCommands } from "./useDocumentCommands";

export function SelectionSection({
  labels,
  commands,
}: {
  labels: AnnotationLabel[];
  commands: DocumentCommands;
}) {
  const hasTaxonomy = labels.length > 1;
  const { selected, tracing, traceError } = commands;
  return (
    <section className="min-h-32 border-t border-line p-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-muted">
        Selection
      </h2>
      {!selected ? (
        <p className="text-xs leading-5 text-fg-subtle">
          Select a region to {hasTaxonomy ? "edit its class or operation" : "make it a cut"},
          drag it with Select, or nudge it with the arrow keys.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {hasTaxonomy && (
            <Select
              aria-label="Selected region label"
              value={selected.label_key}
              options={labels.map((label) => ({ value: label.key, label: label.name }))}
              onValueChange={(value) => commands.updateSelected({ label_key: value })}
            />
          )}
          <div className="flex gap-2">
            <Select
              aria-label="Selected region operation"
              value={selected.operation}
              options={[
                { value: "add", label: "Add defect" },
                { value: "subtract", label: "Subtract" },
              ]}
              onValueChange={(value) =>
                commands.updateSelected({ operation: value as "add" | "subtract" })
              }
            />
            <Button
              variant="danger"
              icon={<Trash2 />}
              onClick={commands.removeSelected}
              aria-label="Delete selected region"
            />
          </div>
          <p className="text-[11px] leading-4 text-fg-subtle">
            Drag to move · arrows nudge 1 px, Shift 10 px
            {selected.kind === "polygon" ? " · drag a vertex to reshape" : " · brush extends it"}
          </p>
          {selected.kind === "bitmap" && (
            <Button
              icon={<WandSparkles />}
              disabled={tracing}
              onClick={() => void commands.traceSelected()}
            >
              {tracing ? "Tracing…" : "Make editable contour"}
            </Button>
          )}
          {traceError && <p className="text-xs text-defect">{traceError}</p>}
        </div>
      )}
    </section>
  );
}
