/**
 * What the tool in hand needs, and nothing else.
 *
 * This section used to be headed "New region" and opened with two dropdowns that set the
 * class and the operation of the *next* shape. Both are gone. The class picker had one option
 * on every dataset this application can produce — there is a taxonomy table but no way to add
 * to it — and the operation picker duplicated a control that already sits in Selection below,
 * on a shape that is already selected, because every path that mints one selects it. Choosing
 * Subtract before drawing and flipping to Subtract after drawing produce the same document,
 * and only one of them needed a permanent dropdown.
 */

import type { AnnotationDocument, AnnotationLabel } from "../../api/client";
import type { EditorTool } from "../../components/annotation/AnnotationCanvas";
import { classKeyAt } from "../../components/annotation/editorKeys";
import { MAX_BRUSH_SIZE, MIN_BRUSH_SIZE } from "../../hooks/useBrushSize";
import { Field, NumberInput, Select, Slider } from "@vitavision/lab-ui";
import type { DocumentCommands } from "./useDocumentCommands";

export function ToolSection({
  tool,
  labels,
  document,
  commands,
  brushSize,
  onBrushSize,
}: {
  tool: EditorTool;
  labels: AnnotationLabel[];
  document: AnnotationDocument;
  commands: DocumentCommands;
  brushSize: number;
  onBrushSize: (size: number) => void;
}) {
  /** Whether this dataset has classes to choose *between*, rather than one class it has. */
  const hasTaxonomy = labels.length > 1;
  const { pendingPoints, selected } = commands;
  const brushing = tool === "brush" || tool === "eraser";
  const boxing = tool === "box";
  // With none of them to show, the section is not drawn rather than standing empty.
  if (!hasTaxonomy && pendingPoints.length === 0 && !brushing && !boxing) return null;

  return (
    <section className="border-b border-line p-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-muted">Tool</h2>
      {hasTaxonomy && (
        // Only where there is something to choose between. A dataset with one class gets no
        // picker, and one with four gets a real one: label count is data in exactly the way
        // channel count is (ADR-0041).
        // Named on screen: under a "Tool" heading, a bare dropdown reading "Defect" was a
        // tool picker to anyone who had not already guessed it was the class.
        <Field as="group" label="Class for new regions">
          <Select
            aria-label="New region label"
            value={commands.labelKey}
            // The digit that picks each class, beside it: the keys are otherwise unguessable,
            // and they skip 0 and 1, which are the view's.
            options={labels.map((label, position) => {
              const key = classKeyAt(position);
              return key
                ? { value: label.key, label: label.name, note: key }
                : { value: label.key, label: label.name };
            })}
            onValueChange={commands.setLabelKey}
          />
        </Field>
      )}
      {pendingPoints.length > 0 && (
        // A readout, not a control. Closing is a click on the first vertex or a double-click
        // anywhere; a "Close" button in a side panel is neither where the hand is nor what a
        // polygon tool is expected to need.
        <p className="mt-2 rounded-control bg-raised px-2 py-1.5 text-xs text-fg-muted">
          {pendingPoints.length} vertex{pendingPoints.length === 1 ? "" : "es"} ·{" "}
          {pendingPoints.length < 3
            ? "three closes a ring"
            : "click the first vertex, double-click, or Enter"}{" "}
          · Backspace undoes one
        </p>
      )}
      {boxing && (
        <p className="mt-2 text-[11px] leading-4 text-fg-subtle">
          Drag from one corner to the opposite one. A click without a drag draws nothing; under
          Select, a box moves and its corners resize it.
        </p>
      )}
      {brushing && (
        <div className="mt-3 flex flex-col gap-2">
          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-fg-muted">
              <span>Brush size</span>
              <span className="font-mono">
                {brushSize} px {brushSize === 1 ? "· one pixel" : ""}
              </span>
            </div>
            {/* Slider *and* a number box: the useful values for correcting a mask are at the
                very bottom of a 128-step track, where a drag cannot reliably land on 1 rather
                than 2. The `,` and `.` keys do the same job with the other hand still on the
                canvas. */}
            <div className="flex items-center gap-2">
              <Slider
                aria-label="Brush size"
                value={brushSize}
                min={MIN_BRUSH_SIZE}
                max={MAX_BRUSH_SIZE}
                step={1}
                onValueChange={onBrushSize}
              />
              <NumberInput
                className="w-16 shrink-0"
                aria-label="Brush size in pixels"
                min={MIN_BRUSH_SIZE}
                max={MAX_BRUSH_SIZE}
                step={1}
                value={brushSize}
                onChange={(event) => onBrushSize(Number(event.target.value))}
              />
            </div>
          </div>
          {/* The rule, where the hand is, because it is the one thing about this tool nobody
              can infer from looking at it. */}
          <p className="text-[11px] leading-4 text-fg-subtle">
            {selected?.kind === "bitmap"
              ? tool === "eraser"
                ? "Erasing region " +
                  `${document.shapes.indexOf(selected) + 1} only. Escape to erase across all of them.`
                : "Painting into region " +
                  `${document.shapes.indexOf(selected) + 1}. Escape starts a new one.`
              : tool === "eraser"
                ? "Nothing selected: this takes paint off whatever it passes over, and never adds a region."
                : "Nothing selected: this starts a new region. Strokes after it extend that one."}
          </p>
        </div>
      )}
    </section>
  );
}
