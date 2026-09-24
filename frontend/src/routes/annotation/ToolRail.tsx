/**
 * The narrow left rail: tools, history, view and the shortcut sheet. Every button names its
 * key, read from `EDITOR_BINDINGS` so a tooltip cannot promise a key the keymap does not bind.
 */

import {
  Brush,
  Eraser,
  Keyboard,
  Maximize2,
  MousePointer2,
  Redo2,
  Shapes,
  Undo2,
  WandSparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import type { ReactNode } from "react";

import type { EditorTool } from "../../components/annotation/AnnotationCanvas";
import { type EditorCommand, withKeys } from "../../components/annotation/editorKeys";
import { Tooltip, cn, focusRing } from "@vitavision/lab-ui";

const TOOLS: { tool: EditorTool; command: EditorCommand; icon: ReactNode; label: string }[] = [
  { tool: "select", command: "tool.select", icon: <MousePointer2 />, label: "Select" },
  { tool: "polygon", command: "tool.polygon", icon: <Shapes />, label: "Polygon" },
  { tool: "brush", command: "tool.brush", icon: <Brush />, label: "Brush" },
  { tool: "eraser", command: "tool.eraser", icon: <Eraser />, label: "Eraser" },
  { tool: "assist", command: "tool.assist", icon: <WandSparkles />, label: "Contour assist" },
];

/** One rail step. The wheel's is finer; a button press should visibly move. */
const ZOOM_STEP = 1.25;

export function ToolRail({
  tool,
  onTool,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onZoom,
  onFit,
  onActualPixels,
  onShortcuts,
}: {
  tool: EditorTool;
  onTool: (tool: EditorTool) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  /** Zoom about the pane's centre by `factor`; the canvas owns the range. */
  onZoom: (factor: number) => void;
  onFit: () => void;
  onActualPixels: () => void;
  onShortcuts: () => void;
}) {
  return (
    <aside
      className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-2"
      aria-label="Annotation tools"
    >
      {TOOLS.map((item) => (
        <ToolButton
          key={item.tool}
          icon={item.icon}
          label={withKeys(item.label, item.command)}
          active={tool === item.tool}
          onClick={() => onTool(item.tool)}
        />
      ))}
      <span className="my-1 h-px w-6 bg-line" />
      <ToolButton icon={<Undo2 />} label={withKeys("Undo", "undo")} disabled={!canUndo} onClick={onUndo} />
      <ToolButton icon={<Redo2 />} label={withKeys("Redo", "redo")} disabled={!canRedo} onClick={onRedo} />
      <span className="my-1 h-px w-6 bg-line" />
      <ToolButton icon={<ZoomIn />} label="Zoom in" onClick={() => onZoom(ZOOM_STEP)} />
      <ToolButton icon={<ZoomOut />} label="Zoom out" onClick={() => onZoom(1 / ZOOM_STEP)} />
      <ToolButton icon={<Maximize2 />} label={withKeys("Fit image", "view.fit")} onClick={onFit} />
      <ToolButton
        icon={<span className="font-mono text-[9px] font-semibold">1:1</span>}
        label={withKeys("Actual pixels", "view.actual")}
        onClick={onActualPixels}
      />
      <span className="mt-auto" />
      <ToolButton
        icon={<Keyboard />}
        label={withKeys("Keyboard shortcuts", "shortcuts")}
        onClick={onShortcuts}
      />
    </aside>
  );
}

function ToolButton({
  icon,
  label,
  active = false,
  disabled = false,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <button
        type="button"
        aria-label={label}
        aria-pressed={active}
        disabled={disabled}
        onClick={onClick}
        className={cn(
          "grid size-9 place-items-center rounded-control text-fg-muted transition-colors disabled:opacity-30",
          active ? "bg-signal text-signal-fg" : "hover:bg-raised hover:text-fg",
          focusRing,
        )}
      >
        <span className="[&_svg]:size-4">{icon}</span>
      </button>
    </Tooltip>
  );
}
