/** The narrow left rail: tools, history and view. Every button names its key. */

import {
  Brush,
  Eraser,
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

import type { CanvasView, EditorTool } from "../../components/annotation/AnnotationCanvas";
import { Tooltip, cn, focusRing } from "@vitavision/lab-ui";

const TOOLS: { tool: EditorTool; icon: ReactNode; label: string }[] = [
  { tool: "select", icon: <MousePointer2 />, label: "Select (V)" },
  { tool: "polygon", icon: <Shapes />, label: "Polygon (P)" },
  { tool: "brush", icon: <Brush />, label: "Brush (B)" },
  { tool: "eraser", icon: <Eraser />, label: "Eraser (E)" },
  { tool: "assist", icon: <WandSparkles />, label: "Contour assist (A)" },
];

export function ToolRail({
  tool,
  onTool,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  view,
  onView,
  onFit,
  onActualPixels,
}: {
  tool: EditorTool;
  onTool: (tool: EditorTool) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  view: CanvasView;
  onView: (view: CanvasView) => void;
  onFit: () => void;
  onActualPixels: () => void;
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
          label={item.label}
          active={tool === item.tool}
          onClick={() => onTool(item.tool)}
        />
      ))}
      <span className="my-1 h-px w-6 bg-line" />
      <ToolButton icon={<Undo2 />} label="Undo (⌘Z)" disabled={!canUndo} onClick={onUndo} />
      <ToolButton icon={<Redo2 />} label="Redo (⇧⌘Z)" disabled={!canRedo} onClick={onRedo} />
      <span className="my-1 h-px w-6 bg-line" />
      <ToolButton icon={<ZoomIn />} label="Zoom in" onClick={() => onView({ ...view, zoom: Math.min(12, view.zoom * 1.25) })} />
      <ToolButton icon={<ZoomOut />} label="Zoom out" onClick={() => onView({ ...view, zoom: Math.max(0.25, view.zoom / 1.25) })} />
      <ToolButton icon={<Maximize2 />} label="Fit image (0)" onClick={onFit} />
      <ToolButton
        icon={<span className="font-mono text-[9px] font-semibold">1:1</span>}
        label="Actual pixels (1)"
        onClick={onActualPixels}
      />
      <span className="mt-auto px-1 text-center font-mono text-[9px] leading-3 text-fg-subtle">
        {view.zoom === 1 ? "Fit" : `${Math.round(view.zoom * 100)}% fit`}
      </span>
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
