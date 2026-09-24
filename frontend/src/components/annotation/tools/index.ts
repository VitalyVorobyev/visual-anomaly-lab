/** Every editing tool, by name. Adding a tool is one module and one entry here. */

import { assistTool } from "./assist";
import { brushTool } from "./brush";
import { polygonTool } from "./polygon";
import { selectTool } from "./select";
import type { EditorTool, ToolModule } from "./types";

export const TOOLS: Record<EditorTool, ToolModule> = {
  select: selectTool,
  polygon: polygonTool,
  brush: brushTool,
  eraser: brushTool,
  assist: assistTool,
};

export type { EditorTool, Gesture, ToolContext, ToolEffect, ToolModule } from "./types";
