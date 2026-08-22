/**
 * Every module a model is made of, with the shapes it actually produced (handbook diagnostics.md).
 *
 * A tree of rows rather than nested boxes or a node-link diagram, for three reasons. A
 * real backbone is hundreds of modules and a canvas of boxes stops being readable well
 * before that. The quantities a reader compares — resolution at each layer, where the
 * parameters are — line up into columns, which is a table. And "charts are hand-rolled
 * SVG" (M4) ruled out pulling in a layout engine, which is what a node-link view would
 * need.
 *
 * **It renders what the payload contains and never infers structure.** `parent` and
 * `depth` come from the plugin's own record of the forward pass; a module that was never
 * called is drawn muted rather than given a plausible shape. The edges are drawn by the
 * card view above, because those are the only connections anyone measured — forward hooks
 * see modules, not the functional operations between them.
 */

import { useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";

import { Badge, Button, Input, cn } from "@vitavision/lab-ui";

export interface ModuleNode {
  id: string;
  label?: string;
  type?: string;
  parameters?: number;
  parameters_own?: number;
  parent?: string | null;
  depth?: number;
  order?: number;
  executed?: boolean;
  calls?: number;
  leaf?: boolean;
  input_shape?: number[];
  output_shape?: number[];
}

/** Whether a payload carries a hierarchy at all, so the caller can fall back. */
export function hasHierarchy(nodes: ModuleNode[]): boolean {
  return nodes.some((node) => node.parent !== undefined && node.parent !== null);
}

export function shape(dimensions: number[] | undefined | null): string {
  return dimensions && dimensions.length > 0 ? `(${dimensions.join(", ")})` : "—";
}

export function parameterCount(value: number | undefined): string {
  if (typeof value !== "number") return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)} k`;
  return String(value);
}

/** Depth 0 and 1 open by default: the branches and their layers, not every leaf. */
const OPEN_TO_DEPTH = 1;

export function ModuleTree({
  nodes,
  truncated = 0,
  maxNodes,
}: {
  nodes: ModuleNode[];
  truncated?: number;
  maxNodes?: number;
}) {
  const [filter, setFilter] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(
    () => new Set(nodes.filter((node) => (node.depth ?? 0) >= OPEN_TO_DEPTH).map((n) => n.id)),
  );

  const children = useMemo(() => {
    const map = new Map<string, ModuleNode[]>();
    for (const node of nodes) {
      const key = node.parent ?? "";
      map.set(key, [...(map.get(key) ?? []), node]);
    }
    for (const list of map.values()) {
      list.sort((left, right) => (left.order ?? 0) - (right.order ?? 0));
    }
    return map;
  }, [nodes]);

  /*
   * A filter has to keep a match's ancestors, or a matched leaf appears at the root with
   * no path to it — which reads as a different model rather than as a filtered one.
   */
  const matched = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return null;
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const keep = new Set<string>();
    for (const node of nodes) {
      const haystack = `${node.id} ${node.type ?? ""}`.toLowerCase();
      if (!haystack.includes(needle)) continue;
      let current: ModuleNode | undefined = node;
      while (current) {
        keep.add(current.id);
        current = current.parent ? byId.get(current.parent) : undefined;
      }
    }
    return keep;
  }, [filter, nodes]);

  const rows: { node: ModuleNode; depth: number; hasChildren: boolean }[] = [];
  const walk = (parentId: string, depth: number) => {
    for (const node of children.get(parentId) ?? []) {
      if (matched && !matched.has(node.id)) continue;
      const kids = (children.get(node.id) ?? []).filter(
        (child) => !matched || matched.has(child.id),
      );
      rows.push({ node, depth, hasChildren: kids.length > 0 });
      // A filter expands what it matched: hiding the hits behind a caret would make the
      // control look broken.
      if (kids.length > 0 && (matched !== null || !collapsed.has(node.id))) {
        walk(node.id, depth + 1);
      }
    }
  };
  walk("", 0);

  const toggle = (id: string) =>
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          className="w-56"
          aria-label="Filter modules"
          placeholder="Filter by name or type"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <Button size="sm" onClick={() => setCollapsed(new Set())}>
          Expand all
        </Button>
        <Button
          size="sm"
          onClick={() => setCollapsed(new Set(nodes.filter((n) => !n.leaf).map((n) => n.id)))}
        >
          Collapse all
        </Button>
        <span className="ml-auto font-mono text-xs text-fg-subtle">
          {rows.length} of {nodes.length} modules
        </span>
      </div>

      {/* Wide content scrolls inside its own box rather than widening the page. */}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[44rem] text-left text-xs">
          <thead className="text-fg-muted">
            <tr>
              <th className="py-1 pr-4 font-medium">module</th>
              <th className="py-1 pr-4 font-medium">in</th>
              <th className="py-1 pr-4 font-medium">out</th>
              <th className="py-1 text-right font-medium">params</th>
            </tr>
          </thead>
          <tbody role="tree">
            {rows.map(({ node, depth, hasChildren }) => {
              const open = matched !== null || !collapsed.has(node.id);
              const dead = node.executed === false;
              return (
                <tr
                  key={node.id}
                  role="treeitem"
                  aria-level={depth + 1}
                  aria-expanded={hasChildren ? open : undefined}
                  className="border-t border-line/60"
                >
                  <td className="py-1 pr-4">
                    <span
                      className="flex items-center gap-1.5"
                      style={{ paddingLeft: `${depth * 1.1}rem` }}
                    >
                      {hasChildren ? (
                        <button
                          type="button"
                          aria-label={open ? `Collapse ${node.id}` : `Expand ${node.id}`}
                          className="text-fg-subtle hover:text-fg"
                          onClick={() => toggle(node.id)}
                        >
                          <ChevronRight
                            className={cn("size-3 transition-transform", open && "rotate-90")}
                          />
                        </button>
                      ) : (
                        <span className="size-3" aria-hidden />
                      )}
                      <span className={cn("font-mono", dead ? "text-fg-subtle" : "text-fg")}>
                        {node.label ?? node.id}
                      </span>
                      {node.type && <Badge tone="info">{node.type}</Badge>}
                      {/* A module reused in a loop ran more than once, and only the first
                          call's shapes are recorded — saying so beats implying one pass. */}
                      {typeof node.calls === "number" && node.calls > 1 && (
                        <span className="font-mono text-fg-subtle">×{node.calls}</span>
                      )}
                      {dead && <span className="text-fg-subtle">not called</span>}
                    </span>
                  </td>
                  <td className="py-1 pr-4 font-mono text-fg-muted">{shape(node.input_shape)}</td>
                  <td className="py-1 pr-4 font-mono text-fg-muted">{shape(node.output_shape)}</td>
                  <td className="py-1 text-right font-mono tabular-nums text-fg-muted">
                    {/* Own first, because that is the column that sums to the total; the
                        subtree count is beside it for a container, and identical for a
                        leaf, so it is only shown where the two differ. */}
                    {parameterCount(node.parameters_own)}
                    {node.parameters !== node.parameters_own && (
                      <span className="text-fg-subtle"> / {parameterCount(node.parameters)}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {rows.length === 0 && (
        <p className="text-xs text-fg-muted">No module matches that filter.</p>
      )}

      {truncated > 0 && (
        <p className="text-xs text-warn">
          {/* A silent truncation would read as "this is the whole model". */}
          {truncated} deeper module(s) were not recorded — the payload is capped at{" "}
          {maxNodes ?? "its limit"} nodes.
        </p>
      )}
    </div>
  );
}
