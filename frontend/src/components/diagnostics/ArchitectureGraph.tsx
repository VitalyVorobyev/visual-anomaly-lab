/**
 * The architecture view: whatever a model recorded about its own structure.
 *
 * Every number here was read off a real forward pass by the plugin, so the diagram cannot
 * go stale against the model it claims to describe — change the experiment's preprocessing
 * size and the shapes change here with no edit to this file. That is the whole reason the
 * graph is captured rather than drawn.
 *
 * Layout follows the recorded edges rather than assuming a chain: nodes are placed in
 * dependency order and anything unreferenced is appended, so a model whose branches fan
 * out instead of stacking draws correctly without a special case here.
 */

import { Badge, Empty } from "../ui";

interface GraphNode {
  id: string;
  label?: string;
  type?: string;
  parameters?: number;
  input_shape?: number[];
  output_shape?: number[];
}

interface GraphEdge {
  source: string;
  target: string;
  label?: string;
}

/**
 * Nodes in dependency order.
 *
 * A depth-first walk from every node with no incoming edge. A cycle — which nothing here
 * should produce, but which a future plugin could — terminates on the `seen` set rather
 * than looping, so a malformed graph renders in some order instead of hanging the tab.
 */
function ordered(nodes: GraphNode[], edges: GraphEdge[]): GraphNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const targets = new Set(edges.map((edge) => edge.target));
  const outgoing = new Map<string, string[]>();
  for (const edge of edges) {
    outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target]);
  }

  const seen = new Set<string>();
  const result: GraphNode[] = [];
  const visit = (id: string) => {
    if (seen.has(id)) return;
    seen.add(id);
    const node = byId.get(id);
    if (node) result.push(node);
    for (const next of outgoing.get(id) ?? []) visit(next);
  };

  for (const node of nodes) if (!targets.has(node.id)) visit(node.id);
  for (const node of nodes) visit(node.id);
  return result;
}

function shape(dimensions: number[] | undefined): string {
  return dimensions && dimensions.length > 0 ? `(${dimensions.join(", ")})` : "—";
}

function parameterCount(value: number | undefined): string {
  if (typeof value !== "number") return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)} k`;
  return String(value);
}

export function ArchitectureGraph({ payload }: { payload: Record<string, unknown> }) {
  const nodes = Array.isArray(payload.nodes) ? (payload.nodes as GraphNode[]) : [];
  const edges = Array.isArray(payload.edges) ? (payload.edges as GraphEdge[]) : [];

  if (nodes.length === 0) {
    return <Empty>This run recorded a graph with no nodes in it.</Empty>;
  }

  const laid = ordered(nodes, edges);
  const labelFor = (source: string, target: string) =>
    edges.find((edge) => edge.source === source && edge.target === target)?.label;

  const total = payload.total_parameters;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-stretch gap-2">
        {laid.map((node, index) => {
          const previous = laid[index - 1];
          const connector = previous ? labelFor(previous.id, node.id) : undefined;
          return (
            <div key={node.id} className="flex items-stretch gap-2">
              {index > 0 && (
                <div className="flex flex-col items-center justify-center px-1 text-fg-subtle">
                  <span aria-hidden className="text-lg leading-none">
                    →
                  </span>
                  {connector && <span className="text-[10px]">{connector}</span>}
                </div>
              )}
              <figure className="min-w-44 rounded-lg border border-line p-3 ">
                <figcaption className="flex items-center gap-2">
                  <span className="text-sm font-semibold">{node.label ?? node.id}</span>
                  {node.type && <Badge tone="info">{node.type}</Badge>}
                </figcaption>
                <dl className="mt-2 flex flex-col gap-0.5 font-mono text-xs text-fg-muted">
                  <div className="flex justify-between gap-3">
                    <dt>in</dt>
                    <dd>{shape(node.input_shape)}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt>out</dt>
                    <dd>{shape(node.output_shape)}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt>params</dt>
                    <dd>{parameterCount(node.parameters)}</dd>
                  </div>
                </dl>
              </figure>
            </div>
          );
        })}
      </div>

      {typeof total === "number" && (
        <p className="font-mono text-xs text-fg-muted">
          {parameterCount(total)} parameters in total
        </p>
      )}
    </div>
  );
}
