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
 *
 * **Two views over one payload.** The cards are the branches and how they are wired — the
 * only connections anyone measured, since forward hooks see modules and not the functional
 * operations between them. Below them, when the payload carries a hierarchy, is every
 * module the pass actually reached (ADR-0024). A payload with no hierarchy — an index
 * written before M4.7, or a method that records only its branches — draws the cards alone,
 * exactly as it did.
 */

import { Badge, Empty } from "../ui";
import { ModuleTree, hasHierarchy, parameterCount, shape } from "./ModuleTree";
import type { ModuleNode } from "./ModuleTree";

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

export function ArchitectureGraph({ payload }: { payload: Record<string, unknown> }) {
  const nodes = Array.isArray(payload.nodes) ? (payload.nodes as ModuleNode[]) : [];
  const edges = Array.isArray(payload.edges) ? (payload.edges as GraphEdge[]) : [];

  if (nodes.length === 0) {
    return <Empty>This run recorded a graph with no nodes in it.</Empty>;
  }

  const tree = hasHierarchy(nodes);
  // The cards are the top of the tree: the branches, and the edges between them. With no
  // hierarchy every node is a card, which is what M4 drew and what an older index still is.
  const cards = tree ? nodes.filter((node) => (node.depth ?? 0) === 0) : nodes;
  const laid = ordered(cards, edges);
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

      {tree && (
        <>
          <p className="text-xs text-fg-muted">
            {/* Said up front rather than after someone asks why their `cat` is missing. */}
            Every module the forward pass reached, with the shapes it produced. Operations
            written directly into a <span className="font-mono">forward</span> — an
            activation, a concatenation — are not modules and so are not here; the arrows
            above are the only connections that were measured.
          </p>
          <ModuleTree
            nodes={nodes}
            truncated={typeof payload.truncated_nodes === "number" ? payload.truncated_nodes : 0}
            maxNodes={typeof payload.max_nodes === "number" ? payload.max_nodes : undefined}
          />
        </>
      )}
    </div>
  );
}
