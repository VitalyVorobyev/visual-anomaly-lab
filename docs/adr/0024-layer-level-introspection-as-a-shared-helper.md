# ADR-0024: Layer-level introspection is a shared torch helper on the existing `graph` kind

**Status:** Accepted (2026-08-08)

Extends **ADR-0018**. The diagnostics contract, the `graph` kind and the render-by-kind rule are
unchanged; this decides how a payload of that kind comes to describe a whole network rather than
three boxes.

## Context

M4's architecture view drew the three EfficientAD branches with the shape going in, the shape coming
out, and a parameter count. That is honest — every number is read off a real forward pass — and it
answers almost nothing anyone opens the tab to ask. Where do the 8 million parameters actually sit?
Which layer drops the resolution, and by how much? The student and the teacher are the same class
with different output widths; where does that show up? The view could not say, because
`_emit_architecture` looped over a hardcoded tuple of three branch names.

The obvious implementation is `torchinfo` or a hand-written recursion inside the EfficientAD plugin.
Both fail the same test: the next method would need its own copy, and **ADR-0018 exists precisely so
that M6's `efficientad_custom` inherits M4's views with no new code**. Whatever produces this has to
be shared and method-agnostic, and it has to work on a checkout where torch is not installed at all —
CI runs without the `dl` extra and that boundary is measured, not asserted.

There is also a real limit worth deciding about rather than discovering. `named_modules()` enumerates
*modules*. Operations written directly into a `forward` — `F.relu`, `torch.cat`, an addition — are
not modules and cannot be seen this way. A node list gathered from hooks is therefore complete; an
edge list inferred from it would not be.

## Decision

**Forward hooks over `named_modules()` during one dry pass, in a shared helper, extending the `graph`
payload additively.**

- **`models/introspect.py`, split in two.** `build_tree(records, *, max_nodes)` is **torch-free** —
  plain `ModuleRecord`s in, a payload out — so the hierarchy, the bounding and the truncation
  reporting are tested in CI. `collect(root, probe, *, prefix)` imports torch inside the function,
  registers a hook on every module, runs one `no_grad` pass, and removes every hook in a `finally`;
  a hook left on a module that is about to be trained would fire on every step of the real loop.
- **A plugin contributes only what it alone knows: which roots to walk, and how they are wired.**
  `_emit_architecture` calls `collect` per branch and states the two inter-branch edges, because
  those live in the training loop's losses and not in any module's `forward`. Everything else is the
  helper's.
- **The payload extends `graph`; no new kind.** Nodes keep `id`, `label`, `type`, `parameters`,
  `input_shape` and `output_shape` with their existing meanings — `parameters` is still the whole
  subtree, so the flat card renderer prints exactly what it always did. `parent`, `depth`, `order`,
  `parameters_own`, `executed`, `calls` and `leaf` are new; a renderer that does not know them
  ignores them, and an index written before this still draws.
- **Two parameter counts, not one.** `parameters_own` is `recurse=False`; `parameters` is the
  subtree. A container's recursive count double-counts against its children, so a single column
  would not sum to anything. `parameters_own` summed over every node equals the model's total, and a
  test asserts it against a real network.
- **`depth` and `parent` come from the surviving tree, not from counting dots.** Dots are wrong twice:
  a root collected without a prefix has no dots but is depth 0, and a node whose intermediate
  ancestors were truncated must attach — and be drawn — where it actually sits.
- **Bounded by node count, not by depth**, at 1500, with `truncated_nodes` reported. Truncating by
  depth would drop exactly the layers a reader opened the tab for, and "expand this subtree" cannot
  work on nodes the payload does not contain. Deepest nodes go first, so the top of the tree — the
  part that orients a reader — always survives.
- **A module that was never called is marked, not blanked.** `executed: false` is a fact about the
  model; a dash where a shape should be says the same thing as a recording failure.
- **`edges` stays branch-level, and the UI says so on screen.** The tree is drawn under the cards
  with a sentence stating that operations written into a `forward` are not modules and so are not
  there. **The view must not draw wiring it did not measure.**
- **The UI is a tree of rows.** Not nested boxes and not a node-link diagram: a real backbone is
  hundreds of modules, the quantities a reader compares line up into columns, and a node-link layout
  needs the layout engine that "charts are hand-rolled SVG" ruled out.

**Ruled out:** `torchinfo` (a dependency in the `dl` extra for something that is forty lines, and it
formats for a terminal rather than returning a structure); a recursion inside the EfficientAD plugin
(the next method copies it); a new diagnostic kind (a renderer switch grows, and every older index
stops drawing); and inferring edges from execution order (it would produce a chain through modules
that are not connected, which is a confident picture of something false).

## Consequences

`efficientad_custom` and `patchcore_anomalib` get the whole view by calling two functions. On the
reference configuration the EfficientAD payload is 37 nodes over three depths, and the resolution
cascade — 256 → 253 → 126 → 123 → 61 → 59 → 56 — is legible for the first time.

Negative consequences, accepted honestly:

- **No functional operations, ever, by construction.** Anyone expecting a Netron-style graph will be
  disappointed, which is why the tab says it in a sentence rather than leaving it to be discovered.
- **A dry forward pass costs a real forward pass**, on the training device, before training starts.
  Milliseconds against a run measured in minutes, and it is what makes the shapes true.
- **A module called more than once records only its first call's shapes**, with the count beside it.
  A module whose shape genuinely changes between calls is described by its first, and the `×n` is
  the only warning.
- **A module returning a tuple records its first element's shape.** The alternative is a field that
  is sometimes a list of lists, which every renderer would then handle for the sake of a few modules.
- **The tree's parameters and the model's total need not agree**, and on EfficientAD they do not:
  8 057 856 summed over the walked branches against a stated total of 8 058 628. The 772 are
  `mean_std` (384 + 384) and the four normalization quantiles, which live on the model and belong to
  no branch. Both numbers are correct for what they measure, they round to the same 8.06 M on
  screen, and a caller that walks every root would see them coincide.
- **`DiagnosticIndex` grows again**, as ADR-0018 predicted and ADR-0019 already noted. The payload is
  `dict[str, Any]`, so none of this is typed on the wire, and the frontend's interface is the
  contract by convention rather than by generation.
