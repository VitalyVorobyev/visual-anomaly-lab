"""What a torch model is actually made of, read from a real forward pass (ADR-0024).

M4's architecture view drew three boxes — teacher, student, autoencoder — with the shape
going in and the shape coming out. That is honest and it is not enough: it says nothing
about where the parameters are, which layer changes the resolution, or how the two
patch-description networks differ, and those are the questions someone opens the tab to
ask.

So this walks `named_modules()` with forward hooks during one dry pass and records every
module's real input and output shapes. **Nothing here is method-specific.** A plugin calls
`collect()` and hands the result to `build_tree()`; `efficientad_custom` in M6 and
`patchcore_anomalib` in M7 inherit the whole view by doing the same, which is the property
ADR-0018 exists to protect.

The module is split deliberately:

  * `build_tree` is **torch-free** — plain records in, a payload out — so the hierarchy,
    the bounding and the truncation reporting are all tested in CI, which installs without
    the `dl` extra.
  * `collect` imports torch inside the function, like every other plugin path, so
    importing this module costs nothing.

**Known limitation, stated rather than discovered.** `named_modules()` sees *modules*.
Functional operations — `F.relu`, `torch.cat`, arithmetic written directly in `forward` —
are invisible to it, so a node list is complete and an edge list would not be. This
records nodes finely and leaves the branch-level edges to the caller: **the view must not
draw wiring it did not measure.**
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_NODES = 1500
"""Bounded by node **count**, not by depth.

Truncating by depth would drop exactly the layers a reader opened the tab for, and
"expand this subtree" cannot work on nodes the payload does not contain. Collapsing is a
display decision and belongs in the UI; this decides only what is *there* to collapse.
"""


@dataclass
class ModuleRecord:
    """One module, as a forward pass found it.

    Deliberately plain data with no torch types, so the half of this file that arranges
    records into a tree can be tested on a checkout with numpy and Pillow and nothing else.
    """

    name: str
    """Dotted path from the root, e.g. `student.conv1`. Empty string for the root itself."""
    type_name: str
    own_parameters: int
    """Parameters declared on this module alone (`recurse=False`)."""
    total_parameters: int
    """Including every descendant. Both are recorded because a container's recursive count
    double-counts against its children, and showing one alone makes the tree not add up."""
    order: int = 0
    """Execution order, so siblings list in the order the data actually flows."""
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    calls: int = 0
    """A module reused in a loop runs more than once; the first call's shapes are kept."""

    @property
    def executed(self) -> bool:
        """Whether the forward pass reached it at all.

        Different from "has no shape". A module that is defined but never called — a
        branch behind a flag, a head the wrapper does not use — is a real fact about the
        model, and printing a dash for it says the same thing as a shape that failed to
        record.
        """
        return self.calls > 0


def build_tree(
    records: Sequence[ModuleRecord],
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """Arrange records into the `graph` payload the architecture view renders.

    Additive over what ADR-0018's `graph` kind already carried: every node keeps `id`,
    `label`, `type`, `parameters`, `input_shape` and `output_shape`, so an index written
    before this — and any payload with no hierarchy — renders exactly as it did. `parent`,
    `depth`, `order`, `parameters_own`, `executed`, `calls` and `leaf` are new, and a
    renderer that does not know them ignores them.
    """
    kept = _bounded(records, max_nodes)
    present = {record.name for record in kept}
    parents = {record.name: _parent_of(record.name, present) for record in kept}

    children: dict[str, int] = {}
    for parent in parents.values():
        if parent is not None:
            children[parent] = children.get(parent, 0) + 1

    # Depth walks the parent chain rather than counting dots. Dots are wrong twice: a root
    # collected without a prefix has an empty name and no dots but is still depth 0, and a
    # node whose intermediate ancestors were truncated must sit at the depth it is actually
    # drawn at rather than the one its full path implies.
    depths: dict[str, int] = {}

    def depth_of(name: str) -> int:
        if name in depths:
            return depths[name]
        parent = parents.get(name)
        depths[name] = 0 if parent is None else depth_of(parent) + 1
        return depths[name]

    nodes = [
        {
            "id": record.name,
            "label": record.name.rsplit(".", 1)[-1] if record.name else "model",
            "type": record.type_name,
            # `parameters` keeps its existing meaning — the whole subtree — so the old
            # flat renderer prints the same number it always did for a branch node.
            "parameters": record.total_parameters,
            "parameters_own": record.own_parameters,
            "parent": parents[record.name],
            "depth": depth_of(record.name),
            "order": record.order,
            "executed": record.executed,
            "calls": record.calls,
            "leaf": children.get(record.name, 0) == 0,
            "input_shape": record.input_shape,
            "output_shape": record.output_shape,
        }
        for record in kept
    ]

    return {
        "nodes": nodes,
        "edges": [],
        "max_nodes": max_nodes,
        "truncated_nodes": len(records) - len(kept),
    }


def _parent_of(name: str, present: set[str]) -> str | None:
    """The nearest ancestor that survived the bound, by longest dotted prefix.

    Walking up rather than trimming one segment matters once truncation is in play: a
    node whose immediate parent was dropped must still attach somewhere, or the tree
    silently loses a subtree it claims to have kept.
    """
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in present:
            return candidate
    return "" if name and "" in present else None


def _bounded(records: Sequence[ModuleRecord], max_nodes: int) -> list[ModuleRecord]:
    """Drop the deepest nodes first, keeping the shape of the model legible.

    Deepest-first because the top of the tree is what orients a reader: losing the leaves
    of one enormous backbone still leaves a readable model, while losing a branch root
    leaves its children orphaned and the totals unexplainable.
    """
    if len(records) <= max_nodes:
        return list(records)
    ordered = sorted(records, key=lambda record: (record.name.count("."), record.order))
    keep = {record.name for record in ordered[:max_nodes]}
    return [record for record in records if record.name in keep]


def collect(root: Any, probe: Any, *, prefix: str = "") -> list[ModuleRecord]:
    """Run one dry forward pass with hooks on every module, and record what happened.

    `prefix` namespaces the records, so a caller with several roots — EfficientAD's three
    branches — gets one flat list whose names do not collide.

    Every hook is removed in a `finally`: a hook left behind on a module that is about to
    be trained would run on every step of the real loop.
    """
    import torch

    counter = {"order": 0}
    records: dict[str, ModuleRecord] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            record = records[name]
            record.calls += 1
            if record.calls == 1:
                record.order = counter["order"]
                counter["order"] += 1
                record.input_shape = _shape_of(args[0] if args else None)
                record.output_shape = _shape_of(output)

        return hook

    for name, module in root.named_modules():
        full = f"{prefix}{name}" if name else prefix.rstrip(".")
        records[full] = ModuleRecord(
            name=full,
            type_name=type(module).__name__,
            own_parameters=int(sum(p.numel() for p in module.parameters(recurse=False))),
            total_parameters=int(sum(p.numel() for p in module.parameters())),
        )
        handles.append(module.register_forward_hook(make_hook(full)))

    try:
        with torch.no_grad():
            root(probe, **probe_arguments(root, probe))
    finally:
        for handle in handles:
            handle.remove()

    return list(records.values())


def _shape_of(value: Any) -> list[int] | None:
    """A tensor's shape, or the first tensor's shape in a tuple.

    A module returning a tuple — two feature maps, a value and a mask — has no single
    output shape. Recording the first is a stated simplification rather than a guess: the
    alternative is a shape field that is sometimes a list of lists, which every renderer
    would then have to handle for the sake of a few modules.
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return _shape_of(value[0]) if value else None
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(size) for size in shape]


def probe_arguments(branch: Any, probe: Any) -> dict[str, Any]:
    """Extra keyword arguments a module needs for a dry forward pass.

    Only `image_size` so far, which EfficientAD's autoencoder requires because its decoder
    upsamples to an explicit target rather than inferring one. **Signature-driven, never
    branch-name-driven**, so a future method whose forward takes its own arguments works
    here without this function learning its name.
    """
    try:
        parameters = inspect.signature(branch.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C-implemented forward
        return {}
    return {"image_size": probe.shape[-2:]} if "image_size" in parameters else {}
