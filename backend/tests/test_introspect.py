"""Arranging module records into the architecture payload (ADR-0024).

**Torch-free on purpose.** `build_tree` takes plain records, which is what lets the
hierarchy, the bounding and the truncation reporting be checked in CI — which installs
without the `dl` extra. The half that actually runs a forward pass is exercised in
`test_dl_efficientad_introspect.py`, which skips wherever anomalib is absent.
"""

from __future__ import annotations

from typing import Any

from anomaly_lab.models.introspect import ModuleRecord, build_tree


def record(
    name: str, *, own: int = 0, total: int = 0, calls: int = 1, order: int = 0
) -> ModuleRecord:
    return ModuleRecord(
        name=name,
        type_name="Conv2d",
        own_parameters=own,
        total_parameters=total or own,
        order=order,
        calls=calls,
        input_shape=[1, 3, 8, 8] if calls else None,
        output_shape=[1, 8, 4, 4] if calls else None,
    )


def node(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return next(entry for entry in payload["nodes"] if entry["id"] == name)


def test_the_hierarchy_comes_from_the_dotted_names() -> None:
    """`named_modules()` yields every container, so the chain is always complete."""
    payload = build_tree(
        [
            record("teacher"),
            record("teacher.conv1"),
            record("teacher.body"),
            record("teacher.body.0"),
        ]
    )

    assert node(payload, "teacher")["parent"] is None
    assert node(payload, "teacher")["depth"] == 0
    assert node(payload, "teacher.conv1")["parent"] == "teacher"
    assert node(payload, "teacher.conv1")["depth"] == 1
    assert node(payload, "teacher.body.0")["parent"] == "teacher.body"
    assert node(payload, "teacher.body.0")["depth"] == 2


def test_a_gap_in_the_chain_attaches_to_the_nearest_surviving_ancestor() -> None:
    """Which is also the depth it is drawn at — not the one its full path implies."""
    payload = build_tree([record("teacher"), record("teacher.body.0")])

    assert node(payload, "teacher.body.0")["parent"] == "teacher"
    assert node(payload, "teacher.body.0")["depth"] == 1


def test_a_node_with_children_is_not_a_leaf() -> None:
    """The UI collapses on this, so getting it wrong hides a subtree behind no caret."""
    payload = build_tree([record("teacher"), record("teacher.conv1")])

    assert node(payload, "teacher")["leaf"] is False
    assert node(payload, "teacher.conv1")["leaf"] is True


def test_both_parameter_counts_are_carried() -> None:
    """One alone makes the tree not add up: a container's recursive count double-counts.

    `parameters` keeps its old meaning — the whole subtree — so the flat renderer that
    predates this prints the same number for a branch node that it always did.
    """
    payload = build_tree(
        [record("teacher", own=0, total=100), record("teacher.conv1", own=100, total=100)]
    )

    assert node(payload, "teacher")["parameters"] == 100
    assert node(payload, "teacher")["parameters_own"] == 0
    assert node(payload, "teacher.conv1")["parameters_own"] == 100


def test_a_module_that_never_ran_is_marked_rather_than_blank() -> None:
    """Different from "has no shape": a branch behind a flag is a fact about the model."""
    payload = build_tree([record("teacher"), record("teacher.unused", calls=0)])

    assert node(payload, "teacher")["executed"] is True
    assert node(payload, "teacher.unused")["executed"] is False
    assert node(payload, "teacher.unused")["input_shape"] is None


def test_a_module_called_more_than_once_reports_its_count() -> None:
    payload = build_tree([record("teacher.block", calls=4)])
    assert node(payload, "teacher.block")["calls"] == 4


def test_the_payload_carries_no_edges_of_its_own() -> None:
    """`named_modules()` sees modules, not wiring.

    Functional operations are invisible to it, so an inferred edge list would be a picture
    of connectivity nobody measured. The caller states the edges it actually knows.
    """
    payload = build_tree([record("teacher"), record("student")])
    assert payload["edges"] == []


def test_truncation_drops_the_deepest_first_and_says_how_many() -> None:
    """A silent truncation would read as "this is the whole model"."""
    records = [record("root")]
    records += [record(f"root.layer{index}", order=index) for index in range(5)]
    records += [record(f"root.layer{index}.conv", order=index) for index in range(5)]

    payload = build_tree(records, max_nodes=6)

    assert payload["truncated_nodes"] == 5
    assert payload["max_nodes"] == 6
    kept = {entry["id"] for entry in payload["nodes"]}
    assert "root" in kept
    # The top of the tree survives; the leaves are what went.
    assert all(f"root.layer{index}" in kept for index in range(5))
    assert not any(entry["id"].endswith(".conv") for entry in payload["nodes"])


def test_a_node_whose_parent_was_truncated_still_attaches() -> None:
    """Otherwise the tree silently loses a subtree it claims to have kept."""
    records = [
        record("root"),
        record("root.a"),
        record("root.a.b"),
        record("root.a.b.c"),
    ]
    # Keep root, root.a and one of the deeper two — the parent chain has a hole.
    payload = build_tree(records, max_nodes=3)
    present = {entry["id"] for entry in payload["nodes"]}

    for entry in payload["nodes"]:
        if entry["parent"] is not None:
            assert entry["parent"] in present, entry["id"]


def test_nothing_is_truncated_when_it_fits() -> None:
    payload = build_tree([record("a"), record("b")], max_nodes=100)
    assert payload["truncated_nodes"] == 0
    assert len(payload["nodes"]) == 2
