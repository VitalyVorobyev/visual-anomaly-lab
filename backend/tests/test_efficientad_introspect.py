"""`collect` against a real torch network (ADR-0024).

**The first `dl`-gated test in this suite.** CI installs without `--extra dl` and must stay
that way — the torch-free boundary is measured rather than asserted (see the M3 summary in
`docs/roadmap.md`) — so this skips there and runs locally under `uv sync --extra dl`.

It builds anomalib's PDN directly rather than a whole `EfficientAd` module: the point is
that the shared helper walks a real network with real shapes, and doing that needs neither
the 40 MB teacher download nor the 1.5 GB penalty set.
"""

from __future__ import annotations

import pytest

from anomaly_lab.models.introspect import build_tree, collect

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomalib.models.image.efficient_ad.torch_model import (  # noqa: E402
    SmallPatchDescriptionNetwork,
)


@pytest.fixture
def network() -> object:
    return SmallPatchDescriptionNetwork(out_channels=384).eval()


def test_every_module_is_recorded_not_only_the_branch(network: object) -> None:
    """The whole point: three cards became a tree with the layers actually in it."""
    probe = torch.zeros(1, 3, 64, 64)
    records = collect(network, probe, prefix="teacher.")

    assert len(records) > 3
    assert any(record.name == "teacher" for record in records)
    assert any(record.name.startswith("teacher.") for record in records)


def test_executed_leaves_carry_real_shapes(network: object) -> None:
    probe = torch.zeros(1, 3, 64, 64)
    payload = build_tree(collect(network, probe, prefix="teacher."))
    leaves = [node for node in payload["nodes"] if node["leaf"] and node["executed"]]

    assert leaves
    for node in leaves:
        assert node["input_shape"] is not None, node["id"]
        assert node["output_shape"] is not None, node["id"]
        # Batch dimension first, as the probe was given.
        assert node["input_shape"][0] == 1


def test_the_recorded_shapes_are_the_ones_the_network_produces(network: object) -> None:
    """Read off the running model, so they cannot go stale the way a diagram does."""
    probe = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        expected = list(network(probe).shape)

    payload = build_tree(collect(network, probe, prefix="teacher."))
    root = next(node for node in payload["nodes"] if node["id"] == "teacher")

    assert root["input_shape"] == [1, 3, 64, 64]
    assert root["output_shape"] == expected


def test_parameter_counts_add_up_across_the_tree(network: object) -> None:
    """`parameters_own` summed over every node equals the model's total.

    The check that catches the obvious mistake: reporting only the recursive count makes a
    container double-count against its children, so the column would not sum to anything.
    """
    payload = build_tree(collect(network, torch.zeros(1, 3, 64, 64), prefix="teacher."))
    total = sum(int(p.numel()) for p in network.parameters())

    assert sum(node["parameters_own"] for node in payload["nodes"]) == total
    root = next(node for node in payload["nodes"] if node["id"] == "teacher")
    assert root["parameters"] == total


def test_collect_leaves_no_hooks_behind(network: object) -> None:
    """A hook left on a module about to be trained runs on every step of the real loop."""
    collect(network, torch.zeros(1, 3, 64, 64), prefix="teacher.")

    for _, module in network.named_modules():
        assert not module._forward_hooks, module


def test_a_hook_removal_survives_a_failing_forward(network: object) -> None:
    """The `finally` matters most exactly when the probe was the wrong shape."""
    # A one-channel probe into a three-channel conv: torch raises RuntimeError.
    with pytest.raises(RuntimeError):
        collect(network, torch.zeros(1, 1, 8, 8), prefix="teacher.")

    for _, module in network.named_modules():
        assert not module._forward_hooks, module
