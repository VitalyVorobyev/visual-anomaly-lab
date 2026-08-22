"""How the diagnostics index merges (ADR-0018, handbook diagnostics.md, handbook diagnostics.md).

Torch-free, and the rules here are subtle enough that the tests are the specification.
Two of them are regression pins for bugs that were invisible on screen: an inference run
silently erasing a training run's entries, and a run's per-image sample becoming the union
of two runs' samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anomaly_lab.models.diagnostics import (
    INDEX_FILENAME,
    DiagnosticKind,
    DiagnosticOrigin,
    DiagnosticWriter,
    load_index,
)


def run_writer(root: Path, **kwargs: object) -> DiagnosticWriter:
    return DiagnosticWriter(root, **kwargs)  # type: ignore[arg-type]


def on_demand(root: Path) -> DiagnosticWriter:
    return DiagnosticWriter(root, origin=DiagnosticOrigin.ON_DEMAND)


def emit_map(writer: DiagnosticWriter, key: str, image_id: int, value: float = 1.0) -> None:
    writer.emit(
        key, key, DiagnosticKind.MAP, np.full((4, 4), value, dtype=np.float32), image_id=image_id
    )


def images_for(index: object, key: str, origin: DiagnosticOrigin) -> set[int]:
    return {
        entry.image_id
        for entry in index.entries  # type: ignore[attr-defined]
        if entry.key == key and entry.origin is origin and entry.image_id is not None
    }


# ------------------------------------------------------------------ run vs run


def test_two_runs_supersede_each_other_wholesale(tmp_path: Path) -> None:
    """The regression pin.

    Identity used to be `(key, image_id)` alone, which was invisible while the budget kept
    the first N — every run sampled the same images. Once the budget spread across the run,
    the index became the *union* of two samples: 74 images under a stated cap of 64, which
    reads as a cap that does not work, and describes a selection no run ever made.
    """
    root = tmp_path / "diag"
    first = run_writer(root)
    for image_id in (1, 2, 3):
        emit_map(first, "map_st", image_id)
    first.flush()

    second = run_writer(root)
    for image_id in (7, 8):
        emit_map(second, "map_st", image_id)
    index = second.flush()

    assert images_for(index, "map_st", DiagnosticOrigin.RUN) == {7, 8}


def test_a_run_leaves_another_key_alone(tmp_path: Path) -> None:
    """`fit` records the architecture, `predict` the error maps; each flushes this file."""
    root = tmp_path / "diag"
    training = run_writer(root)
    training.emit("teacher", "Teacher", DiagnosticKind.MAP, np.zeros((4, 4), dtype=np.float32))
    training.flush()

    inference = run_writer(root)
    emit_map(inference, "map_st", 1)
    index = inference.flush()

    assert {entry.key for entry in index.entries} == {"teacher", "map_st"}


# ------------------------------------------------------- run vs on demand (handbook diagnostics.md)


def test_an_on_demand_entry_does_not_wipe_a_run_s_sample(tmp_path: Path) -> None:
    """It is a question somebody asked, not a sample of anything."""
    root = tmp_path / "diag"
    run = run_writer(root)
    for image_id in (1, 2, 3):
        emit_map(run, "map_st", image_id)
    run.flush()

    asked = on_demand(root)
    emit_map(asked, "map_st", 99)
    index = asked.flush()

    assert images_for(index, "map_st", DiagnosticOrigin.RUN) == {1, 2, 3}
    assert images_for(index, "map_st", DiagnosticOrigin.ON_DEMAND) == {99}


def test_a_later_run_does_not_wipe_what_was_asked_for(tmp_path: Path) -> None:
    """The other direction, which is the one that would lose a reader's work."""
    root = tmp_path / "diag"
    asked = on_demand(root)
    emit_map(asked, "map_st", 99)
    asked.flush()

    run = run_writer(root)
    for image_id in (1, 2):
        emit_map(run, "map_st", image_id)
    index = run.flush()

    assert images_for(index, "map_st", DiagnosticOrigin.ON_DEMAND) == {99}
    assert images_for(index, "map_st", DiagnosticOrigin.RUN) == {1, 2}


def test_asking_twice_about_one_image_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "diag"
    on_demand(root).flush()
    for _ in range(2):
        writer = on_demand(root)
        emit_map(writer, "map_st", 42)
        index = writer.flush()

    matching = [
        entry
        for entry in index.entries
        if entry.key == "map_st" and entry.origin is DiagnosticOrigin.ON_DEMAND
    ]
    assert len(matching) == 1


def test_an_on_demand_array_lands_in_its_own_tree(tmp_path: Path) -> None:
    """So that clearing what was asked for is a directory removal, not an index walk."""
    root = tmp_path / "diag"
    writer = on_demand(root)
    emit_map(writer, "map_st", 42)
    writer.flush()

    assert (root / "on-demand" / "image-42" / "map_st.npy").is_file()
    assert not (root / "image-42").exists()


# ------------------------------------------------------------------------ ranges


def test_a_run_replaces_the_scale_for_the_keys_it_emitted(tmp_path: Path) -> None:
    root = tmp_path / "diag"
    first = run_writer(root)
    emit_map(first, "map_st", 1, value=10.0)
    first.flush()

    second = run_writer(root)
    emit_map(second, "map_st", 2, value=1.0)
    index = second.flush()

    assert index.ranges["map_st"].high == 1.0


def test_an_on_demand_emission_only_widens_the_scale(tmp_path: Path) -> None:
    """One browsed image must not re-fit a run-wide scale.

    Widening keeps every already-drawn picture correct, which is only true because the
    payload validator covers the range (handbook diagnostics.md). Narrowing would
    silently reinterpret
    every other image against a span fitted from one.
    """
    root = tmp_path / "diag"
    run = run_writer(root)
    emit_map(run, "map_st", 1, value=10.0)
    run.flush()

    cold = on_demand(root)
    emit_map(cold, "map_st", 99, value=1.0)
    assert cold.flush().ranges["map_st"].high == 10.0

    hot = on_demand(root)
    emit_map(hot, "map_st", 98, value=50.0)
    assert hot.flush().ranges["map_st"].high == 50.0


# ------------------------------------------------------------------------ budget


def test_an_on_demand_flush_carries_the_run_s_budget_forward(tmp_path: Path) -> None:
    """They are run-level facts. Rewriting them would describe a request that had none."""
    root = tmp_path / "diag"
    run = run_writer(root, image_budget=2, keep_images=[1, 2])
    for image_id in (1, 2, 3, 4):
        emit_map(run, "map_st", image_id)
    before = run.flush()
    assert (before.image_budget, before.truncated_images) == (2, 2)

    asked = on_demand(root)
    emit_map(asked, "map_st", 99)
    after = asked.flush()

    assert (after.image_budget, after.truncated_images) == (2, 2)


# ------------------------------------------------------------------- durability


def test_the_index_is_written_atomically(tmp_path: Path) -> None:
    """A crash-truncated index used to read as "this run produced no diagnostics".

    `load_index` swallows a `JSONDecodeError` and returns an empty index, which is the
    right answer for a file that was never written and the most misleading possible one
    for a file that was half written.
    """
    root = tmp_path / "diag"
    writer = run_writer(root)
    emit_map(writer, "map_st", 1)
    writer.flush()

    # Nothing is left behind, and what is there parses.
    assert not list(root.glob("*.tmp"))
    payload = json.loads((root / INDEX_FILENAME).read_text(encoding="utf-8"))
    assert payload["entries"]


def test_an_index_written_before_origins_reads_as_run(tmp_path: Path) -> None:
    """Additive, so `INDEX_VERSION` does not move — the diagnostics handbook's argument."""
    root = tmp_path / "diag"
    root.mkdir(parents=True)
    (root / INDEX_FILENAME).write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "key": "map_st",
                        "title": "ST",
                        "kind": "map",
                        "scope": "image",
                        "image_id": 5,
                        "path": "image-5/map_st.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    index = load_index(root)
    assert index.entries[0].origin is DiagnosticOrigin.RUN
