"""Clearing diagnostics to reclaim disk (ADR-0027).

Torch-free. The interesting cases are all about what *survives*: a clear that also took
the architecture tree would read as the Architecture tab breaking, and a clear that walked
`entry.path` would leave a crashed run's unreferenced arrays behind for ever — which are
exactly the bytes somebody clearing disk space is trying to recover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anomaly_lab.models.diagnostics import (
    INDEX_FILENAME,
    DiagnosticKind,
    DiagnosticOrigin,
    DiagnosticScope,
    DiagnosticWriter,
    PruneScope,
    load_index,
    prune,
)


def populated(root: Path) -> None:
    """One run's sample, one model-scoped entry, and one image asked about later."""
    run = DiagnosticWriter(root)
    run.emit("teacher", "Teacher", DiagnosticKind.MAP, np.zeros((4, 4), dtype=np.float32))
    run.emit("layers", "Layers", DiagnosticKind.TABLE, {"columns": ["name"], "rows": [["conv1"]]})
    for image_id in (1, 2):
        run.emit(
            "map_st", "ST", DiagnosticKind.MAP, np.ones((4, 4), dtype=np.float32), image_id=image_id
        )
    run.flush()

    asked = DiagnosticWriter(root, origin=DiagnosticOrigin.ON_DEMAND)
    asked.emit("map_st", "ST", DiagnosticKind.MAP, np.ones((4, 4), dtype=np.float32), image_id=99)
    asked.flush()


def keys_by_scope(root: Path, scope: DiagnosticScope) -> set[str]:
    return {entry.key for entry in load_index(root).entries if entry.scope is scope}


# ------------------------------------------------------------------------- scopes


def test_clearing_images_keeps_what_the_other_tabs_draw(tmp_path: Path) -> None:
    """The default scope. Model-scoped entries are kilobytes and are the Architecture tab."""
    root = tmp_path / "diagnostics"
    populated(root)

    result = prune(root, scope=PruneScope.IMAGE)

    assert keys_by_scope(root, DiagnosticScope.IMAGE) == set()
    assert keys_by_scope(root, DiagnosticScope.MODEL) == {"teacher", "layers"}
    assert not (root / "image-1").exists()
    assert not (root / "on-demand").exists()
    assert (root / "teacher.npy").is_file()
    assert result.removed_entries == 3
    assert result.removed_files == 3


def test_clearing_on_demand_leaves_the_run_s_own_sample(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    populated(root)

    result = prune(root, scope=PruneScope.ON_DEMAND)

    index = load_index(root)
    origins = {(entry.key, entry.image_id, entry.origin) for entry in index.entries}
    assert ("map_st", 1, DiagnosticOrigin.RUN) in origins
    assert ("map_st", 99, DiagnosticOrigin.ON_DEMAND) not in origins
    assert (root / "image-1" / "map_st.npy").is_file()
    assert not (root / "on-demand").exists()
    assert result.removed_entries == 1


def test_clearing_everything_leaves_no_index_at_all(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    populated(root)

    result = prune(root, scope=PruneScope.ALL)

    assert not root.exists()
    assert load_index(root).entries == []
    assert result.remaining_bytes == 0


# -------------------------------------------------------------------------- ranges


def test_a_scale_nobody_references_goes_with_its_entries(tmp_path: Path) -> None:
    """Left behind, it would re-colour the next run of that key against absent data."""
    root = tmp_path / "diagnostics"
    run = DiagnosticWriter(root)
    run.emit("map_st", "ST", DiagnosticKind.MAP, np.full((4, 4), 7.0, dtype=np.float32), image_id=1)
    run.emit("teacher", "Teacher", DiagnosticKind.MAP, np.zeros((4, 4), dtype=np.float32))
    run.flush()
    assert set(load_index(root).ranges) == {"map_st", "teacher"}

    prune(root, scope=PruneScope.IMAGE)

    assert set(load_index(root).ranges) == {"teacher"}


# ------------------------------------------------------------------- what is measured


def test_an_orphaned_array_is_reclaimed_even_though_no_entry_names_it(tmp_path: Path) -> None:
    """A run that crashed after writing arrays and before flushing leaves exactly this.

    The reason this deletes directories rather than the paths the index lists: those bytes
    are unreferenced, invisible, and are what a disk clear is for.
    """
    root = tmp_path / "diagnostics"
    populated(root)
    orphan = root / "image-7" / "map_st.npy"
    orphan.parent.mkdir(parents=True)
    np.save(orphan, np.zeros((64, 64), dtype=np.float32))

    result = prune(root, scope=PruneScope.IMAGE)

    assert not (root / "image-7").exists()
    assert result.removed_files == 4
    # Not counted as an entry: the index never knew about it.
    assert result.removed_entries == 3


def test_the_reported_bytes_are_the_measured_delta(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    populated(root)
    before = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    result = prune(root, scope=PruneScope.IMAGE)

    after = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    assert result.remaining_bytes == after
    assert result.bytes_reclaimed == before - after
    assert result.bytes_reclaimed > 0


def test_pruning_nothing_is_not_an_error(tmp_path: Path) -> None:
    """The clear button exists before anything has been recorded."""
    result = prune(tmp_path / "never-written", scope=PruneScope.ALL)

    assert (result.removed_entries, result.removed_files, result.bytes_reclaimed) == (0, 0, 0)


# ---------------------------------------------------------------------- durability


def test_the_rewritten_index_is_written_atomically(tmp_path: Path) -> None:
    """Same guarantee as `flush`, and now the same implementation."""
    root = tmp_path / "diagnostics"
    populated(root)

    prune(root, scope=PruneScope.IMAGE)

    assert not list(root.glob("*.tmp"))
    assert (root / INDEX_FILENAME).is_file()
