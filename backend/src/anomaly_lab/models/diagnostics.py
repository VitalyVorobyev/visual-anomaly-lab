"""Model diagnostics: what a method shows about itself.

A score and a heatmap say whether a model works. They do not say *why*, and "why" is the
whole point of a research workbench — EfficientAD's student-teacher error and its
autoencoder error fail in different ways, and seeing them apart is how the failure is
understood.

The contract is deliberately **declarative and model-agnostic**:

  * a model declares `produces_diagnostics` in its `Capabilities`;
  * during `fit` or `predict` it calls `ctx.emit_diagnostic(key, title, kind, payload)`;
  * arrays land as float32 `.npy` beside the other artifacts, JSON payloads land inline;
  * an index file describes everything that was written.

The UI reads the index and renders by `kind`. **It never branches on model name.** That
is what makes M4's visualization work unchanged for `efficientad_custom` in M6, and it is
why the index is self-describing rather than a schema the frontend has to know in advance.

Nothing here needs a migration: diagnostics live entirely under the experiment's artifact
directory and are deleted with it (ADR-0004).

Scalar series — per-epoch losses, learning rate — deliberately do **not** come through
here. They are already `metric` events in the job protocol (ADR-0009), and inventing a
second channel for the same data would be the wrong kind of completeness.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.schemas import API_MODEL_CONFIG

INDEX_FILENAME = "diagnostics.json"
INDEX_VERSION = 1

# Keys become path segments, so they are constrained rather than trusted.
_SAFE_KEY = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


class DiagnosticKind(StrEnum):
    """How a payload should be read. The UI's renderer switch is over exactly this."""

    MAP = "map"
    """2-D float32 array — rendered as a heatmap, e.g. student-teacher error."""

    IMAGE = "image"
    """`(H, W, 3)` float32 in [0, 1] — rendered as-is, e.g. a PCA-to-RGB composite."""

    GRID = "grid"
    """`(N, H, W)` float32 — small multiples, e.g. per-channel teacher features."""

    GRAPH = "graph"
    """JSON `{nodes, edges}` — the architecture view, captured from a real forward pass."""

    TABLE = "table"
    """JSON `{columns, rows}` — anything that is honestly a table, e.g. layer shapes."""


class DiagnosticOrigin(StrEnum):
    """Whether an entry came from a run, or from someone asking about one image.

    Additive, so `INDEX_VERSION` does not move — the same argument handbook diagnostics.md made for
    `ranges`. It exists because the two have genuinely different lifetimes and different
    supersession rules: a run's per-image entries are a *sample* of that run and replace
    each other wholesale, while an on-demand entry is a question somebody asked and must
    not be swept away by the next run of inference, nor sweep one away (handbook diagnostics.md).
    """

    RUN = "run"
    ON_DEMAND = "on_demand"


class DiagnosticScope(StrEnum):
    """What a diagnostic is *about*, which decides where the UI can offer it."""

    MODEL = "model"
    """One per experiment — the architecture graph, the parameter table."""

    IMAGE = "image"
    """One per image — the per-branch error maps."""


class DisplayRange(BaseModel):
    """The value span one diagnostic key is drawn over, across a whole run.

    Same shape as the `range.json` an inference job writes beside its anomaly maps, and
    for the same reason: every emission of one key has to be colormapped on a single
    scale or two images cannot be compared by eye. Normalizing each array to its own
    extremes makes a clean part look exactly as alarming as a defective one.
    """

    model_config = API_MODEL_CONFIG

    low: float
    high: float


class DiagnosticEntry(BaseModel):
    """One row of the index. Self-describing on purpose."""

    model_config = API_MODEL_CONFIG

    key: str
    title: str
    kind: DiagnosticKind
    scope: DiagnosticScope
    origin: DiagnosticOrigin = DiagnosticOrigin.RUN
    """Where this entry came from (handbook diagnostics.md).

    A literal default is correct here, unlike on a request body: this is a *response*
    model, so "required in TypeScript" means the client reads it without a null check.
    """
    image_id: int | None = None
    path: str | None = Field(
        default=None,
        description="Array payload, relative to the diagnostics directory.",
    )
    payload: dict[str, Any] | None = Field(
        default=None,
        description="Inline JSON payload for the graph and table kinds.",
    )
    shape: list[int] | None = None
    description: str | None = None


class DiagnosticIndex(BaseModel):
    model_config = API_MODEL_CONFIG

    version: int = INDEX_VERSION
    entries: list[DiagnosticEntry] = Field(default_factory=list)
    ranges: dict[str, DisplayRange] = Field(
        default_factory=dict,
        description=(
            "Run-wide value span per colormapped key, so every emission of one key is "
            "drawn on the same scale. Absent for keys rendered without a colormap, and "
            "absent entirely from an index written before ranges were recorded."
        ),
    )
    image_budget: int | None = Field(
        default=None,
        description="How many images were allowed per-image diagnostics, if capped.",
    )
    truncated_images: int = Field(
        default=0,
        description="Images whose diagnostics were dropped because the budget ran out.",
    )


class PruneScope(StrEnum):
    """How much of a run's diagnostics to delete (handbook diagnostics.md)."""

    IMAGE = "image"
    """Every per-image entry, of both origins. The default: it is the bulk of the disk."""

    ON_DEMAND = "on_demand"
    """Only what somebody asked for while browsing. Leaves the run's own sample intact."""

    ALL = "all"
    """The whole directory, model-scoped entries included."""


class PruneResult(BaseModel):
    """What a clear actually reclaimed. Reported, never assumed."""

    model_config = API_MODEL_CONFIG

    removed_entries: int
    removed_files: int
    bytes_reclaimed: int
    remaining_bytes: int


class DiagnosticError(Exception):
    """A model emitted something the contract cannot represent — a bug in the plugin."""


class DiagnosticWriter:
    """Collects a run's diagnostics and writes the index.

    Constructed disabled when the model declares it produces none, or when the operator
    turned them off for a long run. A disabled writer accepts every call and does
    nothing, so **a model never has to ask whether diagnostics are wanted** — which is
    the difference between a capability and a conditional littered through the plugin.
    """

    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = True,
        image_budget: int | None = None,
        keep_images: Iterable[int] | None = None,
        origin: DiagnosticOrigin = DiagnosticOrigin.RUN,
    ) -> None:
        self._root = root
        self._enabled = enabled
        self._origin = origin
        self._image_budget = image_budget
        self._entries: list[DiagnosticEntry] = []
        self._ranges: dict[str, DisplayRange] = {}
        self._chosen: set[int] | None = None if keep_images is None else set(keep_images)
        self._kept_images: set[int] = set()
        self._dropped_images: set[int] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def root(self) -> Path:
        return self._root

    def _within_budget(self, image_id: int) -> bool:
        """Whether this image still gets diagnostics.

        Per-image maps are the bulk of the storage — three float32 maps per image over a
        few hundred images is hundreds of megabytes — so a run caps how many images keep
        them. The cap is enforced here rather than in each plugin, so a model calls
        `emit_diagnostic` unconditionally and the budget stays one decision in one place.
        What was dropped is counted and written into the index; a silent truncation would
        read as "this is all there was".

        **Which images are kept is chosen up front, not by arrival order.** Filling the
        budget with the first N the model happens to reach keeps N neighbours from one
        corner of the dataset: on the reference run that was images 151-162 out of a scored
        range of 151-1249, twelve consecutive files, presented as a sample of the run. The
        caller passes the set — `evenly_spaced` over the images it is about to score — and
        this becomes a membership test. Without a chosen set the old behaviour remains, so
        a caller that does not know its image list in advance still gets a bounded run.
        """
        if self._image_budget is None or image_id in self._kept_images:
            return True
        if self._chosen is not None:
            if image_id in self._chosen:
                self._kept_images.add(image_id)
                return True
            self._dropped_images.add(image_id)
            return False
        if len(self._kept_images) < self._image_budget:
            self._kept_images.add(image_id)
            return True
        self._dropped_images.add(image_id)
        return False

    def emit(
        self,
        key: str,
        title: str,
        kind: DiagnosticKind,
        payload: np.ndarray | dict[str, Any],
        *,
        image_id: int | None = None,
        description: str | None = None,
    ) -> None:
        """Record one diagnostic. Silently does nothing when disabled."""
        if not self._enabled:
            return
        if image_id is not None and not self._within_budget(image_id):
            return
        if not _SAFE_KEY.match(key):
            msg = (
                f"diagnostic key {key!r} is not a lowercase, underscore-separated "
                "identifier; keys become file names"
            )
            raise DiagnosticError(msg)

        scope = DiagnosticScope.IMAGE if image_id is not None else DiagnosticScope.MODEL

        if kind in {DiagnosticKind.GRAPH, DiagnosticKind.TABLE}:
            if not isinstance(payload, dict):
                msg = f"diagnostic {key!r} of kind {kind.value} needs a dict payload"
                raise DiagnosticError(msg)
            self._entries.append(
                DiagnosticEntry(
                    key=key,
                    title=title,
                    kind=kind,
                    scope=scope,
                    origin=self._origin,
                    image_id=image_id,
                    payload=payload,
                    description=description,
                )
            )
            return

        if not isinstance(payload, np.ndarray):
            msg = f"diagnostic {key!r} of kind {kind.value} needs an ndarray payload"
            raise DiagnosticError(msg)

        array = np.ascontiguousarray(payload, dtype=np.float32)
        _check_rank(key, kind, array)
        self._widen_range(key, kind, array)

        relative = f"{key}.npy" if image_id is None else f"image-{image_id}/{key}.npy"
        if self._origin is DiagnosticOrigin.ON_DEMAND and image_id is not None:
            # Its own tree, so "clear what I asked for" is a directory removal rather than
            # a walk that has to tell two runs' files apart by reading the index.
            relative = f"on-demand/image-{image_id}/{key}.npy"
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array)

        self._entries.append(
            DiagnosticEntry(
                key=key,
                title=title,
                kind=kind,
                scope=scope,
                origin=self._origin,
                image_id=image_id,
                path=relative,
                shape=list(array.shape),
                description=description,
            )
        )

    def _widen_range(self, key: str, kind: DiagnosticKind, array: np.ndarray) -> None:
        """Grow this key's run-wide display range to include one more emission.

        Only the colormapped kinds. An `image` payload is `(H, W, 3)` in `[0, 1]` and is
        rendered as-is, so a range for it would be recorded, never read, and misleading
        to anyone who found it.

        The high end is the 99.9th percentile rather than the maximum, the same choice
        `InferContext.write_map` makes: one hot pixel in one image would otherwise flatten
        every other emission of the key to black.
        """
        if kind not in {DiagnosticKind.MAP, DiagnosticKind.GRID}:
            return
        low = float(array.min())
        high = float(np.percentile(array, 99.9))
        current = self._ranges.get(key)
        if current is None:
            self._ranges[key] = DisplayRange(low=low, high=high)
            return
        self._ranges[key] = DisplayRange(
            low=min(current.low, low),
            high=max(current.high, high),
        )

    def _merged_ranges(self, existing: DiagnosticIndex) -> dict[str, DisplayRange]:
        """How a key's display scale survives another emission.

        A **run** replaces the keys it emitted and carries the rest forward: a key belongs
        to whichever run emitted it, and recomputing a union across both would widen
        `predict`'s per-image error maps to include `fit`'s teacher magnitudes, which are
        a different quantity.

        An **on-demand** emission may only *widen*. It is one array, and letting one array
        re-fix a run-wide scale is exactly the failure the diagnostics handbook warns
        about for a
        cancelled run — except that here it would happen every time somebody browsed a hot
        image. Widening keeps every previously drawn picture correct, which is only true
        because the payload ETag now covers the range (handbook diagnostics.md).
        """
        if self._origin is DiagnosticOrigin.RUN:
            merged = {
                key: value for key, value in existing.ranges.items() if key not in self._ranges
            }
            merged.update(self._ranges)
            return merged

        merged = dict(existing.ranges)
        for key, value in self._ranges.items():
            current = merged.get(key)
            merged[key] = (
                value
                if current is None
                else DisplayRange(
                    low=min(current.low, value.low), high=max(current.high, value.high)
                )
            )
        return merged

    def flush(self) -> DiagnosticIndex:
        """Write the index. Called once, by the job handler, after the model returns.

        **Merged with what is already there, not replaced.** An experiment's diagnostics
        come from two runs — `fit` records the architecture and what the teacher sees,
        `predict` records the per-image error maps — and each writes this file when it
        ends. Replacing it made the inference run silently erase the training run's
        entries: the arrays stayed on disk, unreferenced, and the architecture view had
        nothing to draw. Identity is `(key, image_id)`, so re-running inference replaces
        its own entries and leaves training's alone.

        A model that crashed halfway has written its arrays but no index entry for them,
        which reads correctly as "that run produced no usable diagnostics".

        **The rules below are scoped by origin (handbook diagnostics.md).** A run's
        per-image entries are
        that run's *sample* and supersede each other wholesale; an on-demand entry is a
        question somebody asked about one image. Neither may sweep the other away, so
        every rule here applies within an origin and never across one.
        """
        if not self._enabled:
            return DiagnosticIndex(
                entries=list(self._entries),
                ranges=dict(self._ranges),
                image_budget=self._image_budget,
                truncated_images=len(self._dropped_images),
            )

        existing = load_index(self._root)
        superseded = {(entry.key, entry.image_id, entry.origin) for entry in self._entries}
        # A key emitted per-image this run replaces **every** per-image entry under that
        # key, not merely the images this run happened to sample.
        #
        # Identity was `(key, image_id)` alone, which was invisible while the budget kept
        # the first N: every run sampled the same images, so every entry was superseded.
        # Once the budget spread across the run, two runs sampled different images and the
        # index became their union — 74 images under a stated budget of 64, which reads as
        # a cap that does not work. A union is also nobody's sample: half of it describes a
        # selection the current run did not make.
        #
        # It applies to run entries only, and only when *this* writer is a run: an
        # on-demand emission is not a sample of anything, so it must neither wipe a run's
        # sample nor be wiped by the next one.
        run_writer = self._origin is DiagnosticOrigin.RUN
        resampled = {entry.key for entry in self._entries if entry.image_id is not None}
        kept = [
            entry
            for entry in existing.entries
            if (entry.key, entry.image_id, entry.origin) not in superseded
            and not (
                run_writer
                and entry.origin is DiagnosticOrigin.RUN
                and entry.image_id is not None
                and entry.key in resampled
            )
        ]
        ranges = self._merged_ranges(existing)
        index = DiagnosticIndex(
            entries=[*kept, *self._entries],
            ranges=ranges,
            # Run-level facts, so an on-demand flush carries them forward rather than
            # rewriting them to describe a request that had no budget in the first place.
            image_budget=self._image_budget if run_writer else existing.image_budget,
            truncated_images=(
                len(self._dropped_images) if run_writer else existing.truncated_images
            ),
        )

        write_index(self._root, index)
        return index


_EXPECTED_RANK = {
    DiagnosticKind.MAP: (2,),
    DiagnosticKind.IMAGE: (3,),
    DiagnosticKind.GRID: (3,),
}


def _check_rank(key: str, kind: DiagnosticKind, array: np.ndarray) -> None:
    expected = _EXPECTED_RANK.get(kind)
    if expected is not None and array.ndim not in expected:
        wanted = " or ".join(f"{rank}-D" for rank in expected)
        msg = f"diagnostic {key!r} of kind {kind.value} must be {wanted}, got {array.ndim}-D"
        raise DiagnosticError(msg)
    if kind is DiagnosticKind.IMAGE and array.shape[-1] != 3:
        msg = f"diagnostic {key!r} of kind image must end in 3 channels, got {array.shape}"
        raise DiagnosticError(msg)


def load_index(root: Path) -> DiagnosticIndex:
    """Read a run's index, or an empty one if it produced none."""
    path = root / INDEX_FILENAME
    if not path.is_file():
        return DiagnosticIndex()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DiagnosticIndex()
    return DiagnosticIndex.model_validate(payload)


def write_index(root: Path, index: DiagnosticIndex) -> None:
    """Replace a run's index, atomically.

    Written to a sibling and renamed. `load_index` swallows a `JSONDecodeError` and
    returns an *empty* index, so a write truncated by a crash used to read as "this run
    produced no diagnostics" — the most misleading failure available here. `Path.replace`
    is atomic within a filesystem, which this always is.

    Shared by `flush` and `prune` so that guarantee has one implementation: a clear that
    half-wrote the index would be the same failure with a different cause.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / INDEX_FILENAME
    staging = path.with_suffix(".json.tmp")
    staging.write_text(index.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    staging.replace(path)


def prune(root: Path, *, scope: PruneScope = PruneScope.IMAGE) -> PruneResult:
    """Delete some of a run's diagnostics and report what that reclaimed (handbook diagnostics.md).

    **Directories are removed, not the files the index names.** A run that crashed
    part-way wrote its arrays and no index entry for them; walking `entry.path` would
    leave those behind for ever, and they are exactly the bytes somebody clearing disk
    space is trying to recover.

    `image` keeps model-scoped entries — the architecture tree, the score-normalization
    table, what the teacher sees. They are a few kilobytes, they are what the Architecture
    and Inspector tabs draw, and losing them to a routine disk clear would read as those
    tabs breaking.

    **`maps/` is not touched, at any scope.** It is a sibling of this directory, and an
    anomaly map is referenced by `ImageResult.map_path`: deleting one orphans a database
    row and silently breaks the overlay, `has_map` and the map scale. Reclaiming that
    space means deleting the experiment.
    """
    if not root.is_dir():
        return PruneResult(removed_entries=0, removed_files=0, bytes_reclaimed=0, remaining_bytes=0)

    before = _tree_bytes(root)
    index = load_index(root)

    if scope is PruneScope.ALL:
        kept: list[DiagnosticEntry] = []
    elif scope is PruneScope.IMAGE:
        kept = [entry for entry in index.entries if entry.scope is not DiagnosticScope.IMAGE]
    else:
        kept = [entry for entry in index.entries if entry.origin is not DiagnosticOrigin.ON_DEMAND]

    removed_files = 0
    if scope is PruneScope.ALL:
        removed_files += _remove_tree(root)
    else:
        removed_files += _remove_tree(root / "on-demand")
        if scope is PruneScope.IMAGE:
            for child in sorted(root.glob("image-*")):
                removed_files += _remove_tree(child)

    if scope is not PruneScope.ALL:
        surviving = {entry.key for entry in kept}
        write_index(
            root,
            DiagnosticIndex(
                version=index.version,
                entries=kept,
                # A scale describes a key nothing references any more. Left behind it
                # would silently re-colour the next run of that key against a span fitted
                # from data that is gone.
                ranges={key: value for key, value in index.ranges.items() if key in surviving},
                image_budget=index.image_budget,
                truncated_images=index.truncated_images,
            ),
        )

    remaining = _tree_bytes(root)
    return PruneResult(
        removed_entries=len(index.entries) - len(kept),
        removed_files=removed_files,
        # The measured delta, so the rewritten index is accounted for rather than assumed
        # away. What is reported is what the file manager beside it will agree with.
        bytes_reclaimed=max(before - remaining, 0),
        remaining_bytes=remaining,
    )


def _tree_bytes(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _remove_tree(path: Path) -> int:
    """Remove a directory and return how many files went with it."""
    if not path.is_dir():
        return 0
    count = sum(1 for child in path.rglob("*") if child.is_file())
    shutil.rmtree(path)
    return count
