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


class DiagnosticScope(StrEnum):
    """What a diagnostic is *about*, which decides where the UI can offer it."""

    MODEL = "model"
    """One per experiment — the architecture graph, the parameter table."""

    IMAGE = "image"
    """One per image — the per-branch error maps."""


class DiagnosticEntry(BaseModel):
    """One row of the index. Self-describing on purpose."""

    model_config = API_MODEL_CONFIG

    key: str
    title: str
    kind: DiagnosticKind
    scope: DiagnosticScope
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
    image_budget: int | None = Field(
        default=None,
        description="How many images were allowed per-image diagnostics, if capped.",
    )
    truncated_images: int = Field(
        default=0,
        description="Images whose diagnostics were dropped because the budget ran out.",
    )


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
    ) -> None:
        self._root = root
        self._enabled = enabled
        self._image_budget = image_budget
        self._entries: list[DiagnosticEntry] = []
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
        """
        if self._image_budget is None or image_id in self._kept_images:
            return True
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

        relative = f"{key}.npy" if image_id is None else f"image-{image_id}/{key}.npy"
        target = self._root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, array)

        self._entries.append(
            DiagnosticEntry(
                key=key,
                title=title,
                kind=kind,
                scope=scope,
                image_id=image_id,
                path=relative,
                shape=list(array.shape),
                description=description,
            )
        )

    def flush(self) -> DiagnosticIndex:
        """Write the index. Called once, by the job handler, after the model returns.

        A model that crashed halfway has written its arrays but no index, which reads
        correctly as "this run produced no usable diagnostics" rather than as a partial
        set the UI would try to render.
        """
        index = DiagnosticIndex(
            entries=list(self._entries),
            image_budget=self._image_budget,
            truncated_images=len(self._dropped_images),
        )
        if not self._enabled:
            return index
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / INDEX_FILENAME
        path.write_text(index.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
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
