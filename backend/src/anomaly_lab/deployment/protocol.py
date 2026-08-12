"""The method-owned half of the generic portable-export boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from anomaly_lab.models.preprocessing import PreprocessingConfig


class OnnxGraphContract(BaseModel):
    """What the generic bundler needs to execute and describe an exported graph."""

    opset: int
    input_name: str
    output_name: str
    score_percentile: float


@runtime_checkable
class SupportsOnnxExport(Protocol):
    """A fitted method that can write its map computation as ONNX (ADR-0034)."""

    def export_onnx(
        self,
        destination: Path,
        preprocessing: PreprocessingConfig,
    ) -> OnnxGraphContract:
        """Write a fitted, static-shape graph and return its public tensor contract."""
        ...

    def portable_reference(self, input_nchw: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the fitted Python semantics on an already-prepared fixture tensor."""
        ...
