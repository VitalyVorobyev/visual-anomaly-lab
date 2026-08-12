"""The version-two portable deployment manifest.

This is deliberately more explicit than an ONNX graph. The graph computes a map; this
contract says which pixels enter it, what its tensors mean, how the image score is
resolved, and which immutable experiment produced it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from anomaly_lab.models.base import PortableFormat
from anomaly_lab.models.preprocessing import ColorMode
from anomaly_lab.schemas import API_MODEL_CONFIG

BUNDLE_FORMAT_VERSION = 2
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TensorDtype(StrEnum):
    FLOAT32 = "float32"


class TensorLayout(StrEnum):
    NCHW = "NCHW"


class CoordinateFrame(StrEnum):
    PREPARED = "prepared"


class TensorSpec(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str
    dtype: TensorDtype
    layout: TensorLayout
    shape: list[int]


class ScalarTensorSpec(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str
    dtype: TensorDtype
    shape: list[int] = Field(default_factory=lambda: [1])


class PixelInputContract(BaseModel):
    model_config = API_MODEL_CONFIG

    coordinate_frame: CoordinateFrame = CoordinateFrame.PREPARED
    color: ColorMode
    value_min: float = 0.0
    value_max: float = 1.0
    width: int
    height: int
    tensor: TensorSpec
    method_normalization: str = "inside_graph"


class MapOutputContract(BaseModel):
    model_config = API_MODEL_CONFIG

    coordinate_frame: CoordinateFrame = CoordinateFrame.PREPARED
    tensor: TensorSpec
    higher_is_more_anomalous: bool = True


class PercentileReducer(BaseModel):
    model_config = API_MODEL_CONFIG

    kind: Literal["percentile_linear"] = "percentile_linear"
    percentile: float = Field(ge=0.0, le=100.0)


class MaxReducer(BaseModel):
    model_config = API_MODEL_CONFIG

    kind: Literal["max"] = "max"


class TopKMeanReducer(BaseModel):
    model_config = API_MODEL_CONFIG

    kind: Literal["top_k_mean"] = "top_k_mean"
    top_k: int = Field(ge=1)


class TensorScore(BaseModel):
    model_config = API_MODEL_CONFIG

    kind: Literal["tensor"] = "tensor"
    tensor: ScalarTensorSpec


ScoreContract = Annotated[
    PercentileReducer | MaxReducer | TopKMeanReducer | TensorScore,
    Field(discriminator="kind"),
]


class OperatingPointContract(BaseModel):
    model_config = API_MODEL_CONFIG

    rule: str
    value: float
    subset: str | None = None


class RegionContract(BaseModel):
    model_config = API_MODEL_CONFIG

    profile_revision_id: int
    manifest_sha256: str
    runtime_input_is_prepared: bool = True

    @field_validator("manifest_sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        return value


class FileDigest(BaseModel):
    model_config = API_MODEL_CONFIG

    path: str
    bytes: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or not value or ".." in path.parts or "\\" in value:
            raise ValueError("bundle file paths must be safe POSIX-relative paths")
        return value

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class ParityFixture(BaseModel):
    model_config = API_MODEL_CONFIG

    input_path: str
    expected_map_path: str
    expected_score: float
    absolute_tolerance: float = Field(gt=0.0)
    relative_tolerance: float = Field(ge=0.0)
    max_absolute_error: float = Field(ge=0.0)
    score_absolute_error: float = Field(ge=0.0)

    @field_validator("input_path", "expected_map_path")
    @classmethod
    def _safe_fixture_path(cls, value: str) -> str:
        return FileDigest._safe_relative_path(value)


class SourceExperiment(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    model_type: str
    method_config: dict[str, object]
    preprocessing: dict[str, object]


class DeploymentManifest(BaseModel):
    """Everything a non-Python consumer needs to reproduce one fitted method."""

    model_config = API_MODEL_CONFIG

    format_version: int = BUNDLE_FORMAT_VERSION
    portable_format: PortableFormat = PortableFormat.ONNX
    created_at: str
    source: SourceExperiment
    graph_path: str
    opset: int
    input: PixelInputContract
    anomaly_map: MapOutputContract
    score: ScoreContract
    operating_point: OperatingPointContract | None = None
    region: RegionContract
    runtime: str = "ONNX Runtime compatible"
    files: list[FileDigest] = Field(default_factory=list)
    parity: ParityFixture

    @field_validator("graph_path")
    @classmethod
    def _safe_graph_path(cls, value: str) -> str:
        return FileDigest._safe_relative_path(value)
