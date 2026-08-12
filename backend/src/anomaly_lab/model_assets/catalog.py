"""The small, explicit catalog of downloadable model assets.

The catalog is code rather than remote metadata: a release fixes the URL, byte count,
digest and licence that it is willing to execute.  A compromised mutable upstream file
therefore fails closed instead of becoming code-adjacent input to torch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelAssetSpec:
    key: str
    title: str
    purpose: str
    filename: str
    source_url: str
    expected_size: int
    sha256: str
    license_name: str
    license_url: str
    project_url: str


_MOBILE_SAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"

SPECS: tuple[ModelAssetSpec, ...] = (
    ModelAssetSpec(
        key="mobile-sam-vit-t",
        title="MobileSAM · TinyViT",
        purpose="Point and box guided contour suggestions in the annotation editor.",
        filename="mobile_sam.pt",
        source_url=(
            "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/"
            f"{_MOBILE_SAM_COMMIT}/weights/mobile_sam.pt"
        ),
        expected_size=40_728_226,
        sha256="6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
        license_name="Apache-2.0",
        license_url=(
            "https://github.com/ChaoningZhang/MobileSAM/blob/"
            f"{_MOBILE_SAM_COMMIT}/LICENSE"
        ),
        project_url="https://github.com/ChaoningZhang/MobileSAM",
    ),
)

_BY_KEY = {spec.key: spec for spec in SPECS}


def get_spec(key: str) -> ModelAssetSpec | None:
    return _BY_KEY.get(key)
