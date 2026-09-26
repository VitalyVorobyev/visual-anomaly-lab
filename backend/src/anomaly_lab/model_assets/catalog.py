"""The small, explicit catalog of downloadable model assets.

The catalog is code rather than remote metadata: a release fixes the URL, byte count,
digest and licence that it is willing to execute.  A compromised mutable upstream file
therefore fails closed instead of becoming code-adjacent input to torch.

An asset is one main file, optionally with **companions** — the small configuration and
tokenizer files a Hugging Face checkpoint cannot be loaded without. Every one of them is
pinned by size and SHA-256 and lives in the same directory, so a directory that holds the
main file and not its companions is not ready. An asset with a `hub` pin is fetched from
that repository at that immutable revision through `huggingface_hub`, which carries the
account token a gated repository needs; the others stream from `source_url`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetFile:
    """A companion file, pinned exactly as the main file is."""

    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class HubPin:
    """A Hugging Face repository at one immutable revision."""

    repository: str
    revision: str
    access_url: str | None = None
    """Where an account asks for a gated repository; `None` when the repository is open."""


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
    companions: tuple[AssetFile, ...] = ()
    hub: HubPin | None = None

    @property
    def gated(self) -> bool:
        return self.hub is not None and self.hub.access_url is not None

    @property
    def total_size(self) -> int:
        return self.expected_size + sum(entry.size for entry in self.companions)


_MOBILE_SAM_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
_SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

SAM3_KEY = "sam3"

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
            f"https://github.com/ChaoningZhang/MobileSAM/blob/{_MOBILE_SAM_COMMIT}/LICENSE"
        ),
        project_url="https://github.com/ChaoningZhang/MobileSAM",
    ),
    ModelAssetSpec(
        key=SAM3_KEY,
        title="SAM 3",
        purpose="Instance masks for a text phrase in the sample viewer's Explore.",
        filename="model.safetensors",
        source_url=(
            f"https://huggingface.co/facebook/sam3/blob/{_SAM3_REVISION}/model.safetensors"
        ),
        expected_size=3_439_938_512,
        sha256="6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
        license_name="SAM License",
        license_url=f"https://huggingface.co/facebook/sam3/blob/{_SAM3_REVISION}/LICENSE",
        project_url="https://github.com/facebookresearch/sam3",
        companions=(
            AssetFile(
                "config.json",
                25_843,
                "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
            ),
            AssetFile(
                "processor_config.json",
                1_712,
                "6420cf2671fa9309ea95bc0144a8b9861666d1c5f43c8db09e410dacda974fce",
            ),
            AssetFile(
                "tokenizer.json",
                3_642_073,
                "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
            ),
            AssetFile(
                "tokenizer_config.json",
                799,
                "39670ad98457fe8f14ca59f6bb74591e9fc850974380a63993e5b8ffc865baa2",
            ),
            AssetFile(
                "special_tokens_map.json",
                588,
                "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e",
            ),
            AssetFile(
                "vocab.json",
                862_328,
                "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
            ),
            AssetFile(
                "merges.txt",
                524_619,
                "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
            ),
        ),
        hub=HubPin(
            repository="facebook/sam3",
            revision=_SAM3_REVISION,
            access_url="https://huggingface.co/facebook/sam3",
        ),
    ),
)

_BY_KEY = {spec.key: spec for spec in SPECS}


def get_spec(key: str) -> ModelAssetSpec | None:
    return _BY_KEY.get(key)


def gated_message(spec: ModelAssetSpec, exc: Exception | None = None) -> str:
    """What unblocks a gated asset, in words — the token itself is never read or printed.

    The same three sentences a gated DINOv3 encoder gives (`dino_backbone`): the licence,
    where to ask for access, and that an approved account's HF_TOKEN must be set.
    """
    access = spec.hub.access_url if spec.hub is not None else None
    text = (
        f"{spec.title}'s weights are gated under the {spec.license_name}: access must be "
        f"requested from the publisher on Hugging Face at {access}, then a valid HF_TOKEN "
        "environment variable must be set for the account that was approved (or that "
        "account signed in with `hf auth login`)."
    )
    return f"{text} The underlying failure was: {exc}" if exc is not None else text
