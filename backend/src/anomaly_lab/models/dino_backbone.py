"""The frozen self-supervised encoder an in-house method builds on, and nothing else.

`dinomaly_anomalib` reaches its DINOv2 encoder through anomalib, which pins one encoder and
hides it inside a model. An in-house method cannot: importing anomalib to borrow a backbone
would make every "ours" method partly a view of that library, which is exactly the property
`efficientad_custom` exists to avoid. So the backbone is resolved here, from timm directly,
and the methods that follow — a memory bank first, a Dinomaly port later — take it as an
argument rather than owning one each.

**Five encoders, and the shape of the menu is the point.** Two families, at two widths, with
the registered DINOv2 variant that anomalib pins included so a comparison against
`dinomaly_anomalib` measures the method rather than the encoder. Three of the five are
ungated Apache-2.0 weights and are the default any new method should reach for; the two
DINOv3 entries are behind Meta's DINOv3 licence and need an approved Hugging Face account,
which `load_backbone` says in words rather than as a 401.

**The patch size is a hard boundary, not a resize.** DINOv2 strides by 14 and DINOv3 by 16,
so a prepared frame that does not divide cleanly produces a token count that cannot be
reshaped to a grid — timm fails several frames later, in a message about tensor shapes.
`validate_prepared_size` refuses it at the plugin boundary instead, the way
`dinomaly_anomalib.validate_prepared_size` does, and names the sizes that would work.

Heavy imports stay inside functions. This module is imported wherever a plan is made, and a
plan must not cost three seconds of torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from anomaly_lab.models.model_assets import fingerprint_state, huggingface_environment


class DinoBackbone(StrEnum):
    """Which frozen encoder produces the patch features.

    A closed set rather than a free timm name, for the reason `patchcore_anomalib.LayerSet`
    is one: the form is generated from the schema and renders an enum as a picker, where a
    text box would let a typo fail inside timm minutes into a job. The five here are also
    the five whose licence, patch size and width this module can state without asking the
    network.

    Deliberately excluded: every `_qkvb` DINOv3 variant and the `eupe` pretrainings. timm
    carries them, but they resolve to weights under a FAIR noncommercial research licence
    (`fair-noncommercial-research-license` in timm's own pretrained config), and a menu that
    mixes a licence a user may not use with four they may is a menu that has to be read
    rather than chosen from. The larger DINOv3 sizes are left out for a duller reason: at
    ViT-L and above the encoder forward, not the method, is what a run measures.
    """

    DINOV2_VIT_S14 = "dinov2_vit_s14"
    DINOV2_VIT_S14_REG4 = "dinov2_vit_s14_reg4"
    DINOV2_VIT_B14 = "dinov2_vit_b14"
    DINOV3_VIT_S16 = "dinov3_vit_s16"
    DINOV3_VIT_B16 = "dinov3_vit_b16"


@dataclass(frozen=True)
class BackboneSpec:
    """What can be said about an encoder without instantiating it.

    Pure data, so the size guard, the grid arithmetic and the licence sentence are all
    available to the torch-free CI job and to the API process that renders the form.
    """

    timm_name: str
    patch_size: int
    embedding_dim: int
    depth: int
    gated: bool
    license_note: str


BACKBONES: dict[DinoBackbone, BackboneSpec] = {
    DinoBackbone.DINOV2_VIT_S14: BackboneSpec(
        timm_name="vit_small_patch14_dinov2.lvd142m",
        patch_size=14,
        embedding_dim=384,
        depth=12,
        gated=False,
        license_note="Apache-2.0; no Hugging Face account or token needed.",
    ),
    DinoBackbone.DINOV2_VIT_S14_REG4: BackboneSpec(
        timm_name="vit_small_patch14_reg4_dinov2.lvd142m",
        patch_size=14,
        embedding_dim=384,
        depth=12,
        gated=False,
        license_note=(
            "Apache-2.0; no Hugging Face account or token needed. This is the encoder "
            "anomalib pins for Dinomaly, so a run against it compares methods rather "
            "than encoders."
        ),
    ),
    DinoBackbone.DINOV2_VIT_B14: BackboneSpec(
        timm_name="vit_base_patch14_dinov2.lvd142m",
        patch_size=14,
        embedding_dim=768,
        depth=12,
        gated=False,
        license_note="Apache-2.0; no Hugging Face account or token needed.",
    ),
    DinoBackbone.DINOV3_VIT_S16: BackboneSpec(
        timm_name="vit_small_patch16_dinov3.lvd1689m",
        patch_size=16,
        embedding_dim=384,
        depth=12,
        gated=True,
        license_note=(
            "DINOv3 licence: access must be requested from Meta on Hugging Face and a "
            "valid HF_TOKEN must be present before the weights will download."
        ),
    ),
    DinoBackbone.DINOV3_VIT_B16: BackboneSpec(
        timm_name="vit_base_patch16_dinov3.lvd1689m",
        patch_size=16,
        embedding_dim=768,
        depth=12,
        gated=True,
        license_note=(
            "DINOv3 licence: access must be requested from Meta on Hugging Face and a "
            "valid HF_TOKEN must be present before the weights will download."
        ),
    ),
}

ACCESS_REQUEST_URLS: dict[DinoBackbone, str] = {
    DinoBackbone.DINOV3_VIT_S16: (
        "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m"
    ),
    DinoBackbone.DINOV3_VIT_B16: (
        "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
    ),
}
"""Where a user goes to ask for the gated weights. Kept beside the table rather than in it
because only two of the five entries have one, and a field that is `None` four times out of
five is a field that gets read as optional rather than as licence-specific."""


class FeatureLayers(StrEnum):
    """Which transformer blocks the patch features are read from.

    Depth is the real trade and the reason this is a menu rather than a number: the last
    block is the most semantic and the least localized, and a reconstruction method wants
    the middle of the stack while a nearest-neighbour bank often does better nearer the end.
    The four here bracket that, and `mid_late` is deliberately the region Dinomaly
    reconstructs so an in-house port has the same features available to it.

    The indices are **negative**, and that is not a stylistic choice. timm's
    `feature_take_indices` resolves a negative index against the model's own depth, so one
    `FeatureLayers` value means the same relative position on every encoder in `BACKBONES`
    without this module having to look the depth up first.
    """

    LAST = "last"
    LAST_TWO = "last_two"
    MID_LATE = "mid_late"
    LAST_FOUR = "last_four"

    @property
    def indices(self) -> tuple[int, ...]:
        return _LAYER_INDICES[self]


_LAYER_INDICES: dict[FeatureLayers, tuple[int, ...]] = {
    FeatureLayers.LAST: (-1,),
    FeatureLayers.LAST_TWO: (-2, -1),
    FeatureLayers.MID_LATE: (-7, -4),
    FeatureLayers.LAST_FOUR: (-4, -3, -2, -1),
}


def validate_prepared_size(backbone: DinoBackbone, width: int, height: int) -> None:
    """Fail at the plugin boundary, before timm's token reshape fails downstream.

    The same guard `dinomaly_anomalib` carries for its own fixed encoder, generalised to a
    divisor that the chosen backbone decides. A message that only says "shape mismatch" sends
    the reader into timm; this one names the encoder, the divisor, which dimension is wrong
    and the two sizes nearest to what was asked for.
    """
    spec = BACKBONES[backbone]
    patch = spec.patch_size
    if width < patch or height < patch:
        msg = (
            f"{backbone.value} sees {patch}-pixel patches, so a prepared frame smaller than "
            f"{patch}x{patch} has no patches at all; got {width}x{height}."
        )
        raise ValueError(msg)
    offending = [name for name, value in (("width", width), ("height", height)) if value % patch]
    if not offending:
        return
    msg = (
        f"{backbone.value} uses {patch}-pixel patches, so prepared width and height must "
        f"both be divisible by {patch}; got {width}x{height} "
        f"({' and '.join(offending)} {'do' if len(offending) > 1 else 'does'} not divide). "
        f"The nearest valid sizes are {_nearest_valid(width, patch)} wide and "
        f"{_nearest_valid(height, patch)} high — set one of those in the experiment "
        "preprocessing."
    )
    raise ValueError(msg)


def _nearest_valid(value: int, patch: int) -> str:
    """The multiple below and the multiple above, as one readable phrase."""
    lower = max(patch, value - value % patch)
    upper = lower + patch
    return f"{lower} or {upper}"


def patch_grid(backbone: DinoBackbone, width: int, height: int) -> tuple[int, int]:
    """The token grid a prepared frame produces, as `(rows, cols)`.

    Pure arithmetic and validated first, because a grid computed by truncating division on a
    size that does not divide is a plausible number that no forward pass will agree with.
    """
    validate_prepared_size(backbone, width, height)
    patch = BACKBONES[backbone].patch_size
    return height // patch, width // patch


def load_backbone(
    backbone: DinoBackbone,
    *,
    pretrained: bool,
    allow_downloads: bool,
    cache_dir: Path,
    seed: int,
    method: str,
) -> Any:
    """Resolve one frozen encoder inside app-managed storage, evaluated and frozen.

    The seed is set **before** construction, not after. An unpretrained encoder draws its
    weights from torch's *global* stream, which is M6's finding and the reason
    `dinomaly_anomalib._build_model` and `patchcore_anomalib._build_model` both seed on the
    line above the constructor: seeded afterwards, `seed` would control what the method does
    with the encoder while the encoder itself differed run to run — and the fingerprint
    written at save time would never match the one computed at load time.

    Downloads go through `model_assets.huggingface_environment`, so the weights land in the
    app's own cache and `allow_downloads=False` turns a network reach into a local failure
    naming the asset and the knob.
    """
    import timm
    import torch

    spec = BACKBONES[backbone]
    torch.manual_seed(seed)
    try:
        with huggingface_environment(
            cache_dir,
            allow_downloads=allow_downloads,
            method=method,
            asset=spec.timm_name,
        ):
            model = timm.create_model(
                spec.timm_name,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
            )
    except Exception as exc:
        if not spec.gated:
            raise
        raise RuntimeError(_gated_failure_message(backbone, spec, exc)) from exc

    model = model.eval()
    # Frozen means frozen: no method here trains the encoder, and a stray `requires_grad`
    # would put it in an optimizer's parameter list the first time one is built from
    # `model.parameters()`.
    model.requires_grad_(False)
    return model


def _gated_failure_message(backbone: DinoBackbone, spec: BackboneSpec, exc: Exception) -> str:
    """Turn a 401 into the three sentences that actually unblock the reader.

    The token itself is never read, stored or printed here. What the message says is that
    one is needed and where to get permission for it — an environment variable's *value* has
    no place in a log line that will be pasted into an issue.
    """
    ungated = ", ".join(entry.value for entry in DinoBackbone if not BACKBONES[entry].gated)
    return (
        f"{backbone.value} resolves to {spec.timm_name}, whose weights are gated. "
        f"{spec.license_note} Request access at {ACCESS_REQUEST_URLS[backbone]}, then set a "
        "valid HF_TOKEN environment variable for the account that was approved. If that is "
        f"not wanted, {ungated} are ungated Apache-2.0 alternatives that need no account at "
        f"all. The underlying failure was: {exc}"
    )


def extract_patch_features(model: Any, batch: Any, indices: tuple[int, ...]) -> Any:
    """Patch tokens from the chosen blocks as one `(B, D_total, rows, cols)` map.

    Each requested block is taken through timm's `forward_intermediates` with `norm=True`,
    so what comes back is the block output after the encoder's own final normalisation
    rather than a raw residual — the same tensors Dinomaly reconstructs.

    **Each layer is L2-normalized on its own, before the concatenation.** That order is a
    constraint rather than a preference: everything downstream compares these vectors by
    Euclidean distance, and block norms are not comparable across depth — a late block whose
    activations happen to be twice as large would contribute four times the squared distance
    of an early one and silently decide every neighbour. Normalising first makes the
    concatenation an equal vote per layer; normalising after would flatten the whole vector
    and lose exactly that.
    """
    import torch

    maps = model.forward_intermediates(
        batch,
        indices=list(indices),
        norm=True,
        output_fmt="NCHW",
        intermediates_only=True,
    )
    normalized = [torch.nn.functional.normalize(feature, p=2.0, dim=1) for feature in maps]
    return torch.cat(normalized, dim=1)


def backbone_fingerprint(model: Any) -> str:
    """A stable digest of the frozen encoder's weights.

    A one-line wrapper over `model_assets.fingerprint_state` so a caller does not have to
    know that the thing being fingerprinted is a `state_dict` — and so every method that
    loads a backbone through this module fingerprints the same tensors in the same way.
    The weights are an input to the experiment (`docs/architecture/methods.md`), and a run
    that cannot say which ones it used cannot be compared against another.
    """
    return fingerprint_state(model.state_dict())
