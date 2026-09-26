"""`anomalyvfm_anomalib` — AnomalyVFM, a zero-shot adapted RADIO model, through anomalib.

AnomalyVFM ships one complete checkpoint: a RADIO ViT-L/16 with DoRA adapters, a mask
decoder and an image-score head, trained once on synthetic anomalies. Nothing about a
dataset enters it, so there is nothing to fit — a run scores images with the published
weights, and the same image gives the same answer in every experiment.

The wrapper keeps anomalib's network and forward pass and owns everything around them:

  * **The asset is resolved, verified, then handed to the constructor.** anomalib's
    `AnomalyVFMModel.__init__` calls `hf_hub_download(..., local_files_only=False)`, so
    building it bare may reach the network. Here the pinned revision is found in the app
    cache (fetched once only when `allow_downloads` permits), its size and SHA-256 are
    checked, and the constructor's download call is answered with that verified file by a
    scoped substitute that refuses any other request by name.
  * **The frame is bounded before anything is built.** `plan_inference` is torch-free: the
    patch grid, the token count and the attention footprint are known and logged before
    1.4 GB of weights is read, and `check_input` refuses a frame the patch size does not
    divide or whose attention would not fit.
  * **Construction leaves the process as it found it.** The constructor draws random
    initial weights that the strict state-dict load then overwrites; it does so inside a
    forked RNG so the process's torch stream does not move. It builds no timm model, so
    `model_assets.timm_bindings_preserved` has nothing to restore here.

Heavy imports stay inside functions so the method registry remains torch-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from anomaly_lab.models.base import (
    AnomalyModel,
    Availability,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
    module_available,
)
from anomaly_lab.models.model_assets import huggingface_environment
from anomaly_lab.models.preprocessing import (
    PreprocessingConfig,
    expand_planes,
    load_array,
    to_chw,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

METHOD = "AnomalyVFM"
STATE_FILENAME = "anomalyvfm.json"
STATE_FORMAT = 1

PATCH_SIZE = 16
"""RADIO's patch edge. A frame it does not divide cannot be patchified at all."""

PREFIX_TOKENS = 8
"""RADIO's summary and register tokens, attended to beside every patch."""

HEADS = 16
EMBED_DIM = 1024

MEASURED_SIZE = 768
"""The frame the resource gate measured and kept for quality (docs/measurements.md)."""

MAX_ATTENTION_BYTES = 4 * 1024**3
"""The largest float32 attention matrix one layer may need. At 768 px it is 0.32 GiB; the
bound admits frames up to about 1 400 px square, and past it a run would be paging rather
than scoring."""


@dataclass(frozen=True)
class PinnedAsset:
    """One file of a Hugging Face repository at an immutable revision, checksummed."""

    repository: str
    filename: str
    revision: str
    sha256: str
    size: int

    def snapshot_path(self, hub_cache: Path) -> Path:
        """Where `huggingface_hub` stores this revision's file under `hub_cache`."""
        folder = "models--" + self.repository.replace("/", "--")
        return hub_cache / folder / "snapshots" / self.revision / self.filename

    def describe(self) -> str:
        return f"{self.repository}@{self.revision[:12]}/{self.filename}"


ASSET = PinnedAsset(
    repository="MaticFuc/anomalyvfm_radio",
    filename="model.safetensors",
    revision="17654e763c8fae5ae1c44e2ec421a427783d6196",
    sha256="50a219ba436ed656ad3c0405f9e81df8ad00b2e715c98be66d7e2edb62a83a37",
    size=1_421_491_228,
)


ASSET_NAME = "AnomalyVFM RADIO checkpoint"


class AssetError(RuntimeError):
    """The pinned checkpoint is missing, or is not the file that was pinned."""


class AnomalyVfmConfig(BaseModel):
    """Stable experiment controls rendered from JSON Schema."""

    model_config = API_MODEL_CONFIG

    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the pinned 1.42 GB AnomalyVFM checkpoint into the app model "
            "cache on the first run. It is checked by size and SHA-256 either way; turn "
            "off to require the verified file to be cached already."
        ),
    )


# ---------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class InferencePlan:
    """What one image costs at this frame, resolved before the weights are read."""

    width: int
    height: int

    @property
    def grid(self) -> tuple[int, int]:
        return (self.width // PATCH_SIZE, self.height // PATCH_SIZE)

    @property
    def tokens(self) -> int:
        columns, rows = self.grid
        return columns * rows + PREFIX_TOKENS

    @property
    def attention_bytes(self) -> int:
        """One layer's float32 attention matrix, were it materialised whole."""
        return HEADS * self.tokens * self.tokens * 4

    @property
    def weights_bytes(self) -> int:
        return ASSET.size

    def describe(self) -> str:
        columns, rows = self.grid
        text = (
            f"{METHOD} plan: {self.width}x{self.height} frame, {columns}x{rows} patch grid, "
            f"{self.tokens:,} tokens; weights {self.weights_bytes / 1e9:.2f} GB, read once "
            f"per job (about twice that on the host while they load); at most "
            f"{self.attention_bytes / 1024**3:.2f} GiB of attention per layer; batch 1"
        )
        if (self.width, self.height) != (MEASURED_SIZE, MEASURED_SIZE):
            text += (
                f". Its resources and quality were measured at {MEASURED_SIZE}x"
                f"{MEASURED_SIZE}; this frame is not that one"
            )
        return text


def plan_inference(width: int, height: int) -> InferencePlan:
    """Pure, torch-free: refuse a frame the network cannot read, and size one it can."""
    if width % PATCH_SIZE or height % PATCH_SIZE or width < PATCH_SIZE or height < PATCH_SIZE:
        raise ValueError(
            f"{METHOD} reads {PATCH_SIZE}-pixel patches, so the prepared width and height "
            f"must be positive multiples of {PATCH_SIZE}; got {width}x{height}. "
            f"{MEASURED_SIZE}x{MEASURED_SIZE} is the measured size."
        )
    plan = InferencePlan(width=width, height=height)
    if plan.attention_bytes > MAX_ATTENTION_BYTES:
        raise ValueError(
            f"{METHOD} at {width}x{height} attends over {plan.tokens:,} tokens, "
            f"{plan.attention_bytes / 1024**3:.1f} GiB per layer, above the "
            f"{MAX_ATTENTION_BYTES / 1024**3:.0f} GiB bound; prepare a smaller frame "
            f"({MEASURED_SIZE}x{MEASURED_SIZE} is the measured size)"
        )
    return plan


# ------------------------------------------------------------------------- the asset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, asset: PinnedAsset) -> None:
    """Refuse a file that is not, byte for byte, the pinned one."""
    size = path.stat().st_size
    if size != asset.size:
        raise AssetError(
            f"{METHOD}'s checkpoint at {path} has {size:,} bytes; {asset.describe()} has "
            f"{asset.size:,}. Delete the file and let the next run fetch it again."
        )
    actual = _sha256(path)
    if actual != asset.sha256:
        raise AssetError(
            f"{METHOD}'s checkpoint at {path} has SHA-256 {actual}; {asset.describe()} is "
            f"{asset.sha256}. Delete the file and let the next run fetch it again."
        )


def _download(asset: PinnedAsset, cache_dir: Path) -> Path:
    """Fetch the pinned revision into the app cache. The one place the network is reached."""
    from huggingface_hub import hf_hub_download

    with huggingface_environment(cache_dir, allow_downloads=True, method=METHOD, asset=ASSET_NAME):
        return Path(
            hf_hub_download(
                repo_id=asset.repository,
                filename=asset.filename,
                revision=asset.revision,
                cache_dir=cache_dir / "huggingface" / "hub",
            )
        )


def resolve_weights(
    cache_dir: Path,
    *,
    allow_downloads: bool,
    log: Callable[[str], None],
    asset: PinnedAsset | None = None,
) -> Path:
    """The verified checkpoint in the app cache, fetched first only when that is allowed.

    Torch-free and, when the file is already cached, free of `huggingface_hub` too: the
    snapshot layout is fixed by the revision, so finding it needs no library.
    """
    asset = ASSET if asset is None else asset
    path = asset.snapshot_path(cache_dir / "huggingface" / "hub")
    if not path.is_file():
        if not allow_downloads:
            raise AssetError(
                f"{METHOD} needs its pinned checkpoint {asset.describe()} "
                f"({asset.size / 1e9:.2f} GB), downloads are disabled, and the app cache "
                f"does not hold it at {path}. Turn allow_downloads on for one run."
            )
        log(f"fetching {asset.describe()} ({asset.size / 1e9:.2f} GB) into the app cache")
        path = _download(asset, cache_dir)
    started = time.perf_counter()
    verify_asset(path, asset)
    log(
        f"verified {asset.describe()}: {asset.size:,} bytes, SHA-256 {asset.sha256[:12]}… "
        f"({time.perf_counter() - started:.1f}s)"
    )
    return path


def pinned_download(asset: PinnedAsset, weights: Path) -> Callable[..., str]:
    """A stand-in for `hf_hub_download` that answers one request, with a verified file.

    anomalib's constructor names its repository, file and revision; any of them differing
    from the pin means the library moved under the plugin, which is refused by name rather
    than loaded.
    """

    def download(*args: Any, **kwargs: Any) -> str:
        requested = (
            kwargs.get("repo_id", args[0] if args else None),
            kwargs.get("filename", args[1] if len(args) > 1 else None),
            kwargs.get("revision"),
        )
        pinned = (asset.repository, asset.filename, asset.revision)
        if requested != pinned:
            raise AssetError(
                f"anomalib asked for {requested[0]}@{requested[2]}/{requested[1]}, but "
                f"{METHOD} pins {asset.describe()}. The installed anomalib no longer loads "
                "the checkpoint this plugin verifies; update the pin deliberately."
            )
        return str(weights)

    return download


def build_network(weights: Path, device: str) -> Any:
    """anomalib's `AnomalyVFMModel`, loaded from `weights` with no network access."""
    import torch

    # Read as `Any`: the attribute replaced below is the module's own import, not its API.
    torch_model: Any = import_module("anomalib.models.image.anomalyvfm.torch_model")
    original = torch_model.hf_hub_download
    torch_model.hf_hub_download = pinned_download(ASSET, weights)
    try:
        # The constructor draws random initial weights before the strict load replaces
        # every one of them; forking keeps that draw off the process's own stream.
        with torch.random.fork_rng(devices=[]):
            network = torch_model.AnomalyVFMModel()
    finally:
        torch_model.hf_hub_download = original
    return network.eval().to(device)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


# ------------------------------------------------------------------------ the plugin


class AnomalyVfmAnomalibModel(AnomalyModel):
    """AnomalyVFM's published checkpoint, scored per image; nothing is fitted."""

    title = "AnomalyVFM (zero-shot reference)"
    summary = (
        "A foundation model adapted once for anomaly detection and used as published: it "
        "reads no normal images, so every dataset gets the same weights. The zero-shot "
        "reference; its public gate has not run yet."
    )

    def __init__(self, config: AnomalyVfmConfig) -> None:
        super().__init__(config)
        self.config = config
        self._record: dict[str, Any] | None = None
        self._network: Any = None
        self._network_device: str | None = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return AnomalyVfmConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            # Zero-shot: an experiment scores without a train job. A train job, when one
            # runs, fetches and verifies the checkpoint and records which one it was.
            requires_training=False,
            produces_anomaly_map=True,
            produces_diagnostics=False,
            supports_resume=False,
            channel_aware=False,
            dataset_specific=False,
            # anomalib reports export as unsupported for AnomalyVFM, and no parity gate
            # has run, so no portable format is claimed (ADR-0034).
            portable_formats=[],
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("anomalib", "dl", METHOD)

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """768 px square: the frame its resource gate kept and its public gate ran at."""
        return (MEASURED_SIZE, MEASURED_SIZE)

    @classmethod
    def size_multiple(cls, config: BaseModel) -> int:
        return PATCH_SIZE

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, AnomalyVfmConfig):
            raise TypeError(f"expected AnomalyVfmConfig, got {type(config).__name__}")
        plan_inference(preprocessing.width, preprocessing.height)

    # ---------------------------------------------------------------------- fit

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        plan = plan_inference(ctx.preprocessing.width, ctx.preprocessing.height)
        ctx.log(
            f"{METHOD} is zero-shot: it reads none of the {len(train):,} training images. "
            "This job fetches and verifies its checkpoint and records which one it was."
        )
        ctx.log(plan.describe())
        ctx.progress(0.05, f"resolving {ASSET.describe()}")
        resolve_weights(ctx.cache_dir, allow_downloads=self.config.allow_downloads, log=ctx.log)
        self._record = {
            "format": STATE_FORMAT,
            "repository": ASSET.repository,
            "filename": ASSET.filename,
            "revision": ASSET.revision,
            "sha256": ASSET.sha256,
            "width": ctx.preprocessing.width,
            "height": ctx.preprocessing.height,
            "versions": {
                "anomalib": _package_version("anomalib"),
                "torch": _package_version("torch"),
            },
        }
        ctx.progress(1.0, "checkpoint verified")

    # ------------------------------------------------------------------ predict

    def _network_for(self, ctx: InferContext) -> Any:
        device = ctx.device.value
        if self._network is None:
            weights = resolve_weights(
                ctx.cache_dir, allow_downloads=self.config.allow_downloads, log=ctx.log
            )
            ctx.progress(0.0, f"loading {METHOD} ({ASSET.size / 1e9:.2f} GB)")
            started = time.perf_counter()
            self._network = build_network(weights, device)
            ctx.log(f"built {METHOD} on {device} in {time.perf_counter() - started:.1f}s")
        elif self._network_device != device:
            self._network = self._network.to(device)
        self._network_device = device
        return self._network

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        plan = plan_inference(width, height)
        if self._record is not None and (self._record["width"], self._record["height"]) != (
            width,
            height,
        ):
            raise ValueError(
                f"{METHOD} was prepared for {self._record['width']}x{self._record['height']}; "
                f"inference requested {width}x{height}"
            )
        ctx.log(plan.describe())
        network = self._network_for(ctx)

        import torch

        predictions: list[Prediction] = []
        with torch.inference_mode():
            for index, record in enumerate(images):
                ctx.raise_if_cancelled()
                # AnomalyVFM standardises inside its own first module (RADIO's input
                # conditioner, CLIP statistics), so it is handed pixels in [0, 1].
                chw = expand_planes(to_chw(load_array(record.path, ctx.preprocessing)), 3)
                batch = torch.from_numpy(chw[None]).to(ctx.device.value)
                started = time.perf_counter()
                score, anomaly_map = network(batch)
                if ctx.device is Device.MPS:
                    torch.mps.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                path = ctx.write_map(record.image_id, anomaly_map[0, 0].float().cpu().numpy())
                predictions.append(
                    Prediction(
                        image_id=record.image_id,
                        score=float(score.reshape(-1)[0].float().cpu()),
                        anomaly_map=path,
                        inference_ms=elapsed_ms,
                    )
                )
                ctx.progress(
                    (index + 1) / max(len(images), 1),
                    f"scored {index + 1}/{len(images)} images",
                )
        return predictions

    # -------------------------------------------------------------- persistence

    def save(self, artifact_dir: Path) -> None:
        if self._record is None:
            raise RuntimeError(f"{METHOD} has nothing to save; no train job verified its asset")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / STATE_FILENAME
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self._record, indent=2, sort_keys=True), "utf-8")
            temporary.replace(path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def load(self, artifact_dir: Path) -> None:
        stored = json.loads((artifact_dir / STATE_FILENAME).read_text(encoding="utf-8"))
        if int(stored.get("format", 0)) != STATE_FORMAT:
            raise RuntimeError(
                f"unsupported {METHOD} record format {stored.get('format')!r}; "
                f"expected {STATE_FORMAT}"
            )
        recorded = (stored.get("repository"), stored.get("revision"), stored.get("sha256"))
        if recorded != (ASSET.repository, ASSET.revision, ASSET.sha256):
            raise RuntimeError(
                f"this run recorded {METHOD} checkpoint {recorded[0]}@{recorded[1]} "
                f"(SHA-256 {str(recorded[2])[:12]}), but the plugin now pins "
                f"{ASSET.describe()}. Its scores would not be comparable; create a new "
                "experiment."
            )
        self._record = stored
