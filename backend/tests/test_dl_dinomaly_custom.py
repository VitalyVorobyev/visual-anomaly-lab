"""`dinomaly_custom` as a plugin, and as a port: is it Dinomaly, and does it behave here.

**Nothing here downloads anything.** Every encoder is built with `pretrained_encoder=False` —
a seeded random ViT — and that is not a compromise made to keep CI cheap. Reconstruction
needs *some* feature space and an untrained transformer is one: 300 steps on the stamped
fixture below take the loss from 1.04 to 0.046 and score the defect twice the normal. The
published weights make it better on real data and are not what this file is testing.

Two tests are **bring-up pins** rather than contract tests, and they are the evidence this PR
offers for ADR-0029's "the wrapper is the baseline" claim:

  * `test_our_encoder_path_is_the_shared_backbone_module_exactly` pins that
    `dino_backbone.extract_layer_tokens` over `load_backbone` is bit-identical to calling
    timm's `forward_intermediates` directly — so the shared module is the single source of
    encoder truth and this method is not quietly carrying a second one.
  * `test_one_training_step_matches_anomalib_exactly` and its inference sibling pin that our
    loss, our gradients, our StableAdamW step, our anomaly map and our image score are
    **bit-identical** to anomalib's on identical weights. They import anomalib, which nothing
    in `dinomaly_custom.py` or `dinomaly_nets.py` may do.

**Those two are retirable by replacement only.** They exist to prove the port landed, not to
freeze it: the moment a deliberate divergence from anomalib is chosen — a different fusion,
a different score rule — the pin does not get loosened to "the loss is finite", it gets
*replaced* by a test of the new behaviour and a measurement saying the change was worth it.
Weakening a bit-exact pin into a tolerance is how a port stops being one.

The prepared size is 112 throughout: it divides by 14 into an 8x8 token grid, so a forward
pass over an untrained ViT-S is a fraction of a second.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
    SupportsResume,
    TrainContext,
)
from anomaly_lab.models.diagnostics import (  # noqa: E402
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
from anomaly_lab.models.dino_backbone import (  # noqa: E402
    BACKBONES,
    DinoBackbone,
    extract_layer_tokens,
    load_backbone,
)
from anomaly_lab.models.dinomaly_custom import (  # noqa: E402
    STATE_FILENAME,
    TARGET_LAYERS,
    DinomalyCustomConfig,
    DinomalyCustomModel,
    fuse_groups,
    trainable_parameter_count,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402

SIZE = 112
"""Divisible by 14, so ViT-S/14 sees an 8x8 token grid — 64 patches plus five prefix tokens."""

STAMP = slice(42, 70)
"""Rows and columns of the bright square: two whole /14 patches wide and centred, so the
map's maximum has somewhere honest to land."""

FIT_STEPS = 300
"""Enough for reconstruction to mean something on this fixture, measured rather than guessed:
the loss falls from 1.04 to 0.046 and the defect scores about 2x a normal. About 15 s on CPU,
which is why the fitted model is a module-scoped fixture."""

ENCODER = DinoBackbone.DINOV2_VIT_S14_REG4
ANOMALIB_ENCODER = "vit_small_patch14_reg4_dinov2"
"""The same architecture under anomalib's untagged spelling. `BACKBONES` names the tagged
`.lvd142m` pretrained config; with `pretrained=False` the two build identical modules, which
`test_one_training_step_matches_anomalib_exactly` asserts by loading one state dict into both."""


# ------------------------------------------------------------------------ fixtures


def _field(seed: int) -> np.ndarray:
    """A shallow seeded gradient with light noise — a scene, not a random image."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(104, 116, SIZE)[:, None]
    cols = np.linspace(0, 6, SIZE)[None, :]
    return np.clip(rows + cols + rng.normal(0.0, 3.0, size=(SIZE, SIZE)), 0.0, 255.0)


def _save(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="L").convert("RGB").save(path)
    return path


def _normal(path: Path, seed: int) -> Path:
    return _save(path, _field(seed))


def _defect(path: Path, seed: int) -> Path:
    """The same scene with a bright square stamped in — an appearance nothing has seen."""
    values = _field(seed)
    values[STAMP, STAMP] = 255.0
    return _save(path, values)


class _Recorder:
    """A reporter that keeps the metric series, and can be told to cancel after N steps."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.losses: list[tuple[int, float]] = []
        self.logs: list[str] = []
        self.cancel_after = cancel_after
        self.polls = 0

    def progress(self, fraction: float, message: str | None = None) -> None:
        return

    def log(self, message: str, level: str = "info") -> None:
        self.logs.append(message)

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        if name == "loss_total":
            self.losses.append((step or 0, value))

    def should_cancel(self) -> bool:
        self.polls += 1
        return self.cancel_after is not None and self.polls > self.cancel_after


def _contexts(
    root: Path,
    *,
    reporter: Any | None = None,
    size: int = SIZE,
) -> tuple[TrainContext, InferContext]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig(width=size, height=size)
    diagnostics = artifacts / "diagnostics"
    reporter = reporter or NullReporter()
    return (
        TrainContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=preprocessing,
            device=Device.CPU,
            reporter=reporter,
            diagnostics=DiagnosticWriter(diagnostics),
        ),
        InferContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=preprocessing,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(diagnostics),
        ),
    )


def _train_records(root: Path, count: int = 8) -> list[ImageRecord]:
    return [
        ImageRecord(image_id=index, sample_id=index, path=_normal(root / f"n{index}.png", index))
        for index in range(count)
    ]


def _config(**overrides: Any) -> DinomalyCustomConfig:
    """A hermetic configuration: a seeded random encoder and no reach for the network."""
    settings: dict[str, Any] = {
        "encoder": ENCODER,
        "max_steps": 4,
        "pretrained_encoder": False,
        "allow_downloads": False,
        "seed": 0,
    }
    settings.update(overrides)
    return DinomalyCustomConfig(**settings)


def _digest(model: DinomalyCustomModel) -> str:
    """A stable digest of everything training moved — weights only, no optimizer state."""
    assert model._net is not None
    sha = hashlib.sha256()
    for name, state in sorted(model._net.trainable_state().items()):
        sha.update(name.encode())
        for key, tensor in sorted(state.items()):
            sha.update(key.encode())
            sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


@pytest.fixture(scope="module")
def fitted(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[DinomalyCustomModel, Path, list[float]]:
    """One real fit, reused: 300 steps is 15 s and several tests need the same trained model."""
    root = tmp_path_factory.mktemp("fitted")
    recorder = _Recorder()
    train_ctx, _ = _contexts(root, reporter=recorder)
    model = DinomalyCustomModel(_config(max_steps=FIT_STEPS, seed=1))
    model.fit(_train_records(root), train_ctx)
    # The job handler flushes the index once the model returns; a test with no job system
    # has to do the same before it can read what was emitted.
    train_ctx.diagnostics.flush()
    return model, root, [value for _, value in recorder.losses]


# ------------------------------------------------------------- bring-up pin: encoder


def test_our_encoder_path_is_the_shared_backbone_module_exactly(tmp_path: Path) -> None:
    """`load_backbone` + `extract_layer_tokens` is timm's own call, and nothing else.

    A bring-up pin. An in-house method that resolved its encoder any other way — a second
    `create_model`, a different `dynamic_img_size`, a stray final norm — would produce a
    feature space that looks right and is not the one `dino_memory` and the plan arithmetic
    describe. `atol=0`: this is an identity claim, not a numerical one.
    """
    import timm

    spec = BACKBONES[ENCODER]
    batch = torch.from_numpy(
        np.random.default_rng(4).normal(size=(1, 3, SIZE, SIZE)).astype(np.float32)
    )

    ours = load_backbone(
        ENCODER,
        pretrained=False,
        allow_downloads=False,
        cache_dir=tmp_path / "cache",
        seed=17,
        method="test",
    )
    with torch.no_grad():
        our_tokens = extract_layer_tokens(ours, batch, TARGET_LAYERS)

    torch.manual_seed(17)
    reference: Any = timm.create_model(
        spec.timm_name, pretrained=False, num_classes=0, dynamic_img_size=True
    ).eval()
    with torch.no_grad():
        pairs = reference.forward_intermediates(
            batch,
            indices=list(TARGET_LAYERS),
            norm=False,
            return_prefix_tokens=True,
            output_fmt="NLC",
            intermediates_only=True,
        )
    expected = [torch.cat((prefix, patches), dim=1) for patches, prefix in pairs]

    # The prefix tokens are kept and come first: five of them on a reg4 encoder, then 64
    # patch tokens for the 8x8 grid.
    tokens = int(ours.num_prefix_tokens) + (SIZE // spec.patch_size) ** 2
    assert tokens == 69
    assert len(our_tokens) == len(TARGET_LAYERS)
    for actual, wanted in zip(our_tokens, expected, strict=True):
        assert actual.shape == (1, tokens, spec.embedding_dim)
        torch.testing.assert_close(actual, wanted, atol=0.0, rtol=0.0)


# ------------------------------------------------------------ bring-up pin: anomalib


def _anomalib_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, depth: int = 8) -> Any:
    """anomalib's Dinomaly and ours, weight for weight identical, both untrained.

    anomalib's constructor hard-codes `pre_trained=True`, so the extractor is swapped for a
    subclass that forces it off. Everything else — the bottleneck, the eight decoder blocks,
    the loss, the optimizer and the map rule — is the real one.

    Dropout is **zero on both sides**, and that is the pin's one deliberate simplification:
    the bottleneck's three dropout draws would otherwise have to consume the global RNG in
    exactly the same order in two different module trees, which would pin the RNG bookkeeping
    rather than the arithmetic. `test_one_seed_is_one_experiment` covers the stream.
    """
    from anomalib.models.components.feature_extractors import TimmFeatureExtractor
    from anomalib.models.image.dinomaly import torch_model as anomalib_dinomaly

    from anomaly_lab.models.dinomaly_nets import DinomalyNet

    extractor_base: Any = TimmFeatureExtractor

    class Unpretrained(extractor_base):  # type: ignore[misc]
        def __init__(self, **kwargs: Any) -> None:
            kwargs["pre_trained"] = False
            super().__init__(**kwargs)

    monkeypatch.setattr(anomalib_dinomaly, "TimmFeatureExtractor", Unpretrained)

    torch.manual_seed(11)
    theirs = anomalib_dinomaly.DinomalyModel(
        encoder_name=ANOMALIB_ENCODER,
        bottleneck_dropout=0.0,
        decoder_depth=depth,
    )

    spec = BACKBONES[ENCODER]
    encoder = load_backbone(
        ENCODER,
        pretrained=False,
        allow_downloads=False,
        cache_dir=tmp_path / "cache",
        seed=5,
        method="test",
    )
    torch.manual_seed(5)
    ours = DinomalyNet(
        encoder,
        embedding_dim=spec.embedding_dim,
        num_heads=spec.num_heads,
        num_prefix_tokens=int(encoder.num_prefix_tokens),
        patch_size=spec.patch_size,
        decoder_depth=depth,
        target_layers=TARGET_LAYERS,
        encoder_groups=fuse_groups(len(TARGET_LAYERS)),
        decoder_groups=fuse_groups(depth),
        dropout=0.0,
    )

    # Same architecture under two spellings of the timm name: every key lands, none is left.
    report = ours.encoder.load_state_dict(theirs.encoder.feature_extractor.state_dict())
    assert not report.missing_keys and not report.unexpected_keys
    # anomalib's bottleneck is a one-entry ModuleList, so its keys carry a "0." prefix; the
    # decoder's submodule names are identical on both sides and need no remapping at all.
    ours.bottleneck.load_state_dict(
        {key.removeprefix("0."): value for key, value in theirs.bottleneck.state_dict().items()}
    )
    ours.decoder.load_state_dict(theirs.decoder.state_dict())
    return theirs, ours


def test_one_training_step_matches_anomalib_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loss, every gradient and the weights after one StableAdamW step, bit for bit.

    A bring-up pin (see the module docstring): retirable by replacement, never by loosening.
    `global_step=400` puts the hard-mining anneal in its interesting middle rather than at
    either end, so a schedule transcribed off by a constant would show up here.
    """
    from anomalib.models.image.dinomaly.components import StableAdamW as TheirAdamW

    from anomaly_lab.models.dinomaly_nets import (
        StableAdamW,
        hard_mined_cosine_loss,
        mined_fraction,
    )

    step, rate = 400, 1e-3
    theirs, ours = _anomalib_pair(monkeypatch, tmp_path)
    theirs.train()
    ours.train()
    batch = torch.from_numpy(
        np.random.default_rng(2).normal(size=(1, 3, SIZE, SIZE)).astype(np.float32)
    )

    their_optimizer = TheirAdamW(
        [{"params": [p for p in theirs.parameters() if p.requires_grad]}],
        lr=rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=True,
    )
    our_optimizer = StableAdamW(
        [{"params": ours.trainable_parameters()}],
        lr=rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=True,
    )

    their_optimizer.zero_grad(set_to_none=True)
    their_loss = theirs(batch, global_step=step)
    their_loss.backward()

    our_optimizer.zero_grad(set_to_none=True)
    encoded, decoded = ours.features(batch)
    our_loss: Any = hard_mined_cosine_loss(
        encoded, decoded, step=step, fraction=mined_fraction(step, 0.1)
    )
    our_loss.backward()

    assert float(our_loss.detach()) == float(their_loss.detach())

    their_named = {
        **{f"decoder.{key}": value for key, value in theirs.decoder.named_parameters()},
        **{
            f"bottleneck.{key.removeprefix('0.')}": value
            for key, value in theirs.bottleneck.named_parameters()
        },
    }
    our_named = {
        **{f"decoder.{k}": v for k, v in ours.decoder.named_parameters()},
        **{f"bottleneck.{k}": v for k, v in ours.bottleneck.named_parameters()},
    }
    assert set(our_named) == set(their_named)
    for key, parameter in our_named.items():
        assert parameter.grad is not None, key
        torch.testing.assert_close(parameter.grad, their_named[key].grad, atol=0.0, rtol=0.0)

    their_optimizer.step()
    our_optimizer.step()
    for key, parameter in our_named.items():
        torch.testing.assert_close(parameter, their_named[key], atol=0.0, rtol=0.0)


def test_the_map_and_the_score_match_anomalib_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The inference half of the same pin: the anomaly map and the top-one-percent score.

    Both sides here are untrained, which is the point — a map rule is arithmetic over two
    feature stacks and does not need a trained decoder to be wrong. This also covers the
    separable Gaussian in `dinomaly_nets` against the kornia-backed blur anomalib uses.
    """
    from anomaly_lab.models.dinomaly_nets import group_maps, image_score

    theirs, ours = _anomalib_pair(monkeypatch, tmp_path)
    theirs.eval()
    ours.eval()
    batch = torch.from_numpy(
        np.random.default_rng(6).normal(size=(1, 3, SIZE, SIZE)).astype(np.float32)
    )

    with torch.no_grad():
        their_output = theirs(batch)
        encoded, decoded = ours.features(batch)
        our_map = torch.cat(group_maps(encoded, decoded, (SIZE, SIZE)), dim=1).mean(
            dim=1, keepdim=True
        )
        our_score = image_score(our_map)

    torch.testing.assert_close(our_map, their_output.anomaly_map, atol=0.0, rtol=0.0)
    torch.testing.assert_close(our_score, their_output.pred_score, atol=0.0, rtol=0.0)


def test_the_closed_form_parameter_count_is_the_network_s(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The plan's torch-free arithmetic, checked against the thing it describes.

    Also against anomalib's, which is the same claim from the other side: two independently
    written networks with the same parameter count are at least the same size.

    The last assertion records a real difference in *how* the two find their trainable set.
    `DinomalyNet.trainable_parameters` takes the bottleneck and the decoder **by name**;
    anomalib's wrapper filters the whole model on `requires_grad`, and its
    `TimmFeatureExtractor` leaves the encoder's flags alone — so that filter returns the
    encoder's 22 million parameters as well. Harmless there, because the encoder runs under
    `no_grad` and StableAdamW skips a parameter with no gradient, and exactly the kind of
    thing that stops being harmless the first time somebody adds a global gradient clip.
    """
    theirs, ours = _anomalib_pair(monkeypatch, tmp_path, depth=8)
    counted = sum(parameter.numel() for parameter in ours.trainable_parameters())
    assert counted == trainable_parameter_count(BACKBONES[ENCODER].embedding_dim, 8)
    assert counted == sum(
        parameter.numel()
        for module in (theirs.bottleneck, theirs.decoder)
        for parameter in module.parameters()
    )
    assert sum(p.numel() for p in theirs.parameters() if p.requires_grad) > counted


# --------------------------------------------------------------------- house contract


def test_fitting_reduces_the_reconstruction_loss(
    fitted: tuple[DinomalyCustomModel, Path, list[float]],
) -> None:
    """The decoder learns something, and the metric stream is what says so."""
    _, _, losses = fitted
    assert len(losses) >= 4
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[1]


def test_predictions_come_back_one_per_input_in_order(
    fitted: tuple[DinomalyCustomModel, Path, list[float]], tmp_path: Path
) -> None:
    model, root, _ = fitted
    _, infer_ctx = _contexts(root)
    probes = [
        ImageRecord(image_id=70, sample_id=70, path=_normal(tmp_path / "a.png", 70)),
        ImageRecord(image_id=71, sample_id=71, path=_defect(tmp_path / "b.png", 71)),
        ImageRecord(image_id=72, sample_id=72, path=_normal(tmp_path / "c.png", 72)),
    ]
    predictions = model.predict(probes, infer_ctx)

    assert [item.image_id for item in predictions] == [70, 71, 72]
    for prediction in predictions:
        assert prediction.anomaly_map is not None
        values = np.load(prediction.anomaly_map)
        assert values.ndim == 2
        assert values.shape == (SIZE, SIZE)
        assert values.dtype == np.float32
        assert prediction.inference_ms > 0.0


def test_a_stamped_defect_separates_from_the_normals(
    fitted: tuple[DinomalyCustomModel, Path, list[float]], tmp_path: Path
) -> None:
    """Something the decoder has never had to reconstruct scores higher, and lands where it is.

    Both halves matter. A higher image score with a flat map would be a method that detects
    without localising, which is the failure a pixel-level workbench most needs to catch.
    """
    model, root, _ = fitted
    _, infer_ctx = _contexts(root)
    normal = ImageRecord(image_id=80, sample_id=80, path=_normal(tmp_path / "n.png", 80))
    defect = ImageRecord(image_id=81, sample_id=81, path=_defect(tmp_path / "d.png", 81))

    clean, marked = model.predict([normal, defect], infer_ctx)
    assert marked.score > clean.score

    assert marked.anomaly_map is not None
    values = np.load(marked.anomaly_map)
    inside = float(values[STAMP, STAMP].mean())
    outside = float(np.mean(np.delete(values, np.s_[STAMP], axis=0)))
    assert inside > outside


def test_the_two_fusion_groups_are_emitted_as_diagnostics(
    fitted: tuple[DinomalyCustomModel, Path, list[float]], tmp_path: Path
) -> None:
    """The stored map is a mean of two, and both halves are recorded rather than averaged away."""
    model, root, _ = fitted
    _, infer_ctx = _contexts(root)
    probe = ImageRecord(image_id=85, sample_id=85, path=_defect(tmp_path / "g.png", 85))
    model.predict([probe], infer_ctx)
    infer_ctx.diagnostics.flush()

    entries = load_index(root / "artifacts" / "diagnostics").entries
    keys = {entry.key for entry in entries if entry.scope is DiagnosticScope.IMAGE}
    assert {"map_group_0", "map_group_1"} <= keys
    for entry in entries:
        if entry.key.startswith("map_group_"):
            assert entry.kind is DiagnosticKind.MAP


def test_the_plan_reaches_the_run_as_a_table(
    fitted: tuple[DinomalyCustomModel, Path, list[float]],
) -> None:
    _, root, _ = fitted
    entries = load_index(root / "artifacts" / "diagnostics").entries
    plan = next(entry for entry in entries if entry.key == "training_plan")
    assert plan.kind is DiagnosticKind.TABLE


def test_the_plan_is_logged_before_the_first_forward_pass(tmp_path: Path) -> None:
    """`describe()` is arithmetic, so it can be said before anything is allocated."""
    recorder = _Recorder()
    train_ctx, _ = _contexts(tmp_path, reporter=recorder)
    model = DinomalyCustomModel(_config(max_steps=2))
    model.fit(_train_records(tmp_path, count=2), train_ctx)

    assert recorder.logs[0].startswith("dinomaly_custom plan:")
    assert "15,360,000 trainable parameters" in recorder.logs[0]
    assert any("frozen encoder parameters" in line for line in recorder.logs)


def test_predicting_before_fitting_is_an_error_not_a_zero(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path)
    model = DinomalyCustomModel(_config())
    probe = ImageRecord(image_id=1, sample_id=1, path=_normal(tmp_path / "p.png", 1))
    with pytest.raises(RuntimeError, match="before it was fitted or loaded"):
        model.predict([probe], infer_ctx)


def test_one_seed_is_one_experiment_and_another_seed_is_another(tmp_path: Path) -> None:
    """Both directions. Pinning a seed only means something if changing it changes the answer."""
    digests: list[str] = []
    for seed in (7, 7, 8):
        root = tmp_path / f"seed{seed}{len(digests)}"
        train_ctx, _ = _contexts(root)
        model = DinomalyCustomModel(_config(max_steps=6, seed=seed))
        model.fit(_train_records(root, count=4), train_ctx)
        digests.append(_digest(model))

    assert digests[0] == digests[1]
    assert digests[0] != digests[2]


def test_the_capability_flag_and_the_resume_protocol_agree() -> None:
    """A flag without the protocol behind it is a lie the train handler would have to catch."""
    capabilities = DinomalyCustomModel.capabilities()
    assert capabilities.supports_resume is True
    assert isinstance(DinomalyCustomModel(_config()), SupportsResume)
    assert capabilities.produces_diagnostics is True
    assert capabilities.channel_aware is False
    # The honest asymmetry with dinomaly_anomalib, asserted rather than left to a comment.
    assert capabilities.portable_formats == []


def test_cancellation_stops_training_within_a_step(tmp_path: Path) -> None:
    recorder = _Recorder(cancel_after=3)
    train_ctx, _ = _contexts(tmp_path, reporter=recorder)
    model = DinomalyCustomModel(_config(max_steps=500))

    with pytest.raises(ModelCancelledError):
        model.fit(_train_records(tmp_path, count=3), train_ctx)
    assert model.completed_steps() <= 4


# ------------------------------------------------------------------------- resume


def test_a_checkpoint_carries_what_a_continuation_needs(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=5, seed=3))
    model.fit(_train_records(tmp_path, count=4), train_ctx)
    model.save(train_ctx.artifact_dir)

    stored = torch.load(
        train_ctx.artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False
    )
    assert stored["format"] == 1
    assert stored["completed_steps"] == 5
    assert stored["decoder_depth"] == 8
    assert stored["fitted_size"] == [SIZE, SIZE]
    assert stored["optimizer"] is not None
    assert stored["torch_rng_state"] is not None
    assert stored["generator_state"] is not None
    assert stored["encoder_fingerprint"]
    # The encoder is represented by its fingerprint, never copied into the run.
    assert set(stored["trainable"]) == {"bottleneck", "decoder"}


def test_thirty_plus_thirty_is_sixty(tmp_path: Path) -> None:
    """A continuation reproduces an uninterrupted run exactly — in the weights and the score.

    This is what "exact" has to mean, and every stream has to be back where it stopped for it
    to hold: the optimizer's moments, the torch stream the bottleneck dropout draws from, and
    the numpy generator that chooses training images. Missing any one of them still trains,
    still reduces the loss, and quietly makes a continued run a different experiment from an
    uninterrupted one.

    It also depends on the schedule horizon being fixed rather than following `max_steps` —
    see `SCHEDULE_STEPS`. The 30-step leg and the 60-step run read the same learning rate at
    every absolute step because neither of them decides where the cosine ends.
    """
    straight_root = tmp_path / "straight"
    straight_train, straight_infer = _contexts(straight_root)
    straight = DinomalyCustomModel(_config(max_steps=60, seed=4))
    straight.fit(_train_records(straight_root, count=5), straight_train)
    probe = ImageRecord(image_id=9, sample_id=9, path=_defect(tmp_path / "probe.png", 9))
    expected = straight.predict([probe], straight_infer)[0]

    leg_root = tmp_path / "leg"
    leg_train, leg_infer = _contexts(leg_root)
    leg_records = _train_records(leg_root, count=5)
    leg = DinomalyCustomModel(_config(max_steps=30, seed=4))
    leg.fit(leg_records, leg_train)
    assert leg.completed_steps() == 30
    leg.save(leg_train.artifact_dir)

    # A continuation goes through the checkpoint, not through a live object: that is the only
    # path the job layer ever takes, and it is where a dropped random stream would hide.
    continued = DinomalyCustomModel(_config(max_steps=30, seed=4))
    continued.load(leg_train.artifact_dir)
    assert continued.completed_steps() == 30
    continued.fit_more(leg_records, leg_train, additional_steps=30)
    assert continued.completed_steps() == 60

    actual = continued.predict([probe], leg_infer)[0]
    assert actual.score == expected.score
    assert _digest(continued) == _digest(straight)


def test_a_continuation_at_another_prepared_size_is_refused_by_name(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=3))
    records = _train_records(tmp_path, count=3)
    model.fit(records, train_ctx)

    wider_train, _ = _contexts(tmp_path / "wider", size=126)
    with pytest.raises(RuntimeError, match=r"trained at 112x112.*configured for 126x126"):
        model.fit_more(records, wider_train, additional_steps=1)


def test_a_continuation_without_optimizer_state_is_refused_rather_than_restarted(
    tmp_path: Path,
) -> None:
    """A continuation that resets Adam's moments is not a continuation."""
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=3))
    records = _train_records(tmp_path, count=3)
    model.fit(records, train_ctx)
    model.save(train_ctx.artifact_dir)

    checkpoint = train_ctx.artifact_dir / STATE_FILENAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["optimizer"] = None
    torch.save(payload, checkpoint)

    restored = DinomalyCustomModel(_config(max_steps=3))
    restored.load(train_ctx.artifact_dir)
    with pytest.raises(RuntimeError, match="carries no optimizer state"):
        restored.fit_more(records, train_ctx, additional_steps=1)


# ---------------------------------------------------------------------- refusals


def test_a_reloaded_model_scores_exactly_what_the_fitted_one_did(tmp_path: Path) -> None:
    train_ctx, infer_ctx = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=8, seed=5))
    model.fit(_train_records(tmp_path, count=4), train_ctx)
    probe = ImageRecord(image_id=50, sample_id=50, path=_defect(tmp_path / "p.png", 50))
    before = model.predict([probe], infer_ctx)[0]
    model.save(train_ctx.artifact_dir)

    restored = DinomalyCustomModel(_config(max_steps=8, seed=5))
    restored.load(train_ctx.artifact_dir)
    after = restored.predict([probe], infer_ctx)[0]

    assert after.score == before.score
    assert restored.completed_steps() == 8


def test_load_refuses_a_checkpoint_format_it_does_not_read(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=2))
    model.fit(_train_records(tmp_path, count=2), train_ctx)
    model.save(train_ctx.artifact_dir)

    checkpoint = train_ctx.artifact_dir / STATE_FILENAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format"] = 99
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="declares format 99"):
        DinomalyCustomModel(_config(max_steps=2)).load(train_ctx.artifact_dir)


def test_load_refuses_changed_encoder_weights(tmp_path: Path) -> None:
    """The fingerprint turns "the weights might have changed" into a check."""
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=2))
    model.fit(_train_records(tmp_path, count=2), train_ctx)
    model.save(train_ctx.artifact_dir)

    checkpoint = train_ctx.artifact_dir / STATE_FILENAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["encoder_fingerprint"] = "0" * 64
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match="are not the ones this experiment was fitted"):
        DinomalyCustomModel(_config(max_steps=2)).load(train_ctx.artifact_dir)


def test_load_refuses_a_different_encoder_by_name(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=2))
    model.fit(_train_records(tmp_path, count=2), train_ctx)
    model.save(train_ctx.artifact_dir)

    other = DinomalyCustomModel(_config(max_steps=2, encoder=DinoBackbone.DINOV2_VIT_S14))
    with pytest.raises(RuntimeError, match=r"fitted on encoder 'dinov2_vit_s14_reg4'"):
        other.load(train_ctx.artifact_dir)


# ------------------------------------------------------------------- decoder depth


def test_a_four_block_decoder_trains_and_produces_maps(tmp_path: Path) -> None:
    """The configurability claim, run rather than asserted.

    `dinomaly_anomalib` cannot do this: anomalib's fusion topology indexes eight decoder
    outputs whatever depth its constructor was given. Here the groups are derived, so a
    four-block decoder fuses outputs 0-1 against 2-3 and the encoder keeps its own split.
    """
    recorder = _Recorder()
    train_ctx, infer_ctx = _contexts(tmp_path, reporter=recorder)
    model = DinomalyCustomModel(_config(max_steps=20, decoder_depth=4, seed=6))
    model.fit(_train_records(tmp_path, count=4), train_ctx)

    assert model._net.decoder_groups == ((0, 1), (2, 3))
    assert model._net.encoder_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert len(model._net.decoder) == 4
    assert sum(p.numel() for p in model._net.trainable_parameters()) == (
        trainable_parameter_count(BACKBONES[ENCODER].embedding_dim, 4)
    )

    probe = ImageRecord(image_id=60, sample_id=60, path=_defect(tmp_path / "p.png", 60))
    prediction = model.predict([probe], infer_ctx)[0]
    assert prediction.anomaly_map is not None
    assert np.load(prediction.anomaly_map).shape == (SIZE, SIZE)
    assert recorder.losses[-1][1] < recorder.losses[0][1]


def test_the_other_encoder_family_runs_on_the_same_pixels(tmp_path: Path) -> None:
    """The other half of the configurability claim: a DINOv3 encoder, end to end.

    `dinomaly_anomalib` cannot reach one at all. It matters that this is a *different timm
    class* — DINOv2 resolves to `VisionTransformer` and DINOv3 to `Eva`, with a different
    patch size, a rotary position embedding and its own `forward_intermediates`. That one
    `extract_layer_tokens` call serves both without a branch is what makes the encoder a
    field rather than a two-way switch.

    Hermetic: the weights are licence-gated, but `pretrained_encoder=False` builds the same
    architecture from the seed and needs no account and no network. 112 divides by 16 as well
    as by 14, so both families see the same pixels — a 7x7 grid here against DINOv2's 8x8.
    """
    train_ctx, infer_ctx = _contexts(tmp_path)
    model = DinomalyCustomModel(
        _config(encoder=DinoBackbone.DINOV3_VIT_S16, max_steps=3, decoder_depth=2)
    )
    model.fit(_train_records(tmp_path, count=3), train_ctx)

    assert model._net.patch_size == 16
    assert model._net.num_prefix_tokens == 5

    probe = ImageRecord(image_id=64, sample_id=64, path=_defect(tmp_path / "p.png", 64))
    prediction = model.predict([probe], infer_ctx)[0]
    values = np.load(prediction.anomaly_map)  # type: ignore[arg-type]
    assert values.shape == (SIZE, SIZE)
    assert np.isfinite(values).all()


def test_a_checkpoint_records_its_depth_and_refuses_a_mismatch(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    model = DinomalyCustomModel(_config(max_steps=2, decoder_depth=4))
    model.fit(_train_records(tmp_path, count=2), train_ctx)
    model.save(train_ctx.artifact_dir)

    deeper = DinomalyCustomModel(_config(max_steps=2, decoder_depth=8))
    with pytest.raises(
        RuntimeError, match=r"holds a 4-block decoder and the experiment asks for 8"
    ):
        deeper.load(train_ctx.artifact_dir)


# ---------------------------------------------------------------------- map blur


def test_map_blur_smooths_the_stored_map_without_moving_the_score(tmp_path: Path) -> None:
    """The optional map blur is a display-side field, and the score is independent of it.

    A score that moved with the smoothing would make `map_blur_sigma` a scoring
    hyperparameter disguised as a display one — `dino_memory`'s rule, one method over.
    """
    train_ctx, infer_ctx = _contexts(tmp_path)
    records = _train_records(tmp_path, count=4)
    sharp = DinomalyCustomModel(_config(max_steps=8, seed=9))
    sharp.fit(records, train_ctx)
    sharp.save(train_ctx.artifact_dir)

    probe = ImageRecord(image_id=40, sample_id=40, path=_defect(tmp_path / "p.png", 40))
    plain = sharp.predict([probe], infer_ctx)[0]
    plain_values = np.load(plain.anomaly_map)  # type: ignore[arg-type]

    blurred_model = DinomalyCustomModel(_config(max_steps=8, seed=9, map_blur_sigma=4.0))
    blurred_model.load(train_ctx.artifact_dir)
    blurred = blurred_model.predict([probe], infer_ctx)[0]
    blurred_values = np.load(blurred.anomaly_map)  # type: ignore[arg-type]

    assert blurred.score == pytest.approx(plain.score, abs=1e-6)
    assert float(blurred_values.std()) < float(plain_values.std())
    assert blurred_values.shape == plain_values.shape
