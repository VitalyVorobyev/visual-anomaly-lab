"""`dino_memory` as a plugin: does each memory detect, is it bounded, can it be stopped.

**Nothing here downloads anything.** Every encoder is built with `pretrained_backbone=False`
— a seeded random ViT — and that is not a compromise made to keep CI cheap. It was measured
first: on the stamped-defect fixture all three scoring rules separate cleanly, because what
they need is *some* feature space and an untrained transformer is one. The published weights
make them better on real data and are not what this file is testing.

The centrepiece is not the stamped defect, which every method here passes. It is
`test_a_per_position_bank_finds_what_a_global_bank_explains_away`: `scoring` exists as an
axis because the three rules fail differently, and a fixture where a pattern is normal *at
one position* and an anomaly *at another* is the only place that difference is visible. If
that test ever passes for `global_knn` too, the axis has stopped meaning anything.

The prepared size is 112 throughout, which divides by 14 and by 16 — so both encoder families
are exercised on identical pixels rather than on two resizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from anomaly_lab.eval.metrics import roc_auc  # noqa: E402
from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
    Prediction,
    SupportsResume,
    TrainContext,
)
from anomaly_lab.models.diagnostics import (  # noqa: E402
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
from anomaly_lab.models.dino_backbone import DinoBackbone, FeatureLayers  # noqa: E402
from anomaly_lab.models.dino_memory import (  # noqa: E402
    STATE_FILENAME,
    ChannelFusion,
    DinoMemoryConfig,
    DinoMemoryModel,
    Scoring,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402

SIZE = 112
"""Divisible by 14 (8x8 tokens) and by 16 (7x7), so one fixture serves both families and a
forward pass over an untrained ViT-S is a fraction of a second."""

STAMP = slice(42, 70)
"""Rows and columns of the bright square. Two whole /14 patches wide and centred, so the
map's maximum has somewhere honest to land."""


# ------------------------------------------------------------------------ fixtures


def _field(seed: int) -> np.ndarray:
    """A shallow seeded gradient with light noise — a scene, not a random image.

    Deliberately shallow. A steep gradient makes normal patch appearance vary so much across
    the frame that a *global* bank contains something close to almost anything, which is a
    real property of the method and the wrong thing for a detection fixture to measure. The
    `A`/`B` fixture below arranges that case on purpose instead.
    """
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


def _contexts(root: Path) -> tuple[TrainContext, InferContext]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig(width=SIZE, height=SIZE)
    diagnostics = artifacts / "diagnostics"

    return (
        TrainContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=preprocessing,
            device=Device.CPU,
            reporter=NullReporter(),
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


def _config(**overrides: object) -> DinoMemoryConfig:
    """A fit that needs no network and finishes in well under a second."""
    base: dict[str, object] = {
        "pretrained_backbone": False,
        "allow_downloads": False,
        "max_bank_images": 6,
        "max_candidate_vectors": 100_000,
        "coreset_ratio": 0.5,
        "per_position_images": 6,
        "mahalanobis_dims": 16,
        "blur_sigma": 2.0,
        "seed": 0,
    }
    base.update(overrides)
    return DinoMemoryConfig(**base)  # type: ignore[arg-type]


@dataclass
class Fitted:
    """One fitted memory and everything the assertions read off it.

    A module fixture, because fitting is the expensive part and most of these tests only
    read. Adding a test that fits its own model is a decision about the `dl` job's runtime.
    """

    model: DinoMemoryModel
    predictions: list[Prediction]
    labels: np.ndarray
    probe: list[ImageRecord]
    train_ctx: TrainContext
    infer_ctx: InferContext


@pytest.fixture(scope="module")
def fits(tmp_path_factory: pytest.TempPathFactory) -> dict[Scoring, Fitted]:
    """One fit per scoring rule, over one shared set of images."""
    root = tmp_path_factory.mktemp("dino-memory")
    train = [
        ImageRecord(image_id=index, sample_id=index, path=_normal(root / f"n{index}.png", index))
        for index in range(8)
    ]
    probe = [
        ImageRecord(
            image_id=100 + index,
            sample_id=100 + index,
            path=_normal(root / f"p{index}.png", 50 + index),
        )
        for index in range(4)
    ] + [
        ImageRecord(
            image_id=200 + index,
            sample_id=200 + index,
            path=_defect(root / f"d{index}.png", 70 + index),
        )
        for index in range(4)
    ]
    labels = np.array([False] * 4 + [True] * 4)

    fitted: dict[Scoring, Fitted] = {}
    for scoring in Scoring:
        train_ctx, infer_ctx = _contexts(root / scoring.value)
        model = DinoMemoryModel(_config(scoring=scoring))
        model.fit(train, train_ctx)
        train_ctx.diagnostics.flush()
        predictions = model.predict(probe, infer_ctx)
        infer_ctx.diagnostics.flush()
        fitted[scoring] = Fitted(
            model=model,
            predictions=predictions,
            labels=labels,
            probe=probe,
            train_ctx=train_ctx,
            infer_ctx=infer_ctx,
        )
    return fitted


# ------------------------------------------------------------------ the detection bar


@pytest.mark.parametrize("scoring", list(Scoring))
def test_every_memory_separates_a_stamped_defect(
    scoring: Scoring, fits: dict[Scoring, Fitted]
) -> None:
    """The test this method exists to pass, once per scoring rule.

    A hard bar on a deliberately easy task, in the shape M6 settled on: the measured margins
    are 3.4x, 8.7x and 2.7x, so this passes comfortably or something is genuinely broken. A
    softer bar on a harder fixture would flake and teach us to lower it.
    """
    fitted = fits[scoring]
    scores = np.array([prediction.score for prediction in fitted.predictions])
    labels = fitted.labels

    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert float(scores[labels].min()) > 1.5 * float(scores[~labels].max())
    assert bool(np.isfinite(scores).all())


@pytest.mark.parametrize("scoring", list(Scoring))
def test_the_defect_lands_where_the_defect_is(
    scoring: Scoring, fits: dict[Scoring, Fitted]
) -> None:
    """A score can be right for the wrong reason, and a map is how you tell.

    The image score is a percentile and is therefore decided by whatever is hottest, defect
    or not. The stamp is at rows and columns 42..70 of 112, so the map's maximum has to fall
    inside it — otherwise the separation above is a coincidence of this fixture rather than
    localization.
    """
    defect = fits[scoring].predictions[-1]
    assert defect.anomaly_map is not None
    values = np.load(defect.anomaly_map)

    row, col = np.unravel_index(int(np.argmax(values)), values.shape)
    assert STAMP.start <= row < STAMP.stop
    assert STAMP.start <= col < STAMP.stop


# ------------------------------------------------------- what the scoring axis is for


def _placed(path: Path, seed: int, first: float, second: float) -> Path:
    """A flat scene with two marked squares, at two fixed positions.

    Every training image carries pattern `first` in the top-left square and pattern `second`
    in the bottom-right one, so *both* appearances are normal — the question is only whether
    they are normal *where they are*.
    """
    rng = np.random.default_rng(seed)
    values = np.full((SIZE, SIZE), 110.0) + rng.normal(0.0, 3.0, size=(SIZE, SIZE))
    values[14:42, 14:42] = first
    values[70:98, 70:98] = second
    return _save(path, np.clip(values, 0.0, 255.0))


def test_a_per_position_bank_finds_what_a_global_bank_explains_away(tmp_path: Path) -> None:
    """The reason `scoring` is an axis rather than a constant.

    Every training image has a dark square at one position and a bright square at another,
    so both appearances are in the memory. The test image has the *bright* square at the
    *dark* square's position: nothing new has appeared, something has moved.

    A global bank is position-blind by construction, so it finds the misplaced patch's twin
    somewhere else and reports nothing — measured at 0.97x, which is to say the anomalous
    image scores slightly *lower* than a normal one. A per-position bank compares that patch
    only against what belongs there and scores it about twenty times higher.

    The bar is a strict ordering rather than an absolute, because what is being asserted is
    that the two rules disagree in a specific direction, not that either lands on a number.
    """
    train = [
        ImageRecord(
            image_id=index,
            sample_id=index,
            path=_placed(tmp_path / f"n{index}.png", index, 30.0, 225.0),
        )
        for index in range(8)
    ]
    probe = [
        ImageRecord(image_id=100, sample_id=100, path=_placed(tmp_path / "p.png", 60, 30.0, 225.0)),
        ImageRecord(
            image_id=200, sample_id=200, path=_placed(tmp_path / "s.png", 61, 225.0, 225.0)
        ),
    ]

    margins: dict[Scoring, float] = {}
    for scoring in (Scoring.GLOBAL_KNN, Scoring.LOCAL_KNN):
        train_ctx, infer_ctx = _contexts(tmp_path / scoring.value)
        model = DinoMemoryModel(
            _config(
                scoring=scoring,
                max_bank_images=8,
                per_position_images=8,
                coreset_ratio=1.0,
                window_radius=0,
                blur_sigma=0.0,
            )
        )
        model.fit(train, train_ctx)
        normal, swapped = model.predict(probe, infer_ctx)
        margins[scoring] = swapped.score / normal.score

    assert margins[Scoring.GLOBAL_KNN] < 1.2, "a position-blind bank has seen this patch"
    assert margins[Scoring.LOCAL_KNN] > 3.0, "a per-position bank has not seen it *here*"
    assert margins[Scoring.LOCAL_KNN] > 5.0 * margins[Scoring.GLOBAL_KNN]


# ------------------------------------------------------------------ the plugin contract


def test_predictions_come_back_one_per_input_in_order(fits: dict[Scoring, Fitted]) -> None:
    """The contract the infer handler checks; assert it at the plugin too."""
    fitted = fits[Scoring.GLOBAL_KNN]
    assert [p.image_id for p in fitted.predictions] == [r.image_id for r in fitted.probe]
    assert all(p.inference_ms > 0.0 for p in fitted.predictions)


@pytest.mark.parametrize("scoring", list(Scoring))
def test_maps_are_two_dimensional_at_the_prepared_size(
    scoring: Scoring, fits: dict[Scoring, Fitted]
) -> None:
    for prediction in fits[scoring].predictions:
        assert prediction.anomaly_map is not None
        stored = np.load(prediction.anomaly_map)
        assert stored.dtype == np.float32
        assert stored.shape == (SIZE, SIZE)


def test_predicting_before_fitting_is_an_error_not_a_zero(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path)
    record = ImageRecord(image_id=1, sample_id=1, path=_normal(tmp_path / "n.png", 0))

    model = DinoMemoryModel(_config())
    with pytest.raises(RuntimeError, match="before it was fitted or loaded"):
        model.predict([record], infer_ctx)


@pytest.mark.parametrize("scoring", list(Scoring))
def test_one_seed_is_one_experiment_and_a_different_seed_is_a_different_one(
    scoring: Scoring, tmp_path: Path
) -> None:
    """Both directions, because pinning a seed only means something if changing it changes.

    This is M6's lesson reaching a third place. The seed here has to arrive in the encoder's
    own initialisation (torch's *global* stream, which `load_backbone` seeds on the line
    above the constructor), in the coreset's projection and starting point, and in the
    Mahalanobis dimension subset — and each mode uses a different subset of those. Comparing
    the *scores* rather than a bank tensor is what makes one assertion cover all three.
    """
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(4)
    ]
    probe = [ImageRecord(image_id=99, sample_id=99, path=_defect(tmp_path / "d.png", 9))]

    def scores(seed: int, tag: str) -> list[float]:
        train_ctx, infer_ctx = _contexts(tmp_path / f"{scoring.value}-{tag}")
        model = DinoMemoryModel(
            _config(scoring=scoring, seed=seed, max_bank_images=4, per_position_images=4)
        )
        model.fit(train, train_ctx)
        return [p.score for p in model.predict(probe, infer_ctx)]

    first = scores(0, "a")
    again = scores(0, "b")
    other = scores(1, "c")

    assert first == pytest.approx(again, rel=1e-9)
    assert first != pytest.approx(other, rel=1e-6)


def test_the_capability_flags_match_what_the_plugin_actually_does() -> None:
    """A flag the code does not honour is worse than an absent one.

    `supports_resume` is the sharp case: the train handler asks `isinstance` rather than
    trusting the flag, so a memory that declared it without satisfying `SupportsResume` would
    fail inside a continuation. `portable_formats` is empty rather than conditional because
    the export offer is made from the registry before any configuration is read, and
    `feature_concat` has no single-input graph at all.
    """
    capabilities = DinoMemoryModel.capabilities()

    assert capabilities.requires_training is True
    assert capabilities.produces_anomaly_map is True
    assert capabilities.produces_diagnostics is True
    assert capabilities.supports_resume is False
    assert not isinstance(DinoMemoryModel(_config()), SupportsResume)
    assert capabilities.portable_formats == []
    assert capabilities.dataset_specific is False
    assert capabilities.preferred_device is Device.MPS
    # True regardless of `channel_fusion`: the capability says the model *may* consume
    # channel metadata, and a flag that flipped with configuration could not be read from
    # the registry, which is where the UI reads it.
    assert capabilities.channel_aware is True
    assert (
        DinoMemoryModel(_config(channel_fusion=ChannelFusion.PER_IMAGE))
        .capabilities()
        .channel_aware
        is True
    )


def test_availability_names_the_optional_dependency_group() -> None:
    availability = DinoMemoryModel.availability()
    assert availability.available is True, "this file only runs with the dl extra installed"
    assert "DINO patch memory" in (DinoMemoryModel.title or "")


# ----------------------------------------------------------------------- the bounding


class _Recorder(NullReporter):
    """Every log line and every progress message, in the order they were emitted."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def log(self, message: str, level: str = "info") -> None:
        self.events.append(message)

    def progress(self, fraction: float, message: str | None = None) -> None:
        if message:
            self.events.append(message)

    def index_of(self, needle: str) -> int:
        return next(index for index, text in enumerate(self.events) if needle in text)


def test_both_caps_bind_and_the_plan_is_announced_before_the_first_forward(
    tmp_path: Path,
) -> None:
    """Exit criterion one is "it fits and says what it cost", said *before* it is paid.

    The plan is computed from `patch_grid`'s arithmetic and the backbone table with no probe
    at all, which is why `describe()` can precede even loading the encoder. Both caps bind
    here at once: 8 images against a limit of 4, and 4x64 patches against a pool of 100.
    """
    recorder = _Recorder()
    train_ctx, _ = _contexts(tmp_path)
    train_ctx.reporter = recorder
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(8)
    ]

    model = DinoMemoryModel(
        _config(max_bank_images=4, max_candidate_vectors=100, coreset_ratio=0.5)
    )
    model.fit(train, train_ctx)

    plan = model._plan
    assert plan is not None
    assert plan.units_used == 4 and plan.images_available == 8
    assert plan.patches_kept_per_image == 25 and plan.patches_per_image == 64
    assert plan.candidates_kept == 100
    assert plan.coreset_size == 50
    assert model._bank.shape == (50, plan.fused_dim)

    assert model._plan is not None
    assert recorder.index_of("dino_memory plan") < recorder.index_of("loading ")
    assert recorder.index_of("loading ") < recorder.index_of("embedded ")
    dropped = [text for text in recorder.events if "are not in the memory" in text]
    assert dropped and "4 of 8 training images" in dropped[0]
    assert any("of 64 patches per unit are dropped" in text for text in recorder.events)


def test_a_memory_that_fits_uncapped_drops_nothing(tmp_path: Path) -> None:
    """The caps must be bounds, not a tax every run pays."""
    train_ctx, _ = _contexts(tmp_path)
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(4)
    ]

    model = DinoMemoryModel(_config(max_bank_images=64, max_candidate_vectors=100_000))
    model.fit(train, train_ctx)

    plan = model._plan
    assert plan is not None
    assert plan.units_used == 4
    assert plan.patches_kept_per_image == plan.patches_per_image == 64
    assert plan.candidates_kept == 4 * 64
    assert plan.images_dropped == 0 and plan.patches_dropped_per_image == 0
    assert "nothing dropped" in plan.describe()


def test_the_selection_stops_when_the_job_is_cancelled(tmp_path: Path) -> None:
    """The property owning the greedy loop was for.

    A selection that returns only when it is finished is a job that reports nothing and
    ignores cancel for as long as it runs. The reporter here arms as soon as the selection
    announces itself, so what is asserted is that cancellation lands *inside* the loop and
    that nothing half-built is left behind claiming to be a model.
    """

    class CancelAtSelection(_Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.calls = 0

        def progress(self, fraction: float, message: str | None = None) -> None:
            super().progress(fraction, message)
            if message and "projecting" in message:
                self.armed = True

        def should_cancel(self) -> bool:
            self.calls += 1
            return self.armed

    reporter = CancelAtSelection()
    train_ctx, _ = _contexts(tmp_path)
    train_ctx.reporter = reporter
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(6)
    ]

    model = DinoMemoryModel(_config(coreset_ratio=1.0))
    with pytest.raises(ModelCancelledError):
        model.fit(train, train_ctx)

    assert reporter.calls > 0
    assert not any("selected " in text for text in reporter.events), "cancel landed in the loop"
    assert model._bank is None
    with pytest.raises(RuntimeError, match="before it was fitted or loaded"):
        model.predict(train[:1], _contexts(tmp_path / "later")[1])


# ------------------------------------------------------------------------ persistence


@pytest.mark.parametrize("scoring", list(Scoring))
def test_a_saved_memory_reloads_to_the_same_scores(
    scoring: Scoring, fits: dict[Scoring, Fitted], tmp_path: Path
) -> None:
    fitted = fits[scoring]
    destination = tmp_path / scoring.value
    destination.mkdir(parents=True, exist_ok=True)
    fitted.model.save(destination)

    reloaded = DinoMemoryModel(_config(scoring=scoring))
    reloaded.load(destination)
    again = reloaded.predict(fitted.probe, fitted.infer_ctx)

    for before, after in zip(fitted.predictions, again, strict=True):
        assert after.score == pytest.approx(before.score, rel=1e-6)


def test_the_checkpoint_does_not_carry_the_encoder(
    fits: dict[Scoring, Fitted], tmp_path: Path
) -> None:
    """A fingerprint instead of an encoder, and this is the assertion that keeps it that way.

    If somebody later "fixes" the fingerprint check by storing the weights, every experiment
    in a comparison grows by the size of a ViT and this fails loudly rather than the disk
    filling quietly.
    """
    fits[Scoring.GLOBAL_KNN].model.save(tmp_path)
    stored = torch.load(tmp_path / STATE_FILENAME, map_location="cpu", weights_only=False)

    assert isinstance(stored["backbone_fingerprint"], str)
    assert len(stored["backbone_fingerprint"]) == 64
    assert set(stored) & {"encoder", "state_dict", "backbone_state"} == set()
    assert not any(str(key).startswith(("blocks", "patch_embed")) for key in stored)
    assert (tmp_path / STATE_FILENAME).stat().st_size < 2_000_000


def test_a_changed_encoder_is_refused_by_name(fits: dict[Scoring, Fitted], tmp_path: Path) -> None:
    """The weights are an input to the experiment, so a run has to notice them changing.

    The memory was built in the old feature space, so distances against a new one are not
    comparable — and the failure this guards against is that nothing would say so: the
    checkpoint loads, inference runs, and the number is plausible and wrong.
    """
    fits[Scoring.GLOBAL_KNN].model.save(tmp_path)
    stored = torch.load(tmp_path / STATE_FILENAME, map_location="cpu", weights_only=False)
    stored["backbone_fingerprint"] = "0" * 64
    torch.save(stored, tmp_path / STATE_FILENAME)

    reloaded = DinoMemoryModel(_config())
    with pytest.raises(RuntimeError, match="dinov2_vit_s14_reg4") as failure:
        reloaded.load(tmp_path)
    assert "not comparable" in str(failure.value)


def test_an_unknown_checkpoint_format_is_refused(
    fits: dict[Scoring, Fitted], tmp_path: Path
) -> None:
    fits[Scoring.GLOBAL_KNN].model.save(tmp_path)
    stored = torch.load(tmp_path / STATE_FILENAME, map_location="cpu", weights_only=False)
    stored["format"] = 99
    torch.save(stored, tmp_path / STATE_FILENAME)

    with pytest.raises(RuntimeError, match="unsupported dino_memory checkpoint format 99"):
        DinoMemoryModel(_config()).load(tmp_path)


def test_a_checkpoint_of_another_scoring_rule_is_refused_by_name(
    fits: dict[Scoring, Fitted], tmp_path: Path
) -> None:
    """Three memories over one feature space, and one cannot be read as another.

    Refused at load rather than at the first einsum, where the failure would be a shape
    error naming a tensor instead of the two configurations that disagree.
    """
    fits[Scoring.GLOBAL_KNN].model.save(tmp_path)

    reloaded = DinoMemoryModel(_config(scoring=Scoring.LOCAL_KNN))
    with pytest.raises(RuntimeError, match="local_knn") as failure:
        reloaded.load(tmp_path)
    assert "global_knn memory" in str(failure.value)


# ---------------------------------------------------------------------- the channels


def _channel_records(
    root: Path,
    channels: tuple[str, ...],
    *,
    samples: int,
    offsets: dict[str, float] | None = None,
    first_id: int = 0,
    defective: bool = False,
) -> list[ImageRecord]:
    """One sample per index, one image per channel, optionally at different brightnesses."""
    records: list[ImageRecord] = []
    shifts = offsets or {}
    for index in range(samples):
        for position, channel in enumerate(channels):
            values = _field(first_id + index) + shifts.get(channel, 0.0)
            if defective:
                values[STAMP, STAMP] = 255.0
            image_id = first_id + index * 16 + position
            records.append(
                ImageRecord(
                    image_id=image_id,
                    sample_id=first_id + index,
                    channel=channel,
                    path=_save(root / f"{channel}-{first_id + index}.png", np.clip(values, 0, 255)),
                )
            )
    return records


def test_per_image_fusion_scores_every_channel_image_on_its_own(tmp_path: Path) -> None:
    """The pooled default: one memory over every channel image, one score per image."""
    channels = ("bright", "dark", "dome")
    offsets = {"bright": 60.0, "dark": -60.0, "dome": 0.0}
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _channel_records(tmp_path / "t", channels, samples=6, offsets=offsets)
    probe = _channel_records(
        tmp_path / "p", channels, samples=1, offsets=offsets, first_id=90, defective=True
    )

    model = DinoMemoryModel(_config(channel_fusion=ChannelFusion.PER_IMAGE, max_bank_images=18))
    model.fit(train, train_ctx)
    predictions = model.predict(probe, infer_ctx)

    assert len(predictions) == 3
    scores = [prediction.score for prediction in predictions]
    assert len(set(scores)) == 3, "three illuminations are three questions, not one"
    assert model._channel_order == ()


def test_feature_concat_scores_a_sample_once_and_writes_it_to_every_channel(
    tmp_path: Path,
) -> None:
    """One part, one verdict — and the map that produced it lands on each of its images.

    Each image still goes through `write_map` on its own, because each carries its own
    source-frame projection: sharing the array does not mean sharing the transform.
    """
    channels = ("bright", "dark", "dome")
    offsets = {"bright": 60.0, "dark": -60.0, "dome": 0.0}
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _channel_records(tmp_path / "t", channels, samples=6, offsets=offsets)
    probe = _channel_records(
        tmp_path / "p", channels, samples=1, offsets=offsets, first_id=90, defective=True
    )

    model = DinoMemoryModel(_config(channel_fusion=ChannelFusion.FEATURE_CONCAT, max_bank_images=6))
    model.fit(train, train_ctx)
    predictions = model.predict(probe, infer_ctx)

    assert model._channel_order == channels
    assert model._plan is not None
    assert model._plan.channels_fused == 3
    assert model._plan.fused_dim == 3 * model._plan.embedding_dim

    assert len(predictions) == 3
    assert len({prediction.score for prediction in predictions}) == 1
    maps = [np.load(prediction.anomaly_map) for prediction in predictions]  # type: ignore[arg-type]
    for other in maps[1:]:
        assert np.array_equal(maps[0], other)


def test_a_sample_missing_a_fitted_channel_is_refused_by_name(tmp_path: Path) -> None:
    """No silent fallback: a fused vector's meaning is positional, so a gap is not a shorter
    vector, it is a different one."""
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _channel_records(tmp_path / "t", ("bright", "dark"), samples=4)

    model = DinoMemoryModel(_config(channel_fusion=ChannelFusion.FEATURE_CONCAT, max_bank_images=4))
    model.fit(train, train_ctx)

    partial = _channel_records(tmp_path / "p", ("bright",), samples=1, first_id=90)
    with pytest.raises(RuntimeError, match="missing dark") as failure:
        model.predict(partial, infer_ctx)
    assert "Sample 90" in str(failure.value)


def test_a_sample_presenting_an_unseen_channel_is_refused_by_name(tmp_path: Path) -> None:
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _channel_records(tmp_path / "t", ("bright", "dark"), samples=4)

    model = DinoMemoryModel(_config(channel_fusion=ChannelFusion.FEATURE_CONCAT, max_bank_images=4))
    model.fit(train, train_ctx)

    extra = _channel_records(tmp_path / "p", ("bright", "dark", "dome"), samples=1, first_id=90)
    with pytest.raises(RuntimeError, match="it presents dome") as failure:
        model.predict(extra, infer_ctx)
    assert "was not fitted on" in str(failure.value)


def test_a_fused_score_does_not_depend_on_how_many_channels_a_sample_has(
    tmp_path: Path,
) -> None:
    """ "Channel count is data, never schema", as a number rather than as a rule.

    A sample whose channels carry identical pixels scores the same whether it has two of
    them or four, because concatenating unit vectors and normalizing again makes the squared
    distance the *average* of the per-channel distances rather than their sum. Asserted in
    `local_knn`, where the bank is a deterministic function of the features alone: the global
    mode's projection matrix is drawn at the fused width, so widening the vector legitimately
    changes which vectors the coreset selects.
    """

    def scores(count: int) -> list[float]:
        channels = tuple(f"c{index}" for index in range(count))
        root = tmp_path / f"c{count}"
        train_ctx, infer_ctx = _contexts(root)
        train = _channel_records(root / "t", channels, samples=6)
        probe = _channel_records(root / "p", channels, samples=1, first_id=90, defective=True)
        model = DinoMemoryModel(
            _config(
                scoring=Scoring.LOCAL_KNN,
                channel_fusion=ChannelFusion.FEATURE_CONCAT,
                max_bank_images=6,
                per_position_images=6,
            )
        )
        model.fit(train, train_ctx)
        return [prediction.score for prediction in model.predict(probe, infer_ctx)]

    two, four = scores(2), scores(4)

    assert len(two) == 2 and len(four) == 4
    assert two[0] == pytest.approx(four[0], rel=1e-5)


def test_on_a_dataset_with_no_channels_fusing_is_exactly_the_pooled_default(
    tmp_path: Path,
) -> None:
    """A group of one takes the same code path with C=1, where the extra normalization of an
    already-unit vector is the identity — so this is equality, not similarity."""
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(6)
    ]
    probe = [ImageRecord(image_id=90, sample_id=90, path=_defect(tmp_path / "d.png", 90))]

    def scores(fusion: ChannelFusion) -> list[float]:
        train_ctx, infer_ctx = _contexts(tmp_path / fusion.value)
        model = DinoMemoryModel(_config(scoring=Scoring.LOCAL_KNN, channel_fusion=fusion))
        model.fit(train, train_ctx)
        return [prediction.score for prediction in model.predict(probe, infer_ctx)]

    assert scores(ChannelFusion.FEATURE_CONCAT) == pytest.approx(
        scores(ChannelFusion.PER_IMAGE), rel=1e-9
    )


def test_diagnosing_one_image_of_a_fused_sample_gives_the_channel_refusal(
    tmp_path: Path,
) -> None:
    """`experiments/diagnose.py` scores a single record, and this is where that lands.

    A known limitation with a backlog item behind it. What matters until then is that the
    failure is the readable channel refusal naming the sample and the channels, rather than
    an index error several frames inside the fusion.
    """
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _channel_records(tmp_path / "t", ("bright", "dark"), samples=4)

    model = DinoMemoryModel(_config(channel_fusion=ChannelFusion.FEATURE_CONCAT, max_bank_images=4))
    model.fit(train, train_ctx)

    with pytest.raises(RuntimeError, match="feature_concat"):
        model.predict([train[0]], infer_ctx)


# --------------------------------------------------------------- the encoder is a field


def test_the_second_family_works_unpretrained_at_the_shared_size(tmp_path: Path) -> None:
    """112 divides by 16 as well as by 14, so DINOv3 sees the same pixels on a 7x7 grid."""
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(4)
    ]
    probe = [
        ImageRecord(image_id=90, sample_id=90, path=_normal(tmp_path / "p.png", 90)),
        ImageRecord(image_id=91, sample_id=91, path=_defect(tmp_path / "d.png", 91)),
    ]

    model = DinoMemoryModel(_config(backbone=DinoBackbone.DINOV3_VIT_S16, max_bank_images=4))
    model.fit(train, train_ctx)
    normal, defect = model.predict(probe, infer_ctx)

    assert model._grid == (7, 7)
    assert model._plan is not None and model._plan.positions == 49
    assert defect.score > normal.score


def test_reading_more_layers_widens_every_stored_vector(tmp_path: Path) -> None:
    """The layer set is not cosmetic: it decides the width of every vector in the plan.

    Worth pinning because the whole memory calculation is units x patches x **width**, and a
    field that silently did nothing would make the plan's arithmetic a fiction — which is
    exactly what `_verify` refuses on the first real batch.
    """
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_normal(tmp_path / f"n{i}.png", i))
        for i in range(4)
    ]

    widths: dict[FeatureLayers, tuple[int, int]] = {}
    for layers in (FeatureLayers.LAST, FeatureLayers.LAST_FOUR):
        train_ctx, _ = _contexts(tmp_path / layers.value)
        model = DinoMemoryModel(_config(layers=layers, max_bank_images=4))
        model.fit(train, train_ctx)
        assert model._plan is not None
        widths[layers] = (model._plan.fused_dim, int(model._bank.shape[1]))

    # ViT-S is 384 wide, and `extract_patch_features` concatenates the requested blocks.
    assert widths[FeatureLayers.LAST] == (384, 384)
    assert widths[FeatureLayers.LAST_FOUR] == (1536, 1536)


def test_a_prepared_frame_that_does_not_divide_is_refused_before_timm_sees_it(
    tmp_path: Path,
) -> None:
    """Refused at the plugin boundary, naming the encoder and the sizes that would work.

    Left to timm this is a reshape error several frames later, in a message about tensor
    shapes that says nothing about which experiment setting is wrong.
    """
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    train_ctx = TrainContext(
        artifact_dir=artifacts,
        cache_dir=tmp_path / "cache",
        preprocessing=PreprocessingConfig(width=110, height=112),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(artifacts / "diagnostics"),
    )
    record = ImageRecord(image_id=1, sample_id=1, path=_normal(tmp_path / "n.png", 0))

    with pytest.raises(ValueError, match=r"dinov2_vit_s14_reg4.*divisible by 14.*110x112"):
        DinoMemoryModel(_config()).fit([record], train_ctx)


# ------------------------------------------------------------ M4's views, inherited


@pytest.mark.parametrize("scoring", list(Scoring))
def test_every_m4_view_works_with_no_new_code(
    scoring: Scoring, fits: dict[Scoring, Fitted]
) -> None:
    """ADR-0018's prediction, tested against a method that holds three different memories.

    The per-mode diagnostic differs — a coverage map, a per-position count, a shrinkage map —
    and every one of them reaches the screen through the same self-describing index, rendered
    by `kind` and never by method name.
    """
    fitted = fits[scoring]
    index = load_index(fitted.train_ctx.diagnostics.root)
    by_key = {entry.key: entry for entry in index.entries if entry.image_id is None}

    assert by_key["memory_bank"].kind is DiagnosticKind.TABLE
    assert all(entry.scope is DiagnosticScope.MODEL for entry in by_key.values())

    expected = {
        Scoring.GLOBAL_KNN: "coreset_coverage",
        Scoring.LOCAL_KNN: "position_counts",
        Scoring.LOCAL_GAUSSIAN: "shrinkage_lambda",
    }[scoring]
    assert by_key[expected].kind is DiagnosticKind.MAP
    assert by_key[expected].shape == [8, 8], "a per-position view is the patch grid"

    # `fit` and `predict` write the same index, merged rather than replaced, so the run's
    # per-image sample sits beside its model-scoped entries.
    per_image = {entry.key for entry in index.entries if entry.image_id is not None}
    assert per_image == {"patch_scores", "map_unblurred"}
    assert "patch_scores" in index.ranges
    assert expected in index.ranges


def test_the_memory_table_names_what_the_caps_kept_and_what_they_avoided(
    fits: dict[Scoring, Fitted],
) -> None:
    """The table *is* the bounding exit criterion: a memory that fits is only half the claim."""
    index = load_index(fits[Scoring.GLOBAL_KNN].train_ctx.diagnostics.root)
    table = next(entry for entry in index.entries if entry.key == "memory_bank")
    assert table.payload is not None
    rows = {row[0]: row[1] for row in table.payload["rows"]}

    assert rows["scoring"] == "global_knn"
    assert rows["patch grid"] == "8x8"
    assert rows["units used"] == "6 of 8"
    assert rows["channel fusion"] == "per_image"
    assert rows["feature width"] == "1 x 768 = 768"
    assert rows["uncapped pool would have been"] == "512 vectors"
    assert "random, seeded 0" in rows["backbone weights"]


def test_the_unblurred_map_is_sharper_than_the_stored_one(fits: dict[Scoring, Fitted]) -> None:
    """`blur_sigma` is a field, and the diagnostic is what makes its effect visible.

    Blurring cannot raise the maximum of a map, so the unsmoothed one has to peak at least as
    high. Asserting the direction rather than a magnitude keeps this a statement about what
    smoothing does rather than about this fixture's numbers.
    """
    fitted = fits[Scoring.GLOBAL_KNN]
    root = fitted.infer_ctx.diagnostics.root
    index = load_index(root)
    defect_id = fitted.probe[-1].image_id
    entry = next(e for e in index.entries if e.key == "map_unblurred" and e.image_id == defect_id)
    assert entry.path is not None

    unblurred = np.load(root / entry.path)
    map_path = next(p.anomaly_map for p in fitted.predictions if p.image_id == defect_id)
    assert map_path is not None
    stored = np.load(map_path)

    assert unblurred.shape == stored.shape == (SIZE, SIZE)
    assert float(unblurred.max()) >= float(stored.max())


def test_the_image_score_is_taken_on_the_patch_grid_and_not_on_the_blur(
    fits: dict[Scoring, Fitted], tmp_path: Path
) -> None:
    """A score that moved with `blur_sigma` would make a display control a scoring one."""
    fitted = fits[Scoring.GLOBAL_KNN]
    fitted.model.save(tmp_path)

    sharp = DinoMemoryModel(_config(blur_sigma=0.0))
    sharp.load(tmp_path)
    smooth = DinoMemoryModel(_config(blur_sigma=8.0))
    smooth.load(tmp_path)

    probe = fitted.probe[-2:]
    assert [p.score for p in sharp.predict(probe, fitted.infer_ctx)] == pytest.approx(
        [p.score for p in smooth.predict(probe, fitted.infer_ctx)], rel=1e-9
    )
