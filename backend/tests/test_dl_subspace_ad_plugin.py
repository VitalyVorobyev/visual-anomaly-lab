"""`subspace_ad` as a plugin: does it detect, does it persist, does it refuse.

**Nothing here downloads anything**, and — unlike `test_dl_dino_memory.py` — that means
this file cannot assert detection. Measured on the stamped-defect fixture below: with the
published DINOv2 ViT-S weights the method reaches AUROC 1.00 at every tail fraction tried,
and with a seeded random ViT it reaches 0.56. A nearest-neighbour bank still works on random
features because *any* metric space gives it a distance; a PCA residual does not, because
what it needs is variance directions that mean something. On an untrained encoder the blocks
are also nearly interchangeable — the pooled features of `last`, `last_four` and
`upper_half` agree to five significant figures — so the axis this method is most sensitive to
is invisible here too.

That is a fact about the method, not a gap in the fixture, and it is left as a fact: the
detection claim is asserted where it can be honest, in
`test_subspace_ad_math.py::test_a_patch_outside_the_normal_subspace_scores_far_higher` on
arrays, and measured on real data by the campaign (`docs/measurements.md`). What follows
tests the plugin — the plumbing, the round trip, and every refusal it can reach.

The prepared size is 112 throughout — divisible by 14 and by 16, so one fixture serves both
encoder families on identical pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    Prediction,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter  # noqa: E402
from anomaly_lab.models.dino_backbone import DinoBackbone, LayerWindow  # noqa: E402
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402
from anomaly_lab.models.subspace import RotationFill, tail_value_at_risk  # noqa: E402
from anomaly_lab.models.subspace_ad import (  # noqa: E402
    MANIFEST_FILENAME,
    STATE_FILENAME,
    SubspaceAdConfig,
    SubspaceAdModel,
)

SIZE = 112
STAMP = slice(42, 70)


# ------------------------------------------------------------------------ fixtures


def _field(seed: int) -> np.ndarray:
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


def _config(**overrides: object) -> SubspaceAdConfig:
    """A fit that needs no network and finishes in a second or two."""
    base: dict[str, object] = {
        "backbone": DinoBackbone.DINOV2_VIT_S14,
        "layers": LayerWindow.UPPER_HALF,
        "pretrained_backbone": False,
        "allow_downloads": False,
        "rotations": 2,
        "max_fit_images": 3,
        "smoothing_sigma": 2.0,
        "seed": 0,
    }
    base.update(overrides)
    return SubspaceAdConfig(**base)  # type: ignore[arg-type]


@dataclass
class Fitted:
    model: SubspaceAdModel
    predictions: list[Prediction]
    probe: list[ImageRecord]
    train_ctx: TrainContext
    infer_ctx: InferContext
    root: Path


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Fitted:
    """One fit, reused — fitting is the expensive part and most assertions only read."""
    root = tmp_path_factory.mktemp("subspace-ad")
    train = [
        ImageRecord(image_id=index, sample_id=index, path=_normal(root / f"n{index}.png", index))
        for index in range(6)
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
    train_ctx, infer_ctx = _contexts(root / "run")
    model = SubspaceAdModel(_config())
    model.fit(train, train_ctx)
    predictions = model.predict(probe, infer_ctx)
    return Fitted(
        model=model,
        predictions=predictions,
        probe=probe,
        train_ctx=train_ctx,
        infer_ctx=infer_ctx,
        root=root,
    )


# ------------------------------------------------------------------------ scoring


def test_every_image_gets_a_finite_score_and_a_map_in_the_prepared_frame(
    fitted: Fitted,
) -> None:
    """One prediction per input, in order, each with a 2-D map the size of the frame."""
    assert [entry.image_id for entry in fitted.predictions] == [
        record.image_id for record in fitted.probe
    ]
    for entry in fitted.predictions:
        assert np.isfinite(entry.score)
        assert entry.score > 0.0
        assert entry.anomaly_map is not None
        values = np.load(entry.anomaly_map)
        assert values.shape == (SIZE, SIZE)
        assert np.isfinite(values).all()


def test_the_score_is_the_tail_of_the_patch_grid_and_not_of_the_map(fitted: Fitted) -> None:
    """The image score is taken on the token grid, before the upsample and the blur.

    Recomputing it from the stored pixel map gives a different number, which is the point:
    if the two agreed, the smoothing would be inside the score.
    """
    entry = fitted.predictions[-1]
    assert entry.anomaly_map is not None
    from_pixels = tail_value_at_risk(np.load(entry.anomaly_map), [0.002])[0.002]
    assert not np.isclose(from_pixels, entry.score)


def test_the_image_score_cannot_be_moved_by_the_smoothing(fitted: Fitted) -> None:
    """`smoothing_sigma` is a display decision. A score that moved with it would make it a
    scoring hyperparameter wearing a display one's name."""
    blurrier = SubspaceAdModel(_config(smoothing_sigma=8.0))
    blurrier._fits = fitted.model._fits
    blurrier._grid = fitted.model._grid
    blurrier._encoder = fitted.model._encoder
    blurrier._stats = fitted.model._stats

    _, infer_ctx = _contexts(fitted.root / "blur")
    again = blurrier.predict(fitted.probe, infer_ctx)
    assert [entry.score for entry in again] == [entry.score for entry in fitted.predictions]


def test_no_smoothing_is_a_setting_rather_than_a_crash(fitted: Fitted) -> None:
    """`smoothing_sigma` is bounded at zero inclusive, and `pixel_map` refuses a zero sigma,
    so the plugin has to mean the absence of a blur rather than pass it down."""
    unsmoothed = SubspaceAdModel(_config(smoothing_sigma=0.0))
    unsmoothed._fits = fitted.model._fits
    unsmoothed._grid = fitted.model._grid
    unsmoothed._encoder = fitted.model._encoder
    unsmoothed._stats = fitted.model._stats

    _, infer_ctx = _contexts(fitted.root / "sharp")
    raw = unsmoothed.predict(fitted.probe, infer_ctx)
    assert [entry.score for entry in raw] == [entry.score for entry in fitted.predictions]

    entry = raw[-1]
    assert entry.anomaly_map is not None
    sharp = np.load(entry.anomaly_map)
    blurred_entry = fitted.predictions[-1]
    assert blurred_entry.anomaly_map is not None
    blurred = np.load(blurred_entry.anomaly_map)
    assert sharp.shape == blurred.shape == (SIZE, SIZE)
    # An unsmoothed map keeps the extremes a blur pulls in.
    assert sharp.max() > blurred.max()


def test_keeping_every_direction_destroys_the_signal(fitted: Fitted) -> None:
    """The paper measures this collapse rather than guarding against it, and so does this.

    At variance=1.0 the residual is the difference between a number and itself, so what
    survives is float noise several orders of magnitude below a real score.
    """
    degenerate = SubspaceAdModel(_config(variance=1.0))
    degenerate._fits = fitted.model._fits
    degenerate._grid = fitted.model._grid
    degenerate._encoder = fitted.model._encoder
    degenerate._stats = fitted.model._stats

    _, infer_ctx = _contexts(fitted.root / "full-rank")
    collapsed = degenerate.predict(fitted.probe, infer_ctx)
    largest = max(abs(entry.score) for entry in collapsed)
    real = max(abs(entry.score) for entry in fitted.predictions)
    assert largest < real * 1e-3


# ------------------------------------------------------------------------ persistence


def test_a_saved_subspace_scores_identically_when_loaded(fitted: Fitted) -> None:
    """Exactly identically. The state is a mean and a basis; nothing about a round trip
    through disk is allowed to be approximate."""
    store = fitted.root / "saved"
    store.mkdir(parents=True, exist_ok=True)
    fitted.model.save(store)
    assert (store / STATE_FILENAME).exists()
    assert (store / MANIFEST_FILENAME).exists()

    restored = SubspaceAdModel(_config())
    restored.load(store)
    _, infer_ctx = _contexts(fitted.root / "restored")
    again = restored.predict(fitted.probe, infer_ctx)
    assert [entry.score for entry in again] == [entry.score for entry in fitted.predictions]


def test_loading_into_a_different_window_refuses_by_name(fitted: Fitted) -> None:
    store = fitted.root / "saved-window"
    store.mkdir(parents=True, exist_ok=True)
    fitted.model.save(store)

    wrong = SubspaceAdModel(_config(layers=LayerWindow.LAST))
    with pytest.raises(RuntimeError, match="fitted on upper_half"):
        wrong.load(store)


def test_predicting_before_fitting_refuses(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path)
    model = SubspaceAdModel(_config())
    with pytest.raises(RuntimeError, match="before it was fitted"):
        model.predict([], infer_ctx)


# ------------------------------------------------------------------------ channels


def test_each_channel_gets_its_own_subspace(tmp_path: Path) -> None:
    """Two views of one part share no normal appearance. Pooling them would fit a subspace
    spanning both, against which neither is far from normal."""
    train = []
    for index in range(4):
        train.append(
            ImageRecord(
                image_id=index,
                sample_id=index,
                channel="bright",
                path=_normal(tmp_path / f"b{index}.png", index),
            )
        )
        train.append(
            ImageRecord(
                image_id=100 + index,
                sample_id=index,
                channel="dark",
                path=_save(tmp_path / f"d{index}.png", 255.0 - _field(index)),
            )
        )
    train_ctx, infer_ctx = _contexts(tmp_path / "run")
    model = SubspaceAdModel(_config())
    model.fit(train, train_ctx)
    assert set(model._fits) == {"bright", "dark"}

    stranger = ImageRecord(
        image_id=900,
        sample_id=900,
        channel="uv",
        path=_normal(tmp_path / "uv.png", 9),
    )
    with pytest.raises(RuntimeError, match="channel uv, which this model was not fitted on"):
        model.predict([stranger], infer_ctx)


# ------------------------------------------------------------------------ refusals


def test_a_part_whose_orientation_matters_can_switch_rotations_off(tmp_path: Path) -> None:
    """The paper excludes one MVTec category because a rotated copy of it is not normal.
    In a plugin that is a field, not a carve-out, and it has to actually fit."""
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(4)
    ]
    train_ctx, infer_ctx = _contexts(tmp_path / "run")
    model = SubspaceAdModel(_config(rotations=0, max_fit_images=4))
    model.fit(train, train_ctx)

    probe = [ImageRecord(image_id=9, sample_id=9, path=_defect(tmp_path / "d.png", 3))]
    assert model.predict(probe, infer_ctx)[0].score > 0.0


def test_masked_rotations_keep_only_the_real_patches(tmp_path: Path) -> None:
    """Under `masked` a rotated frame contributes fewer patches than under `zeros`, and the
    difference is the invented corners."""
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(2)
    ]
    counts = {}
    for fill in (RotationFill.ZEROS, RotationFill.MASKED):
        train_ctx, _ = _contexts(tmp_path / fill.value)
        model = SubspaceAdModel(_config(rotation_fill=fill, rotations=3, max_fit_images=2))
        model.fit(train, train_ctx)
        counts[fill] = model._fits[""].fit.sample_count
    assert counts[RotationFill.MASKED] < counts[RotationFill.ZEROS]


def test_a_prepared_size_that_does_not_divide_is_refused_at_the_boundary(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    ctx = TrainContext(
        artifact_dir=artifacts,
        cache_dir=tmp_path / "cache",
        preprocessing=PreprocessingConfig(width=100, height=100),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(artifacts / "diagnostics"),
    )
    record = ImageRecord(image_id=1, sample_id=1, path=_normal(tmp_path / "n.png", 1))
    with pytest.raises(ValueError, match="divisible by 14"):
        SubspaceAdModel(_config()).fit([record], ctx)


def test_an_empty_training_set_refuses_before_loading_an_encoder(tmp_path: Path) -> None:
    train_ctx, _ = _contexts(tmp_path)
    with pytest.raises(RuntimeError, match="no training images"):
        SubspaceAdModel(_config()).fit([], train_ctx)


def test_a_cancelled_fit_leaves_a_model_that_refuses_rather_than_answers(tmp_path: Path) -> None:
    """A partly-filled subspace is the worst available outcome: it scores, and every number
    it produces is wrong in a way nothing reports."""

    class Cancelling(NullReporter):
        def should_cancel(self) -> bool:
            return True

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    ctx = TrainContext(
        artifact_dir=artifacts,
        cache_dir=tmp_path / "cache",
        preprocessing=PreprocessingConfig(width=SIZE, height=SIZE),
        device=Device.CPU,
        reporter=Cancelling(),
        diagnostics=DiagnosticWriter(artifacts / "diagnostics"),
    )
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(2)
    ]
    model = SubspaceAdModel(_config())
    with pytest.raises(ModelCancelledError):
        model.fit(train, ctx)
    assert model._fits == {}
