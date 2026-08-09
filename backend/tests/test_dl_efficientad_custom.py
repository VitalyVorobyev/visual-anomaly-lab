"""`efficientad_custom` as a plugin: does it detect, does it refuse, does it resume.

**Gated on torch alone, never on anomalib.** That is not an oversight — this method
deliberately does not depend on anomalib, and a test file that imported it would stop
measuring the difference. The comparison against the reference lives in
`test_dl_efficientad_equivalence.py`, which is a different question.

Nothing here downloads anything. The teacher is the seeded random one (`pretrained_teacher`
off) and the penalty set is eight synthetic images written into `tmp_path`, so the real
asset-loading path runs against a real directory without fetching 1.5 GB.

The centrepiece is `test_it_detects_a_stamped_defect`. Until it existed, **nothing in this
suite asserted that any EfficientAD detects anything** — the two `dl`-gated files that
predate it cover checkpoint exactness and module introspection, and neither calls `predict`
or computes a metric. A method that trains, writes maps, saves and reloads, and separates
nothing at all would have passed everything.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from anomaly_lab.eval.metrics import roc_auc  # noqa: E402
from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    Prediction,
    SupportsResume,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter  # noqa: E402
from anomaly_lab.models.efficientad_assets import PENALTY_SUBDIR  # noqa: E402
from anomaly_lab.models.efficientad_custom import (  # noqa: E402
    EfficientAdCustomConfig,
    EfficientAdCustomModel,
)
from anomaly_lab.models.preprocessing import ColorMode, PreprocessingConfig  # noqa: E402

SIZE = 256
"""EfficientAD's own floor — the autoencoder cannot encode anything smaller."""

TRAIN_IMAGES = 8
STEPS = 100
"""Measured: a random-init teacher separates this defect at ROC-AUC 1.0 from 50 steps, with
a score margin around 20x. 100 is double the point where the margin is already large, and
costs about 30 s on a CI runner — headroom bought where it is cheap, rather than a lowered
bar later."""


def _gradient(seed: int) -> np.ndarray:
    """The same smooth scene every time, with a little noise — a stand-in for a part."""
    generator = np.random.default_rng(seed)
    base = np.linspace(40, 200, SIZE * SIZE).reshape(SIZE, SIZE)
    return np.clip(base + generator.normal(0, 3, size=(SIZE, SIZE)), 0, 255)


def _write(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB").save(path)
    return path


def _normal(path: Path, seed: int) -> Path:
    return _write(path, _gradient(seed))


def _defective(path: Path, seed: int) -> Path:
    """A saturated square — the same anomaly shape the rest of the suite uses."""
    array = _gradient(seed)
    array[96:160, 96:160] = 255
    return _write(path, array)


def _penalty_set(root: Path) -> None:
    """Eight synthetic 'natural images', so the real penalty path runs without ImageNette."""
    generator = np.random.default_rng(4242)
    for index in range(8):
        noise = generator.integers(0, 255, size=(SIZE, SIZE, 3), dtype=np.uint8)
        path = root / PENALTY_SUBDIR / f"noise{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(noise).save(path)


def _contexts(
    tmp_path: Path, *, color: ColorMode = ColorMode.RGB, width: int = SIZE, height: int = SIZE
) -> tuple[TrainContext, InferContext]:
    config = PreprocessingConfig(width=width, height=height, color=color)
    cache = tmp_path / "cache"
    _penalty_set(cache)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    return (
        TrainContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=config,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(tmp_path / "d-train"),
        ),
        InferContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=config,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(tmp_path / "d-infer"),
        ),
    )


def _config(**overrides: object) -> EfficientAdCustomConfig:
    """A run small enough for a test, with no download in it."""
    base: dict[str, object] = {
        "max_steps": STEPS,
        "seed": 7,
        "pretrained_teacher": False,
        "allow_downloads": False,
        "quantile_images": 8,
        "quantile_pixel_budget": 1 << 16,
        "stats_batch_size": 4,
    }
    base.update(overrides)
    return EfficientAdCustomConfig(**base)  # type: ignore[arg-type]


def _training_records(tmp_path: Path) -> list[ImageRecord]:
    return [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(TRAIN_IMAGES)
    ]


def _probe_records(tmp_path: Path) -> tuple[list[ImageRecord], np.ndarray]:
    """Four unseen normals and four defects, with the labels the metric needs."""
    records = [
        ImageRecord(
            image_id=100 + i, sample_id=100 + i, path=_normal(tmp_path / f"tn{i}.png", 500 + i)
        )
        for i in range(4)
    ] + [
        ImageRecord(
            image_id=200 + i, sample_id=200 + i, path=_defective(tmp_path / f"td{i}.png", 900 + i)
        )
        for i in range(4)
    ]
    return records, np.array([False] * 4 + [True] * 4)


@dataclass
class Trained:
    """One trained run and everything the detection assertions read off it.

    A dataclass rather than a dict, because these tests reach into it from six places and
    a `dict[str, object]` would need a cast at every one of them — which is a lot of noise
    around the assertions that matter.
    """

    model: EfficientAdCustomModel
    predictions: list[Prediction]
    labels: np.ndarray
    probe: list[ImageRecord]
    infer_ctx: InferContext
    untrained_error: float


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Trained:
    """One trained run, shared by the detection assertions — training is the expensive part."""
    tmp_path = tmp_path_factory.mktemp("efficientad-detect")
    train_ctx, infer_ctx = _contexts(tmp_path)
    train = _training_records(tmp_path)
    probe, labels = _probe_records(tmp_path)

    model = EfficientAdCustomModel(_config())
    model.fit(train, train_ctx)
    predictions = model.predict(probe, infer_ctx)

    return Trained(
        model=model,
        predictions=predictions,
        labels=labels,
        probe=probe,
        infer_ctx=infer_ctx,
        untrained_error=_mean_student_error(model, train_ctx, probe[:4]),
    )


def _branch_means(
    model: EfficientAdCustomModel, ctx: InferContext, records: list[ImageRecord], branch: int
) -> list[float]:
    with torch.no_grad():
        return [
            float(model._net.maps(model._image(record, ctx, torch), normalize=False)[branch].mean())
            for record in records
        ]


def _branch_peaks(
    model: EfficientAdCustomModel, ctx: InferContext, records: list[ImageRecord], branch: int
) -> list[float]:
    with torch.no_grad():
        return [
            float(model._net.maps(model._image(record, ctx, torch), normalize=False)[branch].max())
            for record in records
        ]


def _mean_student_error(
    model: EfficientAdCustomModel, ctx: TrainContext, records: list[ImageRecord]
) -> float:
    """What the *untrained* student would have scored, under the trained run's conditions.

    Two things have to be held equal for this to mean anything, and getting either wrong
    turns the comparison into noise. The net is rebuilt through the model's own `_build`,
    which seeds torch before constructing, so the teacher and the student's initialisation
    are bit-identical to the run's. And the fitted **teacher statistics are copied across**:
    they rescale the teacher's output to roughly unit variance, so an unfitted net produces
    distances two orders of magnitude smaller for reasons that have nothing to do with
    training. Comparing against that inverted this test the first time it was written.
    """
    net = model._build(ctx)
    net.set_teacher_statistics(model._net.teacher_mean, model._net.teacher_std)
    net.eval()
    with torch.no_grad():
        errors = [
            float(net.maps(model._image(record, ctx, torch), normalize=False)[0].mean())
            for record in records
        ]
    return float(np.mean(errors))


# ------------------------------------------------------------------ the detection bar


def test_it_detects_a_stamped_defect(trained: Trained) -> None:
    """The test this method exists to pass.

    A hard bar on a deliberately easy task, rather than a soft bar on a marginal one: the
    measured score margin is roughly 20x, so this either passes comfortably or something is
    genuinely broken. A `> 0.8` bar on a harder fixture would flake and teach us to lower it.
    """
    scores = np.array([prediction.score for prediction in trained.predictions])
    labels = trained.labels

    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert float(np.median(scores[labels])) > 5 * float(np.median(scores[~labels]))
    assert bool(np.isfinite(scores).all())


def test_each_branch_separates_on_its_own(trained: Trained) -> None:
    """Both branches must work, and the combined score can hide that only one does.

    A wrongly wired autoencoder branch still produces a combined map dominated by a healthy
    student-teacher branch, and every score-level assertion passes. This is the check that
    fails instead — and it reads the *unnormalized* maps, so it is independent of the
    quantile fit, which has its own tests.
    """
    normals = [record for record, bad in zip(trained.probe, trained.labels, strict=True) if not bad]
    defects = [record for record, bad in zip(trained.probe, trained.labels, strict=True) if bad]

    for branch, name in ((0, "student-teacher"), (1, "autoencoder")):
        clean = _branch_peaks(trained.model, trained.infer_ctx, normals, branch)
        broken = _branch_peaks(trained.model, trained.infer_ctx, defects, branch)
        assert min(broken) > max(clean), f"the {name} branch does not separate on its own"


def test_the_student_actually_learned(trained: Trained) -> None:
    """Without this, the whole detection test passes on a training loop that never steps.

    The defect is extreme enough that an *untrained* student already fails on it, so
    separation alone proves nothing about the optimizer. What proves it is the error on
    held-out **normals** falling: that is what training is supposed to reduce, and nothing
    else in the run reduces it.
    """
    normals = [record for record, bad in zip(trained.probe, trained.labels, strict=True) if not bad]
    after = float(np.mean(_branch_means(trained.model, trained.infer_ctx, normals, 0)))
    before = trained.untrained_error
    assert after < before / 3.0, f"student-teacher error on normals went {before} -> {after}"


def test_every_prediction_has_a_two_dimensional_map(trained: Trained) -> None:
    for prediction in trained.predictions:
        assert prediction.anomaly_map is not None
        stored = np.load(prediction.anomaly_map)
        assert stored.ndim == 2
        assert stored.dtype == np.float32


# ------------------------------------------------------------------ plugin contract


def test_predictions_come_back_one_per_input_in_order(trained: Trained) -> None:
    """One `Prediction` per input, in the input's order — the interface's central promise.

    The infer handler checks the count and then trusts the order, because nothing in a
    `Prediction` identifies which input it came from except its position and its id.
    """
    assert [prediction.image_id for prediction in trained.predictions] == [
        record.image_id for record in trained.probe
    ]


def test_one_seed_is_one_experiment(tmp_path: Path) -> None:
    """Two runs of one configuration must produce the same scores, exactly.

    This is the foundation the whole milestone rests on: `efficientad_custom` improves on
    evidence, and evidence means "this change moved the number". If a rerun of the *same*
    configuration moves the number, nothing else measured here means anything.

    It did not hold when this was written. Weight initialisation draws from torch's global
    stream, which our own generators do not cover, so `seed` controlled the training order
    and the augmentations over *different* initial weights — and with `pretrained_teacher`
    off, over a different teacher as well. Found by a detection assertion that compared a
    trained student against an untrained one and got an answer that made no sense.
    """
    scores = []
    for run in ("first", "second"):
        root = tmp_path / run
        train_ctx, infer_ctx = _contexts(root)
        model = EfficientAdCustomModel(_config(max_steps=10))
        model.fit(_training_records(root), train_ctx)
        probe, _ = _probe_records(root)
        scores.append([prediction.score for prediction in model.predict(probe, infer_ctx)])

    assert scores[0] == scores[1]


def test_predicting_before_fitting_is_an_error_not_a_zero(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path)
    model = EfficientAdCustomModel(_config())
    with pytest.raises(RuntimeError, match="before it was fitted"):
        model.predict(_training_records(tmp_path), infer_ctx)


def test_the_capability_flag_and_the_protocol_agree() -> None:
    """The train handler checks these against each other rather than trusting the flag."""
    assert EfficientAdCustomModel.capabilities().supports_resume is True
    assert isinstance(EfficientAdCustomModel(_config()), SupportsResume)


def test_availability_names_torch_rather_than_anomalib() -> None:
    """The dependency difference is the whole reason this method is a separate one."""
    reason = EfficientAdCustomModel.availability().reason
    assert reason is None or "dl" in reason


# ------------------------------------------------------------------ guards


@pytest.mark.parametrize("size", [64, 128, 192])
def test_an_input_below_the_encoder_floor_is_refused_by_name(tmp_path: Path, size: int) -> None:
    """The wrapper fails inside conv2d with a message about a padded input size.

    That message is true and useless: it names neither the setting to change nor the
    component that imposes the limit. This one names both.
    """
    train_ctx, _ = _contexts(tmp_path, width=size, height=size)
    model = EfficientAdCustomModel(_config())
    with pytest.raises(ValueError, match="at least 256x256"):
        model.fit(_training_records(tmp_path), train_ctx)


def test_an_input_that_is_not_a_multiple_of_64_is_refused_by_name(tmp_path: Path) -> None:
    """320x288 is fine; 300x256 lands the reconstruction on a different grid."""
    train_ctx, _ = _contexts(tmp_path, width=300, height=256)
    model = EfficientAdCustomModel(_config())
    with pytest.raises(ValueError, match="multiple of 64"):
        model.fit(_training_records(tmp_path), train_ctx)


def test_a_grayscale_experiment_scores_the_same_as_an_rgb_one(tmp_path: Path) -> None:
    """Grayscale is supported, and this is what makes that a claim rather than an accident.

    The plan for this milestone said to refuse grayscale, on the grounds that ImageNet
    normalization needs three channels. Measured, it does not: a `(N, 1, H, W)` batch minus
    a `(1, 3, 1, 1)` mean broadcasts to three channels on its own. Refusing it would have
    refused a working configuration, so the expansion is written out instead — and pinned
    here, end to end, where a future change to the input path would break it.
    """
    scores = {}
    for color in (ColorMode.RGB, ColorMode.GRAYSCALE):
        root = tmp_path / color.value
        train_ctx, infer_ctx = _contexts(root, color=color)
        model = EfficientAdCustomModel(_config(max_steps=10))
        model.fit(_training_records(root), train_ctx)
        probe, _ = _probe_records(root)
        scores[color] = [p.score for p in model.predict(probe, infer_ctx)]

    assert scores[ColorMode.RGB] == pytest.approx(scores[ColorMode.GRAYSCALE], rel=1e-5)


# ------------------------------------------------------------------ diagnostics contract


def test_the_diagnostic_keys_are_the_ones_the_views_expect(tmp_path: Path) -> None:
    """ADR-0018 names key agreement as a coordination cost M6 has to pay. This pays it.

    The keys are compared against `efficientad_anomalib`'s, read out of its source rather
    than restated here — restating them would make this test agree with the plan instead of
    with the thing the M4 views were written against.
    """
    train_ctx, infer_ctx = _contexts(tmp_path)
    model = EfficientAdCustomModel(_config(max_steps=10))
    model.fit(_training_records(tmp_path), train_ctx)
    model.predict(_training_records(tmp_path)[:1], infer_ctx)

    emitted = {entry.key for entry in train_ctx.diagnostics.flush().entries}
    emitted |= {entry.key for entry in infer_ctx.diagnostics.flush().entries}

    wrapper = Path("src/anomaly_lab/models/efficientad_anomalib.py").read_text(encoding="utf-8")
    expected = {
        key
        for key in (
            "architecture",
            "teacher_features_pca",
            "teacher_features_grid",
            "teacher_magnitude",
            "score_normalization",
            "map_student_teacher",
            "map_autoencoder",
        )
        if f'"{key}"' in wrapper
    }
    assert expected, "the wrapper's diagnostic keys could not be read; this test is not testing"
    assert expected <= emitted


# ------------------------------------------------------------------ resume


@dataclass
class Saved:
    """A fitted model already written to disk, and what it takes to continue it."""

    artifacts: Path
    train_ctx: TrainContext
    records: list[ImageRecord]
    completed: int
    state: dict[str, Any]


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Saved:
    """One short run, fitted and saved once for every test that only needs to *load* it.

    These five tests are about persistence, not about fitting, and each fitting its own
    model cost about a minute of CI apiece for no coverage — the `dl` job ran twenty-one
    minutes before this fixture existed. Every test still constructs its own model and
    loads it from disk, so none of them shares mutable model state.
    """
    root = tmp_path_factory.mktemp("efficientad-saved")
    train_ctx, _ = _contexts(root)
    records = _training_records(root)
    model = EfficientAdCustomModel(_config(max_steps=10))
    model.fit(records, train_ctx)
    model.save(root / "artifacts")
    return Saved(
        artifacts=root / "artifacts",
        train_ctx=train_ctx,
        records=records,
        completed=model.completed_steps(),
        state={key: value.clone() for key, value in model._net.state_dict().items()},
    )


def test_a_checkpoint_loses_nothing(saved: Saved) -> None:
    """Bit-equality, not tolerance. A checkpoint that is nearly right is not right."""
    restored = EfficientAdCustomModel(_config(max_steps=10))
    restored.load(saved.artifacts)

    assert restored.completed_steps() == saved.completed == 10
    actual = restored._net.state_dict()
    assert set(actual) == set(saved.state)
    for key, value in saved.state.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


def test_a_checkpoint_carries_what_a_continuation_needs(saved: Saved) -> None:
    """Weights alone are not a continuation: Adam's moments are most of what a run learned.

    The penalty state is the one the wrapper does not carry — ADR-0025 records its penalty
    sequence restarting on every resume as an accepted cost.
    """
    restored = EfficientAdCustomModel(_config(max_steps=10))
    restored.load(saved.artifacts)
    assert restored._optimizer_state is not None
    assert restored._scheduler_state is not None
    assert restored._generator_state is not None
    assert restored._torch_rng_state is not None
    assert restored._penalty_state is not None


def test_the_continuation_actually_trains(saved: Saved) -> None:
    restored = EfficientAdCustomModel(_config(max_steps=10))
    restored.load(saved.artifacts)
    restored.fit_more(saved.records, saved.train_ctx, additional_steps=5)

    assert restored.completed_steps() == 15
    after = restored._net.state_dict()
    assert any(not torch.equal(after[key], value) for key, value in saved.state.items()), (
        "the weights did not move, so nothing was continued"
    )


def test_a_continuation_does_not_inherit_the_previous_leg_s_decayed_rate() -> None:
    """The bug that made a step-budget curve measure the wrong thing.

    `StepLR.get_lr` multiplies the param group's *current* rate, and
    `Adam.load_state_dict` restores the rate the previous leg ended on — which is always
    the decayed one, because every leg anneals over its own last 5%. So a naive resume
    starts a tenth low and drops again at the next boundary: 1e-5 on the first
    continuation, 1e-9 by the fifth, with nothing on screen to say so.

    Checked on the scheduler alone rather than through `fit_more`, because the arithmetic
    is the whole claim and a training run would cost a minute to say the same thing.
    """
    model = EfficientAdCustomModel(_config(max_steps=4000))
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)

    scheduler = model._build_scheduler(optimizer, 4000, torch)
    for _ in range(4000):
        optimizer.step()  # before the scheduler, which is the order the loop uses
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5), "leg one should anneal"

    # Exactly what `fit_more` does: a fresh optimizer, the previous leg's state on top of
    # it, then the schedule rebuilt against the new total.
    resumed = torch.optim.Adam([parameter], lr=1e-4)
    resumed.load_state_dict(optimizer.state_dict())
    model._build_scheduler(resumed, 8000, torch, last_epoch=3999)
    assert resumed.param_groups[0]["lr"] == pytest.approx(1e-4), (
        "the continuation inherited the previous leg's decayed rate instead of the one "
        "its own schedule calls for"
    )


def test_a_continuation_past_the_new_drop_point_stays_decayed() -> None:
    """The other half of the closed form, which a base-rate reset alone would get wrong.

    Resuming at step 8000 into a total of 8100 puts the drop at 7695 — already behind us.
    The rate belongs at 1e-5 there, and a fix that simply restored the base rate on every
    resume would hand back 1e-4 and quietly undo the anneal.
    """
    model = EfficientAdCustomModel(_config(max_steps=8100))
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)

    model._build_scheduler(optimizer, 8100, torch, last_epoch=7999)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


@pytest.mark.parametrize("size", ["small", "medium"])
def test_a_sequential_teacher_checkpoint_loads_in_the_published_order(
    size: Literal["small", "medium"],
) -> None:
    """nelson1425 keys its teacher by position in an `nn.Sequential`; we key by name.

    The remap is by order of appearance, and this is the test that says so: a file keyed
    `0`, `3`, `6`, `8` — the gaps being ReLUs and pools — has to land on `conv1` … `conv4`
    *in that order*. Getting it wrong produces a network that loads without complaint and
    describes patches with the layers shuffled, which no downstream assertion would catch.
    """
    from anomaly_lab.models.efficientad_nets import PatchDescriptionNetwork, load_pdn_weights

    source = PatchDescriptionNetwork(out_channels=384, size=size)
    named = source.state_dict()
    convolutions = source.convolutions

    # The reference's layout: sparse integer indices, ascending, with gaps.
    sequential = {}
    for index in range(1, convolutions + 1):
        position = 3 * (index - 1)
        sequential[f"{position}.weight"] = named[f"conv{index}.weight"]
        sequential[f"{position}.bias"] = named[f"conv{index}.bias"]

    target = PatchDescriptionNetwork(out_channels=384, size=size)
    load_pdn_weights(target, sequential)

    for key, tensor in named.items():
        assert torch.equal(target.state_dict()[key], tensor), key

    probe = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        assert torch.equal(target(probe), source(probe))


def _reference_pdn(out_channels: int, size: str, padding: bool) -> Any:
    """nelson1425/EfficientAD's `get_pdn_small`/`get_pdn_medium`, transcribed verbatim.

    Copied structure-for-structure from that repository's `common.py` rather than generated
    from our own table, because the whole point is to be a second opinion about what the
    network *is*.
    """
    from torch import nn

    pad_mult = 1 if padding else 0
    if size == "small":
        return nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
            nn.Conv2d(128, 256, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
            nn.Conv2d(256, 256, kernel_size=3, padding=1 * pad_mult),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=4),
        )
    return nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(256, 512, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(512, 512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
    )


@pytest.mark.parametrize("size", ["small", "medium"])
@pytest.mark.parametrize("padding", [False, True])
def test_our_pdn_is_the_reference_pdn(size: Literal["small", "medium"], padding: bool) -> None:
    """Foreign weights may only be loaded into the network they were trained in.

    Shapes agreeing is not that claim: two networks can share every parameter shape and
    differ in where the ReLUs and the pools sit, and the mistake would be invisible — the
    weights load, the run finishes, and the teacher is quietly a different function. This
    builds the reference's `nn.Sequential` from its own source, puts one set of weights in
    both, and compares outputs.

    **The one real difference is preprocessing, and it is why the reference gets a
    pre-normalized input here.** Our PDN standardizes with ImageNet statistics inside its
    `forward`; the reference does it in the dataset transform. Same function, different
    seam — and worth pinning, because a teacher fed unnormalized pixels would also load
    without complaint.
    """
    from anomaly_lab.models.efficientad_nets import (
        PatchDescriptionNetwork,
        imagenet_normalize,
        load_pdn_weights,
    )

    ours = PatchDescriptionNetwork(out_channels=384, size=size, padding=padding).eval()
    theirs = _reference_pdn(384, size, padding).eval()

    # The reference keys by position in its Sequential; ours by name. This is the mapping
    # `load_pdn_weights` performs, built here from the reference's own module list.
    sequential = {}
    convolutions = [i for i, m in enumerate(theirs) if isinstance(m, torch.nn.Conv2d)]
    for index, position in enumerate(convolutions, start=1):
        sequential[f"{position}.weight"] = ours.state_dict()[f"conv{index}.weight"]
        sequential[f"{position}.bias"] = ours.state_dict()[f"conv{index}.bias"]
    theirs.load_state_dict({str(k): v for k, v in sequential.items()})

    pixels = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        assert torch.allclose(ours(pixels), theirs(imagenet_normalize(pixels)), atol=1e-6)

    # And the remap under test agrees with the mapping built from the reference's own layout.
    target = PatchDescriptionNetwork(out_channels=384, size=size, padding=padding).eval()
    load_pdn_weights(target, sequential)
    with torch.no_grad():
        assert torch.equal(target(pixels), ours(pixels))


def test_a_named_teacher_checkpoint_still_loads_unchanged() -> None:
    """anomalib's layout must keep working — it is the default and the shared baseline."""
    from anomaly_lab.models.efficientad_nets import PatchDescriptionNetwork, load_pdn_weights

    source = PatchDescriptionNetwork(out_channels=384, size="small")
    target = PatchDescriptionNetwork(out_channels=384, size="small")
    load_pdn_weights(target, source.state_dict())

    probe = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        assert torch.equal(target(probe), source(probe))


def test_a_teacher_checkpoint_for_the_other_size_is_refused_by_name() -> None:
    """A medium file in a small network must fail here, not deep inside a convolution."""
    from anomaly_lab.models.efficientad_nets import PatchDescriptionNetwork, load_pdn_weights

    medium = PatchDescriptionNetwork(out_channels=384, size="medium")
    sequential = {}
    for index in range(1, medium.convolutions + 1):
        position = 3 * (index - 1)
        sequential[f"{position}.weight"] = medium.state_dict()[f"conv{index}.weight"]
        sequential[f"{position}.bias"] = medium.state_dict()[f"conv{index}.bias"]

    small = PatchDescriptionNetwork(out_channels=384, size="small")
    with pytest.raises(ValueError, match="not a checkpoint for this model size"):
        load_pdn_weights(small, sequential)


def test_a_continuation_at_a_different_input_size_is_refused(saved: Saved, tmp_path: Path) -> None:
    """The normalization was fitted at one resolution and does not transfer to another.

    Continuing anyway would produce maps that are wrong in a way nothing downstream can
    see — the run would look finished and its scores would be quietly meaningless.
    """
    restored = EfficientAdCustomModel(_config(max_steps=10))
    restored.load(saved.artifacts)
    bigger, _ = _contexts(tmp_path / "bigger", width=320, height=320)
    with pytest.raises(RuntimeError, match="does not transfer"):
        restored.fit_more(saved.records, bigger, additional_steps=5)


def test_a_continuation_against_a_different_teacher_is_refused(
    saved: Saved, tmp_path: Path
) -> None:
    """A student cannot change teacher mid-training, and the failure would be silent.

    `fit_more` reloads the teacher from *configuration* and does not refit the teacher
    channel statistics — those were fitted in `fit`, against whichever teacher was loaded
    then. Continuing under a different one gives a student that spent every previous step
    imitating another network, new teacher weights, and the old teacher's normalization.
    It trains, the loss looks plausible, and every number afterwards is wrong.

    Exercised through the random teacher, whose identity carries its seed, so this needs no
    download and covers the same guard the two published sources go through.
    """
    restored = EfficientAdCustomModel(_config(max_steps=10, seed=8))
    restored.load(saved.artifacts)
    train_ctx, _ = _contexts(tmp_path / "other-teacher")
    with pytest.raises(RuntimeError, match="cannot change teacher"):
        restored.fit_more(saved.records, train_ctx, additional_steps=5)


def test_the_teacher_a_student_was_distilled_against_is_recorded(saved: Saved) -> None:
    """The guard above can only work if the checkpoint says what the teacher was."""
    stored = torch.load(
        saved.artifacts / "efficientad_custom.pt", map_location="cpu", weights_only=False
    )
    assert stored["teacher"] == "random:7"


def test_an_unknown_checkpoint_format_is_refused_by_name(saved: Saved, tmp_path: Path) -> None:
    """A future checkpoint may hold the same keys with different meanings.

    Copied out of the shared fixture before being edited, so tampering with a checkpoint
    cannot break the tests that expect to read a good one.
    """
    artifacts = tmp_path / "tampered"
    artifacts.mkdir()
    path = artifacts / "efficientad_custom.pt"
    shutil.copy(saved.artifacts / "efficientad_custom.pt", path)

    stored = torch.load(path, map_location="cpu", weights_only=False)
    stored["format"] = 99
    torch.save(stored, path)

    with pytest.raises(RuntimeError, match="format 99"):
        EfficientAdCustomModel(_config()).load(artifacts)


# ------------------------------------------------------------------ independence


def test_it_trains_with_anomalib_unimportable() -> None:
    """The independence claim, measured rather than asserted.

    `sys.modules[name] = None` makes `import name` raise, so this is the cheapest honest
    way to prove the method never reaches for anomalib — in a subprocess, because poisoning
    `sys.modules` in the test process would leak into every other test.
    """
    script = """
import sys
sys.modules["anomalib"] = None
from anomaly_lab.models.efficientad_custom import EfficientAdCustomModel
from anomaly_lab.models.efficientad_nets import EfficientAdNet
net = EfficientAdNet(size="small")
assert sum(p.numel() for p in net.parameters()) > 0
print("ok")
"""
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr
    assert "ok" in finished.stdout
