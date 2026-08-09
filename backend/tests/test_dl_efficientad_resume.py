"""Continuing a finished run, and whether it is actually the same run (ADR-0025).

`dl`-gated: CI's main backend job installs without `--extra dl`, so this skips there. It
runs in the separate `backend-dl` job, and locally under `uv sync --extra dl`.

The centrepiece is `test_the_checkpoint_loses_nothing`. "Exact resume" is a claim, and a
claim that nothing measures is a claim that quietly stops being true — the whole reason
the checkpoint grew to three times its size is that weights alone are not enough to
continue from.

Note carefully **what the claim is**. Continuing through disk equals continuing in
process; it does *not* equal having trained that long in one go, because `max_steps` is a
per-run budget and the first run already spent its own learning-rate decay. That second
fact has a test of its own, so nobody later mistakes it for a bug and files the
difference away.

The penalty batch is pinned throughout. Its iterator position is **not** in the
checkpoint — stated in the ADR and on screen — so leaving it free would make this test
measure the one thing resume does not control, fail intermittently, and get deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    NullReporter,
    SupportsResume,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomaly_lab.models.efficientad_anomalib import (  # noqa: E402
    STATE_FILENAME,
    EfficientAdAnomalibModel,
    EfficientAdConfig,
)

SIZE = 256


@pytest.fixture
def images(tmp_path: Path) -> list[ImageRecord]:
    """Four synthetic gradients. Small in number, full size — EfficientAD's PDN is fixed."""
    generator = np.random.default_rng(0)
    records: list[ImageRecord] = []
    for index in range(4):
        array = np.clip(
            np.linspace(30, 220, SIZE * SIZE).reshape(SIZE, SIZE)
            + generator.normal(0, 4, size=(SIZE, SIZE)),
            0,
            255,
        ).astype(np.uint8)
        path = tmp_path / f"n{index}.png"
        Image.fromarray(array, mode="L").convert("RGB").save(path)
        records.append(ImageRecord(image_id=index, sample_id=index, channel=None, path=path))
    return records


def context(tmp_path: Path) -> TrainContext:
    return TrainContext(
        artifact_dir=tmp_path / "artifacts",
        cache_dir=tmp_path / "cache",
        preprocessing=PreprocessingConfig(width=SIZE, height=SIZE),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "artifacts" / "diagnostics", enabled=False),
        val=(),
    )


@pytest.fixture
def pinned_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
    """One fixed penalty batch, and no 1.5 GB download.

    The iterator's position is outside the checkpoint by decision, so pinning it is what
    isolates the thing resume actually controls.
    """
    import anomaly_lab.models.efficientad_anomalib as plugin

    batch = torch.zeros(1, 3, SIZE, SIZE)
    monkeypatch.setattr(plugin, "_next_penalty_batch", lambda module: batch.clone())
    monkeypatch.setattr(
        EfficientAdAnomalibModel, "_ensure_penalty_set", lambda self, module, ctx: None
    )


def build(steps: int) -> EfficientAdAnomalibModel:
    return EfficientAdAnomalibModel(
        EfficientAdConfig(model_size="small", max_steps=steps, seed=7, quantile_images=8)
    )


def weights(model: EfficientAdAnomalibModel) -> dict[str, Any]:
    return {key: value.clone() for key, value in model._module.model.student.state_dict().items()}


@pytest.mark.usefixtures("pinned_penalty")
def test_the_protocol_and_the_flag_agree() -> None:
    """A flag without the methods is a plugin bug the handler is entitled to trust."""
    assert EfficientAdAnomalibModel.capabilities().supports_resume is True
    assert isinstance(build(10), SupportsResume)


@pytest.mark.usefixtures("pinned_penalty")
def test_a_checkpoint_round_trips_the_state_a_continuation_needs(
    tmp_path: Path, images: list[ImageRecord]
) -> None:
    model = build(10)
    model.fit(images, context(tmp_path))
    directory = tmp_path / "model"
    directory.mkdir()
    model.save(directory)

    restored = build(10)
    restored.load(directory)

    assert restored.completed_steps() == 10
    assert restored._optimizer_state is not None
    assert restored._scheduler_state is not None
    assert restored._generator_state is not None
    assert restored._torch_rng_state is not None


@pytest.mark.usefixtures("pinned_penalty")
def test_a_format_one_checkpoint_is_refused_by_name(
    tmp_path: Path, images: list[ImageRecord]
) -> None:
    """Not degraded.

    A continuation that silently restarts Adam's moments is not a continuation, and the
    house rule is that a visible gap beats a fabricated number.
    """
    model = build(10)
    model.fit(images, context(tmp_path))
    directory = tmp_path / "model"
    directory.mkdir()
    # Exactly what M4.6 and earlier wrote.
    torch.save(
        {"state_dict": model._module.model.state_dict(), "model_size": "small"},
        directory / STATE_FILENAME,
    )

    restored = build(10)
    restored.load(directory)
    assert restored.completed_steps() == 0

    with pytest.raises(RuntimeError, match="before optimizer state was saved"):
        restored.fit_more(images, context(tmp_path), additional_steps=2)


@pytest.mark.usefixtures("pinned_penalty")
def test_the_checkpoint_loses_nothing(tmp_path: Path, images: list[ImageRecord]) -> None:
    """The claim resume actually makes, measured.

    Continuing through a save and a load must land in the same place as continuing without
    ever leaving the process. That isolates exactly what the checkpoint is responsible for
    — the optimizer moments, the schedule position, the step counter, both RNG streams —
    from everything it is not.

    It is the useful claim because it is the one that silently breaks. Drop the optimizer
    state and this diverges immediately; drop `initial_lr` and it raises; restore the RNG
    wrongly and the two runs walk different images.

    **Bit-equality, not a tolerance.** Measured: on one device the two paths agree exactly,
    so a tolerance here would hide a real divergence behind a number nobody chose.
    """
    base = build(10)
    base.fit(images, context(tmp_path / "base"))
    directory = tmp_path / "model"
    directory.mkdir()
    base.save(directory)

    # A: never went to disk.
    base.fit_more(images, context(tmp_path / "a"), additional_steps=10)

    # B: went to disk and came back.
    restored = build(10)
    restored.load(directory)
    restored.fit_more(images, context(tmp_path / "b"), additional_steps=10)

    assert restored.completed_steps() == base.completed_steps() == 20
    expected, actual = weights(base), weights(restored)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


@pytest.mark.usefixtures("pinned_penalty")
def test_the_continuation_actually_trains(tmp_path: Path, images: list[ImageRecord]) -> None:
    """A resume that loaded everything and then moved nothing would pass the test above."""
    base = build(10)
    base.fit(images, context(tmp_path / "base"))
    directory = tmp_path / "model"
    directory.mkdir()
    base.save(directory)
    before = weights(base)

    resumed = build(10)
    resumed.load(directory)
    resumed.fit_more(images, context(tmp_path / "more"), additional_steps=10)
    after = weights(resumed)

    assert any(not torch.equal(after[key], value) for key, value in before.items()), (
        "the weights did not move, so nothing was continued"
    )


@pytest.mark.usefixtures("pinned_penalty")
def test_the_first_run_already_spent_its_own_schedule(
    tmp_path: Path, images: list[ImageRecord]
) -> None:
    """Continuing is **not** the same as having trained that long in one go, and why.

    `max_steps` is a per-run budget frozen into the experiment (ADR-0025), so a run of 10
    sizes its own `StepLR` for 10 and completes its tenfold decay inside those steps. A
    later continuation resizes the schedule to the new total, which puts the learning rate
    back up. Both facts are deliberate and both are printed on screen; this pins them so
    that nobody later "fixes" the difference into silence.
    """
    one_pass = build(20)
    one_pass.fit(images, context(tmp_path / "long"))

    split = build(10)
    split.fit(images, context(tmp_path / "first"))
    directory = tmp_path / "model"
    directory.mkdir()
    split.save(directory)
    resumed = build(10)
    resumed.load(directory)
    resumed.fit_more(images, context(tmp_path / "second"), additional_steps=10)

    assert resumed.completed_steps() == one_pass.completed_steps() == 20
    differs = any(
        not torch.allclose(actual, expected, rtol=1e-3, atol=1e-5)
        for actual, expected in zip(
            weights(resumed).values(), weights(one_pass).values(), strict=True
        )
    )
    assert differs, (
        "10 + 10 landed exactly on 20, which means the per-run schedule stopped being "
        "per-run — the surprising behaviour the UI explains would then be a lie"
    )


@pytest.mark.usefixtures("pinned_penalty")
def test_a_continued_run_reports_absolute_steps(tmp_path: Path, images: list[ImageRecord]) -> None:
    """So the chart is one curve, with no stitching and no extra log reads (ADR-0020)."""

    class Recorder(NullReporter):
        def __init__(self) -> None:
            self.steps: list[int] = []

        def metric(self, name: str, value: float, step: int | None = None) -> None:
            if name == "loss_total" and step is not None:
                self.steps.append(step)

    model = build(10)
    model.fit(images, context(tmp_path / "first"))
    directory = tmp_path / "model"
    directory.mkdir()
    model.save(directory)

    resumed = build(10)
    resumed.load(directory)
    recorder = Recorder()
    ctx = context(tmp_path / "second")
    resumed.fit_more(
        images,
        TrainContext(
            artifact_dir=ctx.artifact_dir,
            cache_dir=ctx.cache_dir,
            preprocessing=ctx.preprocessing,
            device=ctx.device,
            reporter=recorder,
            diagnostics=ctx.diagnostics,
            val=(),
        ),
        additional_steps=10,
    )

    # Continues from 10, rather than restarting at 0.
    assert recorder.steps
    assert min(recorder.steps) >= 10
    assert max(recorder.steps) == 19
