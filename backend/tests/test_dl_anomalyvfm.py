"""AnomalyVFM's torch half: the scoring loop, and — when the checkpoint is at hand — the network.

CI has no 1.42 GB checkpoint, so the scoring loop runs against a small stand-in with the
network's contract (`(score (B, 1), map (B, 1, H, W))` from pixels in `[0, 1]`). The real
network is exercised only when `ANOMALY_LAB_ANOMALYVFM_CACHE` names a model cache that holds
the pinned file (the directory whose `huggingface/hub` the app uses); it runs on the CPU at
a 64-pixel frame.

The method has no random stream: construction draws initial weights that the strict load
overwrites, and the forward pass has no dropout. So reproducibility is asserted one way —
the same image scores identically across two separately built networks — and there is no
seed whose change could move an answer.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomaly_lab.map_files import read_map  # noqa: E402
from anomaly_lab.models import anomalyvfm_anomalib as vfm  # noqa: E402
from anomaly_lab.models.anomalyvfm_anomalib import (  # noqa: E402
    AnomalyVfmAnomalibModel,
    AnomalyVfmConfig,
    build_network,
    resolve_weights,
)
from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter  # noqa: E402
from anomaly_lab.models.preprocessing import ColorMode, PreprocessingConfig  # noqa: E402

SIZE = 64
TorchModule: Any = torch.nn.Module
REAL_CACHE = os.environ.get("ANOMALY_LAB_ANOMALYVFM_CACHE")


class StandIn(TorchModule):  # type: ignore[misc]
    """The network's contract, with a map that is the image's own brightness."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[int, ...]] = []

    def forward(self, image: Any) -> tuple[Any, Any]:
        self.seen.append(tuple(image.shape))
        brightness = image.mean(dim=1, keepdim=True)
        return brightness.mean(dim=(2, 3)), brightness


class Cancelling(NullReporter):
    def __init__(self, after: int) -> None:
        self.after = after
        self.polls = 0

    def should_cancel(self) -> bool:
        self.polls += 1
        return self.polls > self.after


def _images(tmp_path: Path, count: int, *, mode: str = "RGB") -> list[ImageRecord]:
    records = []
    rng = np.random.default_rng(3)
    for image_id in range(1, count + 1):
        pixels = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
        path = tmp_path / f"{image_id}.png"
        Image.fromarray(pixels).convert(mode).save(path)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
    return records


def _ctx(tmp_path: Path, *, color: ColorMode = ColorMode.RGB, reporter: Any = None) -> InferContext:
    return InferContext(
        artifact_dir=tmp_path / "run",
        cache_dir=tmp_path / "cache",
        preprocessing=PreprocessingConfig(width=SIZE, height=SIZE, color=color),
        device=Device.CPU,
        reporter=reporter or NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    )


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> StandIn:
    network = StandIn()
    monkeypatch.setattr(vfm, "resolve_weights", lambda *_, **__: Path("verified"))
    monkeypatch.setattr(vfm, "build_network", lambda weights, device: network)
    return network


def test_an_unfitted_model_scores_every_image_with_a_frame_sized_map(
    tmp_path: Path, stand_in: StandIn
) -> None:
    records = _images(tmp_path, 3)
    ctx = _ctx(tmp_path)
    predictions = AnomalyVfmAnomalibModel(AnomalyVfmConfig()).predict(records, ctx)

    assert [prediction.image_id for prediction in predictions] == [1, 2, 3]
    for record, prediction in zip(records, predictions, strict=True):
        assert prediction.anomaly_map is not None
        stored = np.asarray(read_map(prediction.anomaly_map))
        assert stored.shape == (SIZE, SIZE)
        # Pixels arrive in [0, 1], unstandardised: RADIO's conditioner is the network's own.
        expected = np.asarray(Image.open(record.path), dtype=np.float32).mean(axis=2) / 255.0
        np.testing.assert_allclose(stored, expected, atol=1e-6)
        assert prediction.score == pytest.approx(float(expected.mean()), abs=1e-6)


def test_a_grey_frame_is_expanded_to_the_three_planes_the_network_reads(
    tmp_path: Path, stand_in: StandIn
) -> None:
    records = _images(tmp_path, 1, mode="L")
    AnomalyVfmAnomalibModel(AnomalyVfmConfig()).predict(
        records, _ctx(tmp_path, color=ColorMode.GRAYSCALE)
    )
    assert stand_in.seen == [(1, 3, SIZE, SIZE)]


def test_scoring_stops_at_the_next_image_when_cancelled(tmp_path: Path, stand_in: StandIn) -> None:
    records = _images(tmp_path, 3)
    with pytest.raises(ModelCancelledError):
        AnomalyVfmAnomalibModel(AnomalyVfmConfig()).predict(
            records, _ctx(tmp_path, reporter=Cancelling(after=1))
        )
    assert len(stand_in.seen) == 1


def test_a_recorded_frame_refuses_another(tmp_path: Path, stand_in: StandIn) -> None:
    model = AnomalyVfmAnomalibModel(AnomalyVfmConfig())
    model._record = {"width": 768, "height": 768}
    with pytest.raises(ValueError, match="768x768"):
        model.predict(_images(tmp_path, 1), _ctx(tmp_path))


@pytest.mark.skipif(
    REAL_CACHE is None,
    reason="ANOMALY_LAB_ANOMALYVFM_CACHE does not name a model cache holding the checkpoint",
)
def test_the_published_network_builds_offline_and_answers_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert REAL_CACHE is not None
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    weights = resolve_weights(Path(REAL_CACHE), allow_downloads=False, log=print)
    vit: Any = importlib.import_module("timm.models.vision_transformer")
    binding = vit.resample_abs_pos_embed
    image = torch.from_numpy(np.random.default_rng(5).random((1, 3, SIZE, SIZE), np.float32))

    answers = []
    for _ in range(2):
        stream = torch.get_rng_state()
        network = build_network(weights, "cpu")
        assert torch.equal(torch.get_rng_state(), stream), "construction moved torch's stream"
        with torch.inference_mode():
            score, anomaly_map = network(image)
        answers.append((score.clone(), anomaly_map.clone()))
        del network

    assert vit.resample_abs_pos_embed is binding, "construction rebound timm"
    (first_score, first_map), (second_score, second_map) = answers
    assert tuple(first_score.shape) == (1, 1)
    assert tuple(first_map.shape) == (1, 1, SIZE, SIZE)
    assert torch.equal(first_score, second_score)
    assert torch.equal(first_map, second_map)
    assert 0.0 <= float(first_score) <= 1.0
