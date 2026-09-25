"""`dino_linear_det`'s torch-free half: how probabilities become boxes, the shared component
decoder, and the configuration the form is generated from. What needs an encoder is in
`test_dl_dino_linear_det.py`."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import MAX_INSTANCES_PER_IMAGE
from anomaly_lab.models.color_detector import component_boxes
from anomaly_lab.models.dino_linear_det import (
    DinoLinearDetConfig,
    DinoLinearDetModel,
    decode,
)
from anomaly_lab.models.dino_linear_seg import LogitBias, PixelSampling
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import describe

CLASSES = ("rust", "moss")


def _probability(height: int = 12, width: int = 12) -> np.ndarray:
    """Background everywhere, two `rust` blobs of different certainty, one `moss` blob."""
    probability = np.zeros((3, height, width), dtype=np.float32)
    probability[0] = 0.9
    probability[1] = 0.05
    probability[2] = 0.05
    probability[:, 1:4, 1:5] = np.array([0.2, 0.7, 0.1])[:, None, None]  # rust, 3x4
    probability[:, 6:10, 7:9] = np.array([0.05, 0.9, 0.05])[:, None, None]  # rust, 4x2
    probability[:, 8:11, 1:3] = np.array([0.1, 0.1, 0.8])[:, None, None]  # moss, 3x2
    return probability


def test_it_declares_object_detection_alone_needs_torch_and_exports_nothing() -> None:
    description = describe("dino_linear_det")
    assert description.capabilities.tasks == [Task.OBJECT_DETECTION]
    assert description.capabilities.portable_formats == []
    assert description.capabilities.requires_training
    for name, field in description.config_schema["properties"].items():
        assert field.get("description"), name
    # The head's defaults are the ones its segmentation gate measured.
    config = DinoLinearDetConfig()
    assert config.pixel_sampling is PixelSampling.PER_CLASS
    assert config.logit_bias is LogitBias.HELD_OUT_IOU
    assert (config.min_area, config.max_detections) == (4, MAX_INSTANCES_PER_IMAGE)
    with pytest.raises(ValidationError):
        DinoLinearDetConfig(max_detections=MAX_INSTANCES_PER_IMAGE + 1)
    with pytest.raises(ValidationError):
        DinoLinearDetConfig(min_area=0)


def test_importing_the_module_does_not_import_torch() -> None:
    probe = (
        "import sys, anomaly_lab.models.dino_linear_det; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_it_refuses_a_frame_the_encoder_cannot_tile() -> None:
    config = DinoLinearDetConfig()
    DinoLinearDetModel.check_input(config, PreprocessingConfig(width=224, height=224))
    with pytest.raises(ValueError, match="14"):
        DinoLinearDetModel.check_input(config, PreprocessingConfig(width=230, height=224))


def test_each_component_of_the_argmax_is_one_box_ranked_by_its_mean_probability() -> None:
    found = decode(_probability(), CLASSES, min_area=1, max_detections=10)
    assert [(item.label_key, item.box) for item in found] == [
        ("rust", (7.0, 6.0, 9.0, 10.0)),
        ("moss", (1.0, 8.0, 3.0, 11.0)),
        ("rust", (1.0, 1.0, 5.0, 4.0)),
    ]
    assert [item.confidence for item in found] == pytest.approx([0.9, 0.8, 0.7])


def test_small_components_and_detections_past_the_cap_are_dropped() -> None:
    assert len(decode(_probability(), CLASSES, min_area=1, max_detections=2)) == 2
    kept = decode(_probability(), CLASSES, min_area=7, max_detections=10)
    # The 3x4 rust blob and the 4x2 rust blob survive; the 3x2 moss blob does not.
    assert [item.label_key for item in kept] == ["rust", "rust"]


def test_a_class_whose_probability_is_zero_everywhere_is_never_boxed() -> None:
    probability = _probability()
    probability[2] = 0.0
    found = decode(probability, CLASSES, min_area=1, max_detections=10)
    assert {item.label_key for item in found} == {"rust"}
    assert (
        decode(
            np.stack([np.ones((4, 4)), np.zeros((4, 4))]), ("rust",), min_area=1, max_detections=10
        )
        == []
    )


def test_diagonal_neighbours_are_one_component_and_each_box_covers_its_pixels() -> None:
    labels = np.zeros((6, 6), dtype=np.int64)
    labels[0, 0] = labels[1, 1] = labels[2, 2] = 1
    labels[4:6, 4:6] = 1
    plane = np.where(labels == 1, 0.6, 0.0)
    found = component_boxes(labels, {1: plane}, ("rust",), min_area=1, max_detections=10)
    assert sorted(item.box for item in found) == [(0.0, 0.0, 3.0, 3.0), (4.0, 4.0, 6.0, 6.0)]
