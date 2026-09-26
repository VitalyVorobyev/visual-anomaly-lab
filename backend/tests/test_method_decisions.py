"""What the method picker is told a person decides, and where each method stands.

A configuration form is generated from the plugin's schema, and a field marked `x-primary`
is shown in front of the rest. Every method names a few such decisions and folds the others;
a form of nineteen equal fields says nothing about which of them matter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import MethodStatus, ModelDescription
from anomaly_lab.models.registry import RECOMMENDED, STATUS, describe_all, registered_keys


def _primary(schema: dict[str, object]) -> list[str]:
    properties = schema.get("properties")
    assert isinstance(properties, dict)
    return [name for name, node in properties.items() if node.get("x-primary") is True]


@pytest.mark.parametrize("entry", describe_all(), ids=lambda entry: entry.key)
def test_every_method_marks_between_one_and_four_decisions(entry: ModelDescription) -> None:
    primary = _primary(entry.config_schema)
    assert 1 <= len(primary) <= 4, (entry.key, primary)


def test_reading_every_schema_imports_no_torch() -> None:
    """The picker describes every method; the hint must not cost a deep-learning import."""
    probe = (
        "import sys; from anomaly_lab.models.registry import describe_all; describe_all(); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_every_registered_method_has_a_recorded_status() -> None:
    assert set(STATUS) == set(registered_keys())


def test_each_task_floor_is_its_numpy_baseline() -> None:
    floors = {key for key, status in STATUS.items() if status is MethodStatus.FLOOR}
    assert floors == {"pixel_reference", "color_prototype", "color_classifier", "color_detector"}


def test_a_recommended_method_serves_its_task_and_is_supported() -> None:
    described = {entry.key: entry for entry in describe_all()}
    for task, key in RECOMMENDED.items():
        entry = described[key]
        assert task in entry.capabilities.tasks
        assert entry.status is MethodStatus.SUPPORTED
        assert task in entry.recommended_for


def test_the_listing_carries_status_and_recommendation() -> None:
    described = {entry.key: entry for entry in describe_all()}
    assert described["proto_seg"].recommended_for == [Task.FEW_SHOT_SEGMENTATION]
    assert described["dino_linear_seg"].status is MethodStatus.SUPPORTED
    assert described["dino_linear_det"].status is MethodStatus.EXPERIMENTAL
    assert described["dino_linear_det"].recommended_for == []
    assert described["anomalyvfm_anomalib"].status is MethodStatus.SUPPORTED


def test_only_the_aggregation_of_the_evaluation_options_reaches_every_task() -> None:
    """Every evaluator rebuilds sample rows; only the anomaly one reads the pixel options."""
    from anomaly_lab.eval.runner import EvalConfig

    properties = EvalConfig.model_json_schema()["properties"]
    scoped = {name: node.get("x-tasks") for name, node in properties.items()}
    assert scoped == {
        "aggregation": None,
        "channel_normalization": None,
        "pixel_metrics": ["anomaly"],
        "pixel_bins": ["anomaly"],
        "localization_tolerance": ["anomaly"],
    }
