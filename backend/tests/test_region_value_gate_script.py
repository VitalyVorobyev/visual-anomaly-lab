"""Decision-contract tests for the reproducible M10 measurement script."""

from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "region-value-gate.py"
NAMESPACE = runpy.run_path(str(SCRIPT))
decide = cast(Callable[[dict[str, Any], str], dict[str, Any]], NAMESPACE["_decision"])


def _leg(pixel_roc_auc: float, au_pro: float, *, failure_rate: float = 0.0) -> dict[str, Any]:
    return {
        "build": {"failure_rate": failure_rate},
        "run": {
            "infer": {
                "metrics": {
                    "test": {
                        "pixel": {
                            "pixel_roc_auc": pixel_roc_auc,
                            "au_pro": au_pro,
                        }
                    }
                }
            }
        },
    }


def test_value_gate_reads_nested_pixel_metrics_and_accepts_clear_gains() -> None:
    categories = {
        "one": {"identity": _leg(0.80, 0.70), "localized": _leg(0.83, 0.73)},
        "two": {"identity": _leg(0.75, 0.65), "localized": _leg(0.77, 0.67)},
    }

    decision = decide(categories, "mobile_sam")

    assert decision["mean_primary_delta"]["pixel_roc_auc"] == pytest.approx(0.025)
    assert decision["mean_primary_delta"]["au_pro"] == pytest.approx(0.025)
    assert decision["localization_becomes_default"] is True
    assert decision["recommendation"].startswith("mobile_sam")


def test_value_gate_rejects_one_large_class_loss_and_any_build_failure() -> None:
    categories = {
        "one": {"identity": _leg(0.80, 0.70), "localized": _leg(0.95, 0.85)},
        "two": {
            "identity": _leg(0.90, 0.80),
            "localized": _leg(0.85, 0.75, failure_rate=0.01),
        },
    }

    decision = decide(categories, "foreground_threshold")

    assert decision["all_builds_complete"] is False
    assert decision["no_large_single_class_loss"] is False
    assert decision["localization_becomes_default"] is False
