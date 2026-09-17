"""The sweep's aggregation, tested where it could silently give a wrong verdict.

These are not tests that the arithmetic mean works. They are tests of the three choices
`report.py` makes that a plausible alternative implementation would make differently, and
that would change which configuration the campaign declares the winner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from research.subspace_ad.report import (
    ARM_AXES,
    best_over,
    load_rows,
    paired_difference,
    summarize,
)


def _row(
    *,
    category: str,
    seed: int,
    image_auroc: float,
    view: str = "final7-mean",
    tau: float = 0.97,
    shots: int = 4,
    backbone: str = "dinov2_vit_s14_reg4",
    resolution: int = 672,
) -> dict[str, Any]:
    return {
        "backbone": backbone,
        "resolution": resolution,
        "view": view,
        "benchmark": "visa",
        "category": category,
        "shots": shots,
        "seed": seed,
        "tau": tau,
        "rho": 0.01,
        "image_auroc": image_auroc,
        "image_ap": None,
        "pixel_auroc": None,
        "au_pro": None,
    }


def test_seeds_fold_inside_a_category_before_categories_are_averaged() -> None:
    """An interrupted sweep must not let a category's vote depend on its seed count.

    `candle` finished four seeds and `pcb1` finished one. Pooling all five rows would give
    `candle` four fifths of the mean; folding seeds first gives each category one vote.
    The two answers differ by a lot here precisely so that a regression cannot hide.
    """
    rows = [_row(category="candle", seed=seed, image_auroc=0.90) for seed in range(4)]
    rows.append(_row(category="pcb1", seed=0, image_auroc=0.50))

    summary = summarize(rows, benchmark="visa")[0]

    assert summary.means["image_auroc"] == pytest.approx(0.70)
    assert summary.means["image_auroc"] != pytest.approx(sum(r["image_auroc"] for r in rows) / 5)


def test_the_reported_error_is_across_categories_and_the_seed_spread_is_kept_apart() -> None:
    """Two arms can share a benchmark mean and differ entirely in where the variance is.

    Here every seed of a category agrees exactly, so the seed spread is zero while the
    category spread is large. An implementation that reported the seed standard error
    would print a confidence interval of zero on a number that swings by twenty points
    between categories.
    """
    rows = [_row(category="candle", seed=seed, image_auroc=0.90) for seed in range(3)] + [
        _row(category="pcb1", seed=seed, image_auroc=0.70) for seed in range(3)
    ]

    summary = summarize(rows, benchmark="visa")[0]

    assert summary.means["image_auroc"] == pytest.approx(0.80)
    assert summary.seed_spread["image_auroc"] == pytest.approx(0.0)
    error = summary.errors["image_auroc"]
    assert error is not None and error == pytest.approx(0.1)


def test_a_comparison_differences_within_a_category_rather_than_between_two_means() -> None:
    """The pairing is the whole point: it removes the variance both arms share.

    Both arms swing by thirty points across these categories and one is better by exactly
    two everywhere. Unpaired, that difference sits well inside either arm's own spread;
    paired, it is exact and its standard error is zero.
    """
    left = summarize(
        [
            _row(category="candle", seed=0, image_auroc=0.92, view="final7-mean"),
            _row(category="pcb1", seed=0, image_auroc=0.62, view="final7-mean"),
        ],
        benchmark="visa",
    )[0]
    right = summarize(
        [
            _row(category="candle", seed=0, image_auroc=0.90, view="last-mean"),
            _row(category="pcb1", seed=0, image_auroc=0.60, view="last-mean"),
        ],
        benchmark="visa",
    )[0]

    comparison = paired_difference(left, right)

    assert comparison.mean == pytest.approx(0.02)
    assert comparison.error == pytest.approx(0.0)
    assert (comparison.wins, comparison.losses, comparison.ties) == (2, 0, 0)
    assert comparison.categories == ("candle", "pcb1")


def test_a_comparison_uses_only_the_categories_both_arms_reached() -> None:
    """A sweep in progress has arms at different depths; the shared set is the honest one."""
    left = summarize(
        [
            _row(category="candle", seed=0, image_auroc=0.92, view="final7-mean"),
            _row(category="pcb1", seed=0, image_auroc=0.62, view="final7-mean"),
        ],
        benchmark="visa",
    )[0]
    right = summarize(
        [_row(category="candle", seed=0, image_auroc=0.90, view="last-mean")],
        benchmark="visa",
    )[0]

    comparison = paired_difference(left, right)

    assert comparison.categories == ("candle",)
    assert comparison.mean == pytest.approx(0.02)


def test_two_arms_with_no_category_in_common_are_a_fault_not_a_tie() -> None:
    left = summarize(
        [_row(category="candle", seed=0, image_auroc=0.92, view="final7-mean")],
        benchmark="visa",
    )[0]
    right = summarize(
        [_row(category="pcb1", seed=0, image_auroc=0.90, view="last-mean")],
        benchmark="visa",
    )[0]

    with pytest.raises(ValueError, match="share no category"):
        paired_difference(left, right)


def test_each_axis_value_is_represented_by_its_own_best_configuration() -> None:
    """A band must be judged at its own optimum, not at whatever tau suits its rival.

    `last-mean` is poor at tau = 0.95 and strong at 0.99; `final7-mean` is the reverse.
    Comparing the two bands at any single tau picks a winner by picking a tau.
    """
    rows = [
        _row(category="candle", seed=0, image_auroc=0.70, view="last-mean", tau=0.95),
        _row(category="candle", seed=0, image_auroc=0.94, view="last-mean", tau=0.99),
        _row(category="candle", seed=0, image_auroc=0.91, view="final7-mean", tau=0.95),
        _row(category="candle", seed=0, image_auroc=0.60, view="final7-mean", tau=0.99),
    ]

    best = best_over(summarize(rows, benchmark="visa"), "view")

    assert best["last-mean"].means["image_auroc"] == pytest.approx(0.94)
    assert best["final7-mean"].means["image_auroc"] == pytest.approx(0.91)
    assert best["last-mean"].key[ARM_AXES.index("tau")] == 0.99


def test_a_sweep_killed_mid_write_still_reads_back(tmp_path: Path) -> None:
    """Losing the fragment beats refusing the file.

    A campaign appends and flushes per row, so a kill lands inside one line. That row is
    gone either way; the question is whether the other rows come back.
    """
    path = tmp_path / "rows.jsonl"
    complete = [_row(category="candle", seed=0, image_auroc=0.9)] * 2
    path.write_text(
        "\n".join(json.dumps(row) for row in complete) + '\n{"backbone": "dinov2',
        encoding="utf-8",
    )

    assert len(load_rows(path)) == 2


def test_an_arm_that_reported_no_pixel_metric_summarizes_as_none_rather_than_zero() -> None:
    """The repository's rule for an uncomputable metric, applied to the sweep's rows.

    Only some taus get pixel metrics, so most arms carry `None` there. A zero would sort
    them beside genuinely bad localization instead of out of the ranking.
    """
    summary = summarize([_row(category="candle", seed=0, image_auroc=0.9)], benchmark="visa")[0]

    assert summary.means["au_pro"] is None
    assert summary.errors["au_pro"] is None
    assert summary.worst("au_pro") is None
