"""Reading a sweep's rows back: aggregation, and comparisons that respect the pairing.

A campaign writes one row per arm per category per seed. Turning that into a verdict is
two reductions and one comparison, and each of them is a place to get the statistics
quietly wrong.

**The reductions are ordered, and the order is not arbitrary.** Seeds are averaged *within*
a category first, then categories are averaged. Pooling all rows at once would weight a
category by how many seeds happened to finish for it, so a category whose run was
interrupted would silently count less than its neighbours. Averaging in the stated order
gives every category one vote, which is what a benchmark mean is supposed to mean.

**The spread that matters is across categories, not across seeds.** A 1-shot fit re-drawn
under five seeds on VisA's `candle` moves the AUROC by a few points; `candle` and `pcb4`
differ by far more than that. So the number reported beside a mean is the standard error
over categories, and the seed spread is kept separately as what it is -- a statement about
how stable one category's arm is, not about the benchmark.

**Comparisons are paired.** Two arms are read on the *same* categories, and the statistic
is the mean of the per-category differences with its own standard error, plus a count of
how many categories each side won. Comparing two benchmark means directly would throw away
the pairing and hand back a confidence interval several times too wide, because almost all
of the variance is the category axis that both arms share.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARM_AXES: tuple[str, ...] = ("backbone", "resolution", "view", "shots", "tau", "rho")
"""The configuration axes a sweep varies. Everything else in a row is either an outcome or
a constant of the protocol."""

METRICS: tuple[str, ...] = ("image_auroc", "image_ap", "pixel_auroc", "au_pro")


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Every row of a JSON-lines sweep file.

    A partial final line is dropped rather than raising. The writer appends one line per
    row and flushes, but a sweep killed mid-write leaves a fragment, and losing one arm is
    a better outcome than refusing to read the other fifty thousand.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


def arm_key(row: dict[str, Any], axes: Sequence[str] = ARM_AXES) -> tuple[Any, ...]:
    return tuple(row[axis] for axis in axes)


def describe(key: Sequence[Any], axes: Sequence[str] = ARM_AXES) -> str:
    """A one-line name for an arm, short enough to sit in a table's first column."""
    parts = []
    for axis, value in zip(axes, key, strict=True):
        if axis == "backbone":
            parts.append(str(value))
        elif axis == "resolution":
            parts.append(f"{value}px")
        elif axis == "shots":
            parts.append(f"k={value}")
        elif axis in {"tau", "rho"}:
            parts.append(f"{axis[0]}={value:g}")
        else:
            parts.append(str(value))
    return " ".join(parts)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _standard_error(values: Sequence[float]) -> float | None:
    """The mean's standard error, or None when one sample cannot give one."""
    if len(values) < 2:
        return None
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


@dataclass(frozen=True)
class ArmSummary:
    """One configuration's standing on one benchmark, reduced in the documented order."""

    key: tuple[Any, ...]
    axes: tuple[str, ...]
    benchmark: str
    categories: tuple[str, ...]
    seeds: tuple[int, ...]
    means: dict[str, float | None]
    errors: dict[str, float | None]
    per_category: dict[str, dict[str, float]]
    seed_spread: dict[str, float | None]

    @property
    def name(self) -> str:
        return describe(self.key, self.axes)

    def worst(self, metric: str = "image_auroc") -> tuple[str, float] | None:
        """The category this arm does least well on -- where a mean hides its failures."""
        scores = self.per_category.get(metric, {})
        if not scores:
            return None
        category = min(scores, key=lambda name: scores[name])
        return category, scores[category]


def _by_category(
    rows: Iterable[dict[str, Any]], metric: str
) -> tuple[dict[str, list[float]], list[float]]:
    """Per-category lists of seed values, and the per-category seed spreads."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None:
            grouped[row["category"]].append(float(value))
    spreads = [
        math.sqrt(sum((value - _mean(values)) ** 2 for value in values) / (len(values) - 1))
        for values in grouped.values()
        if len(values) > 1
    ]
    return dict(grouped), spreads


def summarize(
    rows: Iterable[dict[str, Any]],
    *,
    benchmark: str | None = None,
    axes: Sequence[str] = ARM_AXES,
) -> list[ArmSummary]:
    """Every arm's standing, best image AUROC first.

    Arms that ran on different category sets are summarized anyway -- an interrupted sweep
    is the normal state of a campaign -- so `categories` is part of each summary and a
    reader comparing two arms should use `paired_difference`, which refuses to ignore it.
    """
    selected = [row for row in rows if benchmark is None or row["benchmark"] == benchmark]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[arm_key(row, axes)].append(row)

    summaries: list[ArmSummary] = []
    for key, arm_rows in grouped.items():
        means: dict[str, float | None] = {}
        errors: dict[str, float | None] = {}
        per_category: dict[str, dict[str, float]] = {}
        seed_spread: dict[str, float | None] = {}
        for metric in METRICS:
            by_category, spreads = _by_category(arm_rows, metric)
            if not by_category:
                means[metric] = errors[metric] = seed_spread[metric] = None
                continue
            folded = {name: _mean(values) for name, values in by_category.items()}
            per_category[metric] = folded
            ordered = list(folded.values())
            means[metric] = _mean(ordered)
            errors[metric] = _standard_error(ordered)
            seed_spread[metric] = _mean(spreads) if spreads else None
        summaries.append(
            ArmSummary(
                key=key,
                axes=tuple(axes),
                benchmark=benchmark or "all",
                categories=tuple(sorted({row["category"] for row in arm_rows})),
                seeds=tuple(sorted({int(row["seed"]) for row in arm_rows})),
                means=means,
                errors=errors,
                per_category=per_category,
                seed_spread=seed_spread,
            )
        )
    summaries.sort(key=lambda entry: entry.means.get("image_auroc") or -1.0, reverse=True)
    return summaries


@dataclass(frozen=True)
class PairedComparison:
    """Two arms read on the categories they share, differenced within each."""

    metric: str
    left: str
    right: str
    categories: tuple[str, ...]
    deltas: dict[str, float]
    mean: float
    error: float | None
    wins: int
    losses: int
    ties: int

    @property
    def decisive(self) -> bool:
        """Whether the paired mean clears twice its own standard error.

        Deliberately not a p-value. With twelve categories the normal approximation is
        doing real work, and the campaign's question is "is this difference worth building
        the plugin around", which two standard errors answers well enough. A comparison
        that lands near the line is reported as near the line.
        """
        return self.error is not None and abs(self.mean) > 2.0 * self.error


def paired_difference(
    left: ArmSummary,
    right: ArmSummary,
    *,
    metric: str = "image_auroc",
) -> PairedComparison:
    """`left` minus `right`, category by category, over the categories both ran.

    Raises when they share none. That is a configuration error -- almost always a typo in
    an arm key or a sweep that has not reached the second arm yet -- and returning a mean
    of an empty set would present it as a tie.
    """
    a = left.per_category.get(metric, {})
    b = right.per_category.get(metric, {})
    shared = tuple(sorted(set(a) & set(b)))
    if not shared:
        msg = (
            f"{left.name!r} and {right.name!r} share no category with a {metric}; "
            f"left has {len(a)}, right has {len(b)}"
        )
        raise ValueError(msg)
    deltas = {name: a[name] - b[name] for name in shared}
    ordered = list(deltas.values())
    return PairedComparison(
        metric=metric,
        left=left.name,
        right=right.name,
        categories=shared,
        deltas=deltas,
        mean=_mean(ordered),
        error=_standard_error(ordered),
        wins=sum(1 for value in ordered if value > 0),
        losses=sum(1 for value in ordered if value < 0),
        ties=sum(1 for value in ordered if value == 0),
    )


def best_over(
    summaries: Sequence[ArmSummary],
    axis: str,
    *,
    metric: str = "image_auroc",
) -> dict[Any, ArmSummary]:
    """The strongest arm at each value of one axis.

    This is how a sweep answers "which layer band is best" without the answer being an
    artefact of whatever tau happened to accompany it: every band is represented by its own
    best configuration, so the bands are compared at their own optima rather than at a
    shared setting that may suit one of them.
    """
    best: dict[Any, ArmSummary] = {}
    for summary in summaries:
        if not summary.axes or axis not in summary.axes:
            msg = f"{axis!r} is not one of the summarized axes {summary.axes}"
            raise ValueError(msg)
        value = summary.key[summary.axes.index(axis)]
        score = summary.means.get(metric)
        if score is None:
            continue
        current = best.get(value)
        if current is None or score > (current.means.get(metric) or -1.0):
            best[value] = summary
    return best


def table(
    summaries: Sequence[ArmSummary],
    *,
    limit: int = 20,
    metric: str = "image_auroc",
) -> Iterator[str]:
    """A fixed-width leaderboard of the best arms *by the metric it prints*.

    `summarize` returns its list ordered by image AUROC, which is the right default and
    the wrong order for any other column: a localization table sorted by a detection score
    puts a 0.933 above a 0.945 and reads as a ranking. The sort happens here instead.
    """
    ranked = sorted(
        (entry for entry in summaries if entry.means.get(metric) is not None),
        key=lambda entry: entry.means[metric] or -1.0,
        reverse=True,
    )
    shown = ranked[:limit]
    if not shown:
        yield f"no arm reported {metric}"
        return
    width = max(len(entry.name) for entry in shown)
    yield f"{'arm':<{width}}  {metric:>10}  {'+/-':>7}  {'seed sd':>7}  {'cats':>4}  worst"
    for entry in shown:
        mean = entry.means[metric]
        error = entry.errors.get(metric)
        spread = entry.seed_spread.get(metric)
        worst = entry.worst(metric)
        assert mean is not None
        yield (
            f"{entry.name:<{width}}  {mean:>10.4f}  "
            f"{'' if error is None else format(error, '.4f'):>7}  "
            f"{'' if spread is None else format(spread, '.4f'):>7}  "
            f"{len(entry.categories):>4}  "
            f"{'' if worst is None else f'{worst[0]} {worst[1]:.3f}'}"
        )


def _axis_report(summaries: Sequence[ArmSummary], axis: str, metric: str) -> Iterator[str]:
    """One axis, each of its values at its own best configuration."""
    best = best_over(summaries, axis, metric=metric)
    if len(best) < 2:
        return
    yield ""
    yield f"-- {axis}, each at its own optimum --"
    ranked = sorted(best.items(), key=lambda pair: -(pair[1].means[metric] or -1.0))
    width = max(len(str(value)) for value, _ in ranked)
    leader = ranked[0][1]
    for value, arm in ranked:
        score = arm.means[metric]
        assert score is not None
        line = f"  {value!s:<{width}}  {score:.4f}   {arm.name}"
        if arm is not leader:
            try:
                delta = paired_difference(leader, arm, metric=metric)
            except ValueError:
                yield line
                continue
            mark = "*" if delta.decisive else " "
            line += (
                f"   [{delta.mean:+.4f}{mark} {delta.wins}-{delta.losses}"
                f" of {len(delta.categories)}]"
            )
        yield line


def main(argv: Sequence[str] | None = None) -> int:
    """Read a sweep file and print its leaderboard and its per-axis verdicts.

    The per-axis blocks are the part worth reading: each value of an axis is shown at its
    own best configuration, with a paired comparison against the leader. A `*` marks a
    difference larger than twice its standard error over the categories the two arms share.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m research.subspace_ad.report")
    parser.add_argument("path", type=Path, help="the sweep's JSON-lines output")
    parser.add_argument("--benchmark", default=None, help="restrict to one benchmark")
    parser.add_argument("--metric", default="image_auroc", choices=METRICS)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--axes",
        nargs="*",
        default=list(ARM_AXES),
        help="which axes to break down; empty for none",
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.path)
    if not rows:
        print(f"{args.path} holds no rows")
        return 1
    benchmarks = sorted({row["benchmark"] for row in rows})
    targets = [args.benchmark] if args.benchmark else benchmarks
    for benchmark in targets:
        summaries = summarize(rows, benchmark=benchmark)
        categories = sorted({row["category"] for row in rows if row["benchmark"] == benchmark})
        print(f"\n=== {benchmark}: {len(summaries)} arms over {len(categories)} categories ===")
        for line in table(summaries, limit=args.top, metric=args.metric):
            print(line)
        for axis in args.axes:
            for line in _axis_report(summaries, axis, args.metric):
                print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - a thin argparse shell
    raise SystemExit(main())
