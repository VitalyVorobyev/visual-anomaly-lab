"""Command line for the SubspaceAD campaign.

    uv run --directory backend python -m research.subspace_ad \\
        --benchmarks visa --backbones dinov2_vit_b14 --resolutions 672 \\
        --views mid7 last --shots 1 2 4 --seeds 0 1 2 3 4 \\
        --output ../results/subspace/phase1.jsonl --resume

Every axis is a list, and the loop arranges them by what they cost rather than by the order
they are typed in. A view is `<band>` optionally suffixed `:concat` and `:l2` -- for example
`mid7`, `mid7:concat`, `last:l2` -- where the band names come from `features.LAYER_BANDS`.

Output goes to the gitignored `results/` tree. Nothing here writes to the application
database or reaches its API (ADR-0038).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anomaly_lab.models.dino_backbone import DinoBackbone
from research.subspace_ad.campaign import CampaignSpec, run_campaign
from research.subspace_ad.features import LAYER_BANDS, Aggregation, FeatureView, RotationFill

REPOSITORY = Path(__file__).resolve().parents[3]


def parse_view(text: str) -> FeatureView:
    """`<band>[:concat][:l2]` into a `FeatureView`, naming the bands on a typo."""
    parts = text.split(":")
    band = LAYER_BANDS.get(parts[0])
    if band is None:
        known = ", ".join(sorted(LAYER_BANDS))
        msg = f"unknown layer band {parts[0]!r}; known bands are {known}"
        raise argparse.ArgumentTypeError(msg)
    aggregation = Aggregation.MEAN
    normalize = False
    for modifier in parts[1:]:
        if modifier == "concat":
            aggregation = Aggregation.CONCAT
        elif modifier == "l2":
            normalize = True
        else:
            msg = f"unknown view modifier {modifier!r}; expected 'concat' or 'l2'"
            raise argparse.ArgumentTypeError(msg)
    return FeatureView(band=band, aggregation=aggregation, l2_normalize=normalize)


def parse_backbone(text: str) -> DinoBackbone:
    try:
        return DinoBackbone(text)
    except ValueError as exc:
        known = ", ".join(entry.value for entry in DinoBackbone)
        msg = f"unknown backbone {text!r}; known encoders are {known}"
        raise argparse.ArgumentTypeError(msg) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmarks", nargs="+", default=["visa"])
    parser.add_argument("--categories", nargs="*", default=[], help="Empty means every one.")
    parser.add_argument("--backbones", nargs="+", type=parse_backbone, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[672])
    parser.add_argument("--views", nargs="+", type=parse_view, required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--taus", nargs="+", type=float, default=[0.95, 0.97, 0.99, 1.0])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.01])
    parser.add_argument(
        "--pixel-taus",
        nargs="*",
        type=float,
        default=[],
        help="Thresholds that also get pixel metrics. Empty means all of --taus.",
    )
    parser.add_argument("--augmentations", type=int, default=30, help="The paper's Na.")
    parser.add_argument(
        "--rotation-fill",
        type=RotationFill,
        choices=list(RotationFill),
        default=RotationFill.ZEROS,
    )
    parser.add_argument("--no-final-norm", action="store_true")
    parser.add_argument("--no-pixel-metrics", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--datasets-dir", type=Path, default=REPOSITORY / "datasets")
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY / "data" / "model-cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = CampaignSpec(
        benchmarks=tuple(args.benchmarks),
        backbones=tuple(args.backbones),
        resolutions=tuple(args.resolutions),
        views=tuple(args.views),
        shots=tuple(args.shots),
        seeds=tuple(args.seeds),
        taus=tuple(args.taus),
        rhos=tuple(args.rhos),
        pixel_taus=tuple(args.pixel_taus),
        augmentations=args.augmentations,
        rotation_fill=args.rotation_fill,
        final_norm=not args.no_final_norm,
        pixel_metrics=not args.no_pixel_metrics,
        batch_size=args.batch_size,
        categories=tuple(args.categories),
    )
    arms = len(spec.views) * len(spec.shots) * len(spec.seeds) * len(spec.taus) * len(spec.rhos)
    print(
        f"{len(spec.backbones)} backbone(s) x {len(spec.resolutions)} resolution(s), "
        f"{arms} arms per category",
        file=sys.stderr,
        flush=True,
    )
    written = run_campaign(
        spec,
        datasets_dir=args.datasets_dir,
        cache_dir=args.cache_dir,
        output=args.output,
        resume=args.resume,
    )
    print(f"wrote {written} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
