#!/usr/bin/env python3
"""Pre-flight for a built region profile: does the crop keep every annotated defect pixel?

The public region-profile gate failed the way crops fail: image-level ROC-AUC went *up*
while the crop had silently dropped a quarter of all defect pixels — a better-looking
number measuring a worse protocol. The accounting that catches this
(`pixel.uncovered_defect_pixels`) exists, but only after a full train-and-infer pass has
been paid for. This script answers the same question from the built manifest and the
resolved ground truth alone, before any experiment is created.

For every image whose ground truth resolves (newest completed annotation, else imported
mask), the source mask is pushed through the build's own `prepare_mask` and the two pixel
counts are compared. Any loss on any image is a failing verdict: raise the profile's
`padding_fraction` (or fix the extractor) and rebuild before spending training time.

**Read-only, numpy + Pillow only, no torch.** Dataset-agnostic: the profile id names the
dataset; nothing here assumes what is in the pictures.

    uv run --directory backend python ../scripts/region-truth-coverage.py 19
    uv run --directory backend python ../scripts/region-truth-coverage.py 19 --data-dir ../data

Exit codes: 0 all truth survives the crop, 1 pixels lost, 2 profile/build not usable.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parent.parent
BACKEND_SRC = REPOSITORY / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from anomaly_lab.config import Settings  # noqa: E402
from anomaly_lab.db.repositories import annotations as annotations_repo  # noqa: E402
from anomaly_lab.db.repositories import region_profiles as profiles_repo  # noqa: E402
from anomaly_lab.models.preprocessing import load_mask  # noqa: E402
from anomaly_lab.regions.preparation import load_prepared_build, read_build_summary  # noqa: E402


def dataset_image_ids(conn: sqlite3.Connection, dataset_id: int) -> list[int]:
    rows = conn.execute(
        """
        SELECT image.id AS id
          FROM image
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
         ORDER BY image.id
        """,
        (dataset_id,),
    ).fetchall()
    return [int(row["id"]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile_id", type=int, help="Region profile revision with a built output.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY / "data",
        help="App data directory (holds app.sqlite3 and region-profiles/).",
    )
    args = parser.parse_args(argv)

    settings = Settings(data_dir=args.data_dir)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        profile = profiles_repo.get_profile(conn, args.profile_id)
        if profile is None:
            print(f"no region profile {args.profile_id} in {settings.db_path}", file=sys.stderr)
            return 2
        summary = read_build_summary(settings, profile.id)
        if summary is None:
            print(f"region profile {profile.id} has no completed build", file=sys.stderr)
            return 2
        build = load_prepared_build(settings, profile, manifest_sha256=summary.manifest_sha256)

        image_ids = dataset_image_ids(conn, profile.dataset_id)
        truths = annotations_repo.resolve_ground_truth_masks(conn, image_ids, verify_bytes=True)
    finally:
        conn.close()

    if not truths:
        print(
            f"profile {profile.id} (dataset {profile.dataset_id}): no image resolves ground "
            "truth — nothing to check, and nothing this build can be judged against."
        )
        return 0

    print(
        f"profile {profile.id} · dataset {profile.dataset_id} · {profile.extractor_type} -> "
        f"{profile.prepared_width}x{profile.prepared_height} · "
        f"{len(truths)}/{len(image_ids)} images with resolved truth"
    )
    print(f"{'image':>7}  {'truth px':>9}  {'kept px':>9}  {'lost':>6}  {'crop frac':>9}")

    lost_images = 0
    total_truth = 0
    total_lost = 0
    crop_fractions: list[float] = []
    for image_id in sorted(truths):
        transform = build.transform_for(image_id)
        mask = load_mask(Path(truths[image_id].path))
        if mask.shape != (transform.source_height, transform.source_width):
            # An imported mask may be stored at a different resolution; judge it in the
            # source frame the transform was resolved against.
            mask = load_mask(
                Path(truths[image_id].path),
                size=(transform.source_width, transform.source_height),
            )
        source_px = int(mask.sum())
        kept_px = int(transform.prepare_mask(mask).sum())
        crop_area = transform.crop_width * transform.crop_height
        source_area = transform.source_width * transform.source_height
        crop_fraction = crop_area / source_area
        crop_fractions.append(crop_fraction)

        # NEAREST resampling can only merge pixels, not move them outside the crop: a
        # cropped-away pixel is a real loss, a resampling merge is not. Compare against
        # the crop, not the resample, so the verdict blames geometry alone.
        cropped_px = int(
            mask[
                transform.crop_top : transform.crop_bottom,
                transform.crop_left : transform.crop_right,
            ].sum()
        )
        lost = source_px - cropped_px
        total_truth += source_px
        if lost > 0:
            lost_images += 1
            total_lost += lost
        marker = "  <-- LOST" if lost > 0 else ""
        print(
            f"{image_id:>7}  {source_px:>9}  {kept_px:>9}  {lost:>6}  {crop_fraction:>9.3f}"
            f"{marker}"
        )

    mean_crop = float(np.mean(crop_fractions))
    print(
        f"\n{len(truths)} masks · {total_truth} truth pixels · mean crop fraction "
        f"{mean_crop:.3f}"
    )
    if lost_images:
        print(
            f"FAIL: {lost_images} image(s) lose {total_lost} defect pixel(s) to the crop. "
            "Do not build experiments on this profile — raise padding_fraction or fix the "
            "extractor, then rebuild."
        )
        return 1
    print("OK: every resolved truth pixel survives the crop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
