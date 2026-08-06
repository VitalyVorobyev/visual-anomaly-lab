"""The thumbnail pre-warm job (§9).

Tiers are generated lazily on first request, which is correct but makes the first scroll
through a freshly imported dataset the slowest one. This job renders them up front.

It is the one genuinely slow operation in M2: decoding and re-encoding measures at roughly
160 ms per image, so a few hundred images is minutes rather than seconds — which is
exactly why the job system exists and why this reports progress and honours cancellation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.media.cache import TIERS, ImageTier, ensure_cached
from anomaly_lab.media.decode import UnreadableImageError

MAX_FAILURES_REPORTED = 50


def _cacheable_tiers() -> list[ImageTier]:
    return [tier for tier, spec in TIERS.items() if spec.cached]


class PrewarmParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: int
    tiers: list[ImageTier] = Field(default_factory=_cacheable_tiers)


def run_prewarm_job(context: JobContext) -> dict[str, Any]:
    params = PrewarmParams.model_validate(dict(context.params))
    tiers = [tier for tier in params.tiers if TIERS[tier].cached]

    with connection(context.settings.db_path) as conn:
        dataset = datasets_repo.get_dataset(conn, params.dataset_id)
        if dataset is None:
            msg = f"no dataset with id {params.dataset_id}"
            raise LookupError(msg)
        images = images_repo.list_images_for_dataset(conn, dataset.id)

    total = len(images) * len(tiers)
    context.log(f"rendering {len(tiers)} tiers for {len(images)} images")

    rendered = 0
    failures: list[str] = []
    completed = 0

    for image in images:
        # Checked per image rather than per tier: an image's tiers are cheap together and
        # stopping between them would leave a half-warmed sample.
        context.raise_if_cancelled()
        for tier in tiers:
            try:
                ensure_cached(context.settings, image, tier)
                rendered += 1
            except (UnreadableImageError, OSError) as exc:
                failures.append(f"{image.path}: {exc}")
            completed += 1
        if total:
            context.progress(completed / total, f"{completed} of {total}")

    if failures:
        context.log(f"{len(failures)} tiers could not be rendered", level="warning")

    return {
        "dataset_id": dataset.id,
        "images": len(images),
        "tiers": [tier.value for tier in tiers],
        "rendered": rendered,
        "failed": len(failures),
        "failures": failures[:MAX_FAILURES_REPORTED],
    }
