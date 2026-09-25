"""A few-shot method fitted on the reference studio's current references, for looking (ADR-0040).

The studio asks "what would these references find in that image?" many times while the
references change, and a job per question would pay a model load for a tenth of a second
of work. So the resident worker (ADR-0026) holds one `PreviewSession`: the method at its
defaults, fitted on the references through the same `PreparedClassTargets` and region
build a run uses, answering one image at a time.

A preview is not a result. It is written under `previews/`, never into an experiment,
evaluated by nothing and stored for nobody; "freeze as experiment" is how references
become a run, and the run page is where one is read.

**The generation is the identity.** It fingerprints the class, the method, the pinned
region build and every reference image's pinned truth, so a change to any of them is a
different resident rather than a stale one — the same guarantee `generation_of` gives an
experiment's checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.annotations.class_truth import ClassTruth, resolve_class_truth
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.db.repositories.images import SplitImage
from anomaly_lab.domain.entities import Subset, Task
from anomaly_lab.experiments.context import to_records
from anomaly_lab.experiments.targets import PreparedClassTargets
from anomaly_lab.map_files import read_map
from anomaly_lab.models.base import AnomalyModel, InferContext, NullReporter, TrainContext
from anomaly_lab.models.device import resolve_device
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import UnknownModelError, get_model_class
from anomaly_lab.regions.preparation import (
    PreparedRegionBuild,
    load_prepared_build,
    read_build_summary,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

MAX_REFERENCES = 10
# One batch is one request under the resident's timeout, and one page of the studio's rail.
MAX_BATCH = 48


class PreviewError(ValueError):
    """The preview cannot be fitted, for a reason the reader can act on."""


class PreviewSpec(BaseModel):
    """What a preview is fitted from. Serialised whole onto the resident's command line."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    class_key: str
    method: str
    profile_id: int
    references: list[int] = Field(min_length=1, max_length=MAX_REFERENCES)

    def canonical(self) -> str:
        spec = self.model_copy(update={"references": sorted(set(self.references))})
        return json.dumps(spec.model_dump(mode="json"), sort_keys=True)


@dataclass(frozen=True)
class Resolved:
    spec: PreviewSpec
    build: PreparedRegionBuild
    reference_images: list[SplitImage]
    truths: dict[int, ClassTruth]
    generation: str


def _split_images(conn: sqlite3.Connection, sample_ids: list[int]) -> list[SplitImage]:
    found: list[SplitImage] = []
    for sample_id in sample_ids:
        sample = samples_repo.get_sample(conn, sample_id)
        if sample is None:
            raise PreviewError(f"no sample with id {sample_id}")
        for image in images_repo.list_images_for_sample(conn, sample_id):
            found.append(
                SplitImage(
                    image_id=image.id,
                    sample_id=sample_id,
                    channel=None,
                    path=image.path,
                    sha256=image.sha256,
                    label=sample.label,
                    subset=Subset.TRAIN,
                )
            )
    return found


def resolve(settings: Settings, spec: PreviewSpec) -> Resolved:
    """Check the spec against the catalogue and fingerprint what it would fit on."""
    try:
        model_class = get_model_class(spec.method)
    except UnknownModelError as exc:
        raise PreviewError(str(exc)) from exc
    if Task.FEW_SHOT_SEGMENTATION not in model_class.capabilities().tasks:
        raise PreviewError(f"method {spec.method!r} does not segment a class from references")
    if not model_class.availability().available:
        raise PreviewError(model_class.availability().reason or f"{spec.method} is unavailable")

    with connection(settings.db_path) as conn:
        profile = region_profiles_repo.get_profile(conn, spec.profile_id)
        if profile is None or profile.dataset_id != spec.dataset_id:
            raise PreviewError(f"no region profile {spec.profile_id} in dataset {spec.dataset_id}")
        summary = read_build_summary(settings, profile.id)
        if summary is None or summary.failed:
            raise PreviewError(f"region profile {profile.id} has no complete build")
        try:
            model_class.check_input(
                model_class.config_model().model_validate({}),
                PreprocessingConfig(width=profile.prepared_width, height=profile.prepared_height),
            )
        except ValueError as exc:
            raise PreviewError(str(exc)) from exc
        build = load_prepared_build(settings, profile, manifest_sha256=summary.manifest_sha256)
        images = _split_images(conn, sorted(set(spec.references)))
        truths = resolve_class_truth(
            conn, spec.dataset_id, [image.image_id for image in images], spec.class_key
        )

    answered = [image for image in images if image.image_id in truths]
    if not answered:
        raise PreviewError(f"no reference has ground truth for {spec.class_key!r}")
    identity = [spec.canonical(), summary.manifest_sha256] + [
        f"{image.image_id}:{truths[image.image_id].identity}" for image in answered
    ]
    generation = hashlib.sha256("\n".join(identity).encode()).hexdigest()
    return Resolved(spec, build, answered, truths, generation)


def preview_dir(settings: Settings, generation: str) -> Path:
    return settings.data_dir / "previews" / generation[:24]


class PreviewSession:
    """One fitted preview: fitted once, then asked about one image at a time."""

    def __init__(self, settings: Settings, spec: PreviewSpec) -> None:
        resolved = resolve(settings, spec)
        model_class = get_model_class(spec.method)
        self._build = resolved.build
        self._directory = preview_dir(settings, resolved.generation)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._preprocessing = PreprocessingConfig(
            width=resolved.build.profile.prepared_width,
            height=resolved.build.profile.prepared_height,
        )
        device = resolve_device(model_class.capabilities().preferred_device)
        self.device = device.device.value
        self._model: AnomalyModel = model_class.build({})
        self._cache_dir = settings.model_cache_dir
        diagnostics = DiagnosticWriter(self._directory / "diagnostics", enabled=False)
        self._model.fit(
            to_records(resolved.reference_images, resolved.build),
            TrainContext(
                artifact_dir=self._directory,
                cache_dir=self._cache_dir,
                preprocessing=self._preprocessing,
                device=device.device,
                reporter=NullReporter(),
                diagnostics=diagnostics,
                targets=PreparedClassTargets(spec.class_key, resolved.truths, resolved.build),
            ),
        )
        build = resolved.build
        self._infer = InferContext(
            artifact_dir=self._directory,
            cache_dir=self._cache_dir,
            preprocessing=self._preprocessing,
            device=device.device,
            reporter=NullReporter(),
            diagnostics=diagnostics,
            maps_subdir="maps",
            map_transform=build.transform_for,
            mask_projector=lambda image_id, values: build.transform_for(image_id).project_mask(
                values
            ),
            label_projector=lambda image_id, values: build.transform_for(image_id).project_labels(
                values, fill=0
            ),
        )
        self.generation = resolved.generation

    def answer(self, image_id: int, settings: Settings) -> dict[str, object]:
        with connection(settings.db_path) as conn:
            image = images_repo.get_image(conn, image_id)
            if image is None:
                raise PreviewError(f"no image with id {image_id}")
            sample = samples_repo.get_sample(conn, image.sample_id)
        if sample is None:  # pragma: no cover - an image always has a sample
            raise PreviewError(f"image {image_id} has no sample")
        record = SplitImage(
            image_id=image.id,
            sample_id=image.sample_id,
            channel=None,
            path=image.path,
            sha256=image.sha256,
            label=sample.label,
            subset=Subset.TEST,
        )
        (prediction,) = self._model.predict(to_records([record], self._build), self._infer)
        values = read_map(self._infer.map_path(image_id))
        return {
            "image_id": image_id,
            "score": prediction.score,
            "foreground_share": float(np.nanmean(values >= 0.5)),
            "generation": self.generation,
        }

    def answer_many(self, image_ids: list[int], settings: Settings) -> dict[str, object]:
        """Several images, one request: what orders the studio's rail by uncertainty."""
        if len(image_ids) > MAX_BATCH:
            raise PreviewError(f"at most {MAX_BATCH} images per request, not {len(image_ids)}")
        results = [self.answer(image_id, settings) for image_id in image_ids]
        return {"results": results, "generation": self.generation}
