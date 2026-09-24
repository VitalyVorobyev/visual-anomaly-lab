"""A targeted task's ground truth, handed to `fit` in the frame the model sees (ADR-0040).

The plugin boundary allows ground truth through `TrainContext.targets` alone. This is the
one implementation: it reads each reference's class region from what pinned it, verified,
and applies the image's pinned region transform so the mask lines up with the prepared
image the model was given.
"""

from __future__ import annotations

import numpy as np

from anomaly_lab.annotations.class_truth import ClassTruth, ClassTruthError, load_class_mask
from anomaly_lab.regions.preparation import PreparedRegionBuild


class PreparedClassTargets:
    """`TargetProvider` over the references of one split, for one class."""

    def __init__(
        self, label_key: str, truths: dict[int, ClassTruth], region_build: PreparedRegionBuild
    ) -> None:
        self._label_key = label_key
        self._truths = truths
        self._region_build = region_build

    @property
    def label_key(self) -> str:
        return self._label_key

    def mask(self, image_id: int) -> np.ndarray:
        truth = self._truths.get(image_id)
        if truth is None:
            raise ClassTruthError(f"image {image_id} is not a reference of this run")
        source = load_class_mask(truth)
        return self._region_build.transform_for(image_id).prepare_mask(source)
