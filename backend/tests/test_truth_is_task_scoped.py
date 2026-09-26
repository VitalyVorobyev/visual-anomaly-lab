"""Truth is task-scoped (ADR-0041): class truth reads the same over one multi-class dataset as it
did over one dataset per class, and a dataset reports which truth it holds.

The per-class-panel shape made a target's images `defect` samples with imported masks and every
other class's images `normal` samples. The one-dataset shape leaves every sample unlabelled and
enters each image's mask as a region of its own class. A few-shot run over either must draw the
same references, see the same queries, and measure the same numbers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotations.imported_truth import ImportedClass
from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings
from anomaly_lab.datasets.adapters.folder_classes import (
    FolderClassesAdapter,
    FolderClassesOptions,
)
from anomaly_lab.datasets.commit import commit_manifest
from anomaly_lab.datasets.reference_packs import (
    DatasetSpec,
    MaskTruthSpec,
    register_class_truth,
)
from anomaly_lab.datasets.splitting import plan_few_shot_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import ImageResult, Subset, Task
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.eval.runner import rebuild_sample_results
from anomaly_lab.eval.segmentation import sample_outcomes

CLASSES = ("alpha", "beta", "gamma")
IMAGES_PER_CLASS = 4
SIZE = 12


def _tree(root: Path) -> Path:
    """Three classes of four 12 x 12 images, each with a mask of a square of its own place."""
    for position, name in enumerate(CLASSES):
        directory = root / name
        directory.mkdir(parents=True)
        for index in range(1, IMAGES_PER_CLASS + 1):
            Image.new("RGB", (SIZE, SIZE), (position * 60, index * 40, 90)).save(
                directory / f"{index}.jpg"
            )
            mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            corner = (position + index) % 5
            mask[corner : corner + 4 + position, corner + 1 : corner + 5] = 255
            Image.fromarray(mask).save(directory / f"{index}.png")
    return root


def _commit(settings: Settings, root: Path, name: str, options: dict[str, Any]) -> int:
    manifest = FolderClassesAdapter.scan(
        root, FolderClassesOptions.model_validate(options), dataset_name=name
    )
    manifest = manifest.model_copy(update={"root_path": str(root / f".{name}")})
    with connection(settings.db_path) as conn:
        return commit_manifest(conn, settings, manifest).dataset_id


def _panel_dataset(settings: Settings, root: Path, target: str) -> int:
    """The retired shape: the target's images are defects with masks, the rest normals."""
    return _commit(
        settings,
        root,
        f"panel-{target}",
        {
            "defect_dirs": [target],
            "normal_dirs": [other for other in CLASSES if other != target],
            "mask_dir": "{dir}",
            "masks_for_normal_dirs": False,
            "import_unnamed_dirs": False,
            "extensions": [".jpg"],
        },
    )


def _one_dataset(settings: Settings, root: Path) -> int:
    """The shape a class pack registers as: unlabelled samples, masks as class truth."""
    dataset_id = _commit(
        settings,
        root,
        "whole",
        {"unlabeled_dirs": list(CLASSES), "import_unnamed_dirs": False, "extensions": [".jpg"]},
    )
    spec = DatasetSpec(
        key="test:whole",
        name="whole",
        root=root,
        scan_root=root,
        adapter="folder_classes",
        options={},
        class_truth=MaskTruthSpec(
            pattern="{class}/{stem}.png",
            classes=tuple((name, ImportedClass(key=name, name=name)) for name in CLASSES),
        ),
    )
    entered = register_class_truth(settings, spec, dataset_id)
    assert (entered.images, entered.regions) == (len(CLASSES) * IMAGES_PER_CLASS,) * 2
    return dataset_id


def _prediction(image_path: str) -> np.ndarray:
    """A fixed map per source file, whatever dataset it sits in: its own object, blurred by a
    field seeded from its name, so some absent images are flagged and some present ones missed."""
    truth = np.asarray(Image.open(Path(image_path).with_suffix(".png")).convert("L")) > 0
    seed = int(hashlib.sha256(Path(image_path).as_posix().encode()).hexdigest()[:8], 16)
    noise = np.random.default_rng(seed).random((SIZE, SIZE), dtype=np.float32)
    return np.clip(truth * 0.45 + noise * (0.52 if seed % 2 else 0.45), 0.0, 1.0).astype(np.float32)


def _run(
    settings: Settings, dataset_id: int, label_key: str, seed: int
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Draw a one-shot split, store the fixed predictions for its queries, and evaluate them.

    Returns the test metrics, each query's outcome by file, and the references by file.
    """
    with connection(settings.db_path) as conn:
        assignments = plan_few_shot_split(conn, dataset_id, seed=seed, label_key=label_key, shots=1)
        split = splits_repo.create_split(
            conn,
            dataset_id,
            name=f"{label_key} {seed}",
            strategy="few_shot",
            seed=seed,
            params={"strategy": "few_shot", "label_key": label_key, "shots": 1},
            assignments=assignments,
        )
        profile = conn.execute(
            """
            INSERT INTO region_profile_revision
                   (dataset_id, name, revision_no, extractor_type, prepared_width,
                    prepared_height)
            VALUES (?, ?, 1, 'identity', ?, ?)
            """,
            (dataset_id, f"identity {label_key} {seed}", SIZE, SIZE),
        ).lastrowid
        experiment = experiments_repo.create_experiment(
            conn,
            name="few-shot",
            dataset_id=dataset_id,
            split_id=split.id,
            region_profile_id=int(profile or 0),
            region_manifest_sha256="-",
            model_type="color_prototype",
            task=Task.FEW_SHOT_SEGMENTATION.value,
            target_label=label_key,
            model_config={},
            preprocessing_config={},
            eval_config={},
            artifact_dir=str(settings.experiment_dir(0) / f"{dataset_id}-{label_key}-{seed}"),
        )
        images = images_repo.list_images_for_dataset(conn, dataset_id)
    maps = Path(experiment.artifact_dir) / "maps"
    maps.mkdir(parents=True)
    by_id = {image.id: image for image in images}
    rows = []
    references = []
    for image in images:
        if assignments[image.sample_id] is Subset.TRAIN:
            references.append(_file(image.path))
            continue
        probability = _prediction(image.path)
        np.save(maps / f"{image.id}.npy", probability)
        rows.append(
            ImageResult(
                experiment_id=experiment.id,
                image_id=image.id,
                score=float(probability.mean()),
                map_path=str(maps / f"{image.id}.npy"),
                inference_ms=4.0,
            )
        )
    evaluator = evaluator_for(Task.FEW_SHOT_SEGMENTATION)
    with connection(settings.db_path) as conn:
        results_repo.replace_image_results(conn, experiment.id, rows)
        rebuild_sample_results(conn, experiment)
        metrics = evaluator.evaluate_and_store(conn, experiment)[Subset.TEST]
        verdicts = sample_outcomes(conn, experiment, Subset.TEST).samples
    sample_file = {image.sample_id: _file(image.path) for image in by_id.values()}
    outcomes = {sample_file[verdict.sample_id]: verdict.outcome for verdict in verdicts}
    return metrics, outcomes, references


def _file(path: str) -> str:
    return f"{Path(path).parent.name}/{Path(path).name}"


@pytest.mark.parametrize("target", CLASSES)
def test_one_multi_class_dataset_measures_what_the_per_class_panel_measured(
    tmp_path: Path, target: str
) -> None:
    root = _tree(tmp_path / "tree")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=tmp_path / "none")
    apply_schema(settings.db_path)
    panel = _panel_dataset(settings, root, target)
    whole = _one_dataset(settings, root)

    for seed in (0, 1, 2):
        panel_metrics, panel_outcomes, panel_references = _run(settings, panel, "defect", seed)
        whole_metrics, whole_outcomes, whole_references = _run(settings, whole, target, seed)
        assert whole_references == panel_references
        assert whole_outcomes == panel_outcomes
        assert whole_metrics == panel_metrics
        # Every query answers: the negatives are the other classes' completed annotations.
        assert whole_metrics["images"] == {
            "present": IMAGES_PER_CLASS - 1,
            "absent": (len(CLASSES) - 1) * IMAGES_PER_CLASS,
            "unlabeled": 0,
            "without_prediction": 0,
        }
        assert set(whole_outcomes.values()) - {"unlabeled"}
        assert whole_metrics["image_absent_false_positive_rate"] is not None


def test_a_dataset_reports_the_truth_it_holds(tmp_path: Path) -> None:
    root = _tree(tmp_path / "tree")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=tmp_path / "none")
    apply_schema(settings.db_path)
    panel = _panel_dataset(settings, root, "beta")
    whole = _one_dataset(settings, root)
    bare = _commit(
        settings,
        root,
        "bare",
        {"unlabeled_dirs": list(CLASSES), "import_unnamed_dirs": False, "extensions": [".jpg"]},
    )

    with TestClient(create_app(settings)) as client:
        found = {entry["id"]: entry for entry in client.get("/api/datasets").json()}
        detail = client.get(f"/api/datasets/{whole}").json()
        whole_images = {
            image["id"]: image["path"]
            for sample in client.get(f"/api/datasets/{whole}/samples").json()["items"]
            for image in sample["images"]
        }
        panel_images = {
            image["id"]: (sample["label"], image["path"])
            for sample in client.get(f"/api/datasets/{panel}/samples").json()["items"]
            for image in sample["images"]
        }

    # An anomaly dataset: labels, no class truth (its imported masks are anomaly truth), and a
    # normal sample on its cover.
    assert found[panel]["truth"] == ["labels"]
    assert found[panel]["class_counts"] == []
    assert panel_images[found[panel]["cover_image_id"]][0] == "normal"

    # A class dataset: no labels, every class with the samples that show it, and a cover
    # showing the first of the most frequent classes.
    assert found[whole]["truth"] == ["classes"]
    assert detail["truth"] == ["classes"]
    assert [(entry["key"], entry["samples"]) for entry in found[whole]["class_counts"]] == [
        (name, IMAGES_PER_CLASS) for name in CLASSES
    ]
    assert found[whole]["label_counts"]["unlabeled"] == len(CLASSES) * IMAGES_PER_CLASS
    assert Path(whole_images[found[whole]["cover_image_id"]]).parent.name == "alpha"

    # Neither truth: nothing to report, and the cover is simply the first image.
    assert (found[bare]["truth"], found[bare]["class_counts"]) == ([], [])
    assert found[bare]["cover_image_id"] is not None
