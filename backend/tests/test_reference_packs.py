"""Local public benchmark discovery and one-action registration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotations.class_truth import load_boxes, resolve_box_truth
from anomaly_lab.annotations.imported_boxes import write_box_truth
from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings
from anomaly_lab.datasets.commit import commit_manifest
from anomaly_lab.datasets.reference_packs import (
    FSS_PANEL,
    PCB_DEFECTS,
    pack_specs,
    register_box_truth,
    scan_spec,
)
from anomaly_lab.datasets.splitting import plan_class_stratified_split, plan_few_shot_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.entities import ClassPresence, Subset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path)


def _wait(client: TestClient, job_id: int) -> dict[str, Any]:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return dict(job)
        time.sleep(0.02)
    raise AssertionError("reference registration did not finish")


def test_absent_packs_are_instructional_and_gkn_registers_in_one_action(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references"
    gkn = references / "GKN Blade Surface Defect Dataset" / "Data_GKN"
    _write_image(gkn / "Good" / "good.png")
    _write_image(gkn / "Nick" / "nick.png")
    _write_image(gkn / "Scratch" / "scratch.png")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        catalog = client.get("/api/reference-packs").json()
        assert [(pack["key"], pack["status"]) for pack in catalog["packs"]] == [
            ("visa", "absent"),
            ("gkn", "available"),
            ("fss1000", "absent"),
            ("pku_pcb", "absent"),
        ]
        assert catalog["pending_datasets"] == 1
        assert catalog["packs"][0]["install_url"].startswith("https://")

        started = client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]})
        assert started.status_code == 200, started.text
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["registered"] == 1

        datasets = client.get("/api/datasets").json()
        assert [(dataset["name"], dataset["samples"]) for dataset in datasets] == [
            ("GKN Blade Surface Defect", 3)
        ]
        assert datasets[0]["label_counts"] == {
            "normal": 1,
            "defect": 2,
            "unlabeled": 0,
        }

        after = client.get("/api/reference-packs").json()
        assert after["packs"][1]["status"] == "registered"
        assert after["pending_datasets"] == 0
        assert (
            client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]}).status_code
            == 409
        )


def test_a_registered_dataset_inherits_its_pack_collection_and_blurb(tmp_path: Path) -> None:
    """Membership is derived, so it holds for datasets registered before the column existed."""
    references = tmp_path / "references"
    gkn = references / "GKN Blade Surface Defect Dataset" / "Data_GKN"
    _write_image(gkn / "Good" / "good.png")
    _write_image(gkn / "Nick" / "nick.png")
    _write_image(gkn / "Scratch" / "scratch.png")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        started = client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]})
        _wait(client, started.json()["id"])

        registered = client.get("/api/datasets").json()[0]
        # Nothing was written at registration; both values come from the pack spec.
        assert registered["notes"] is None
        assert registered["collection"] == "GKN"
        assert "turbine blades" in registered["description"]

        dataset_id = registered["id"]
        moved = client.patch(
            f"/api/datasets/{dataset_id}",
            json={"collection": "Blades", "notes": "My working copy."},
        ).json()
        assert (moved["collection"], moved["description"]) == ("Blades", "My working copy.")

        # Clearing the override falls back to the pack rather than to nothing.
        restored = client.patch(
            f"/api/datasets/{dataset_id}", json={"collection": None, "notes": None}
        ).json()
        assert restored["collection"] == "GKN"
        assert "turbine blades" in restored["description"]


def test_a_dataset_that_merely_shares_a_name_inherits_nothing(tmp_path: Path) -> None:
    """The matcher guards this; a user's own `candle` must not be filed under VisA."""
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=tmp_path / "references")
    with TestClient(create_app(settings)) as client:
        with connection(settings.db_path) as conn:
            datasets_repo.create_dataset(
                conn, name="candle", root_path=str(tmp_path / "mine"), adapter="csv_table"
            )

        mine = client.get("/api/datasets").json()[0]
        assert mine["collection"] is None
        assert mine["description"] is None


def test_an_incomplete_pack_is_not_offered_for_registration(tmp_path: Path) -> None:
    references = tmp_path / "references"
    (references / "VisA_20220922").mkdir(parents=True)
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)
    with TestClient(create_app(settings)) as client:
        visa = client.get("/api/reference-packs").json()["packs"][0]
        assert visa["status"] == "incomplete"
        assert visa["missing"]
        response = client.post("/api/reference-packs/register", json={"pack_keys": ["visa"]})
        assert response.status_code == 409


def _write_fss_pair(directory: Path, index: int, shade: int) -> None:
    """One FSS-1000-shaped pair: `<n>.jpg` and the `<n>.png` mask of the object in it."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (shade, index * 40, 90)).save(directory / f"{index}.jpg")
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (2, 2, 6, 6))
    mask.save(directory / f"{index}.png")


def test_fss1000_registers_one_few_shot_dataset_per_panel_class(tmp_path: Path) -> None:
    """A target's own masks are its truth; the rest of the panel are confirmed absences."""
    references = tmp_path / "references"
    classes = references / "FSS-1000" / "fewshot_data"
    for position, name in enumerate(FSS_PANEL):
        for index in (1, 2):
            _write_fss_pair(classes / name, index, shade=position * 12)
    # Outside the panel, and a stray `.jpeg` beside a paired `.jpg`: neither is imported.
    _write_fss_pair(classes / "not_in_panel", 1, shade=250)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(classes / FSS_PANEL[0] / "1.jpeg")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        fss = client.get("/api/reference-packs").json()["packs"][2]
        assert (fss["key"], fss["status"], len(fss["datasets"])) == ("fss1000", "available", 20)

        started = client.post("/api/reference-packs/register", json={"pack_keys": ["fss1000"]})
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["registered"] == 20

        datasets = {dataset["name"]: dataset for dataset in client.get("/api/datasets").json()}
        target = datasets[FSS_PANEL[0]]
        assert target["samples"] == 2 * 20
        assert target["label_counts"] == {"normal": 38, "defect": 2, "unlabeled": 0}
        assert target["collection"] == "FSS-1000"

    with connection(settings.db_path) as conn:
        presence = annotations_repo.class_presence(conn, target["id"], "defect")
        assert sum(found is ClassPresence.PRESENT for found in presence.values()) == 2
        assert sum(found is ClassPresence.ABSENT for found in presence.values()) == 38
        split = plan_few_shot_split(conn, target["id"], seed=0, label_key="defect", shots=1)
    references_drawn = [sample for sample, subset in split.items() if subset is Subset.TRAIN]
    assert len(references_drawn) == 1
    assert presence[references_drawn[0]] is ClassPresence.PRESENT


def _write_voc(
    path: Path, size: tuple[int, int], objects: list[tuple[str, int, int, int, int]]
) -> None:
    """A Pascal VOC file: 1-based, inclusive corners, as PKU-Market-PCB's are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f"<object><name>{name}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
        for name, x0, y0, x1, y1 in objects
    )
    path.write_text(
        f"<annotation><size><width>{size[0]}</width><height>{size[1]}</height>"
        f"<depth>3</depth></size>{body}</annotation>",
        encoding="utf-8",
    )


def _pcb_tree(references: Path) -> Path:
    """Two boards of every defect kind, each with one box of its kind, and the two trees the
    pack leaves out: rotated copies and the defect-free templates."""
    root = references / "PKU-PCB" / "PCB_DATASET"
    for position, (directory, name) in enumerate(PCB_DEFECTS):
        for index in (1, 2):
            stem = f"{index:02d}_{name}_01"
            image = root / "images" / directory / f"{stem}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 12), (position * 30, index * 60, 90)).save(image)
            _write_voc(
                root / "Annotations" / directory / f"{stem}.xml",
                (16, 12),
                [(name, 3, 2, 6 + index, 5)],
            )
        _write_image(root / "rotation" / f"{directory}_rotation" / "01.jpg")
    _write_image(root / "PCB_USED" / "01.jpg")
    return root


def test_pku_pcb_registers_its_voc_boxes_as_class_truth(tmp_path: Path) -> None:
    """Every box is an instance of its class, and every image answers every class."""
    references = tmp_path / "references"
    _pcb_tree(references)
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        pcb = client.get("/api/reference-packs").json()["packs"][3]
        assert (pcb["key"], pcb["status"]) == ("pku_pcb", "available")
        started = client.post("/api/reference-packs/register", json={"pack_keys": ["pku_pcb"]})
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        entered = job["result"]["box_truth"]["pku_pcb:pcb"]
        assert (entered["images"], entered["boxes"], entered["kept"]) == (12, 12, 0)
        assert (entered["clipped"], entered["reshaped"], entered["without_file"]) == (0, 0, 0)
        dataset = client.get("/api/datasets").json()[0]
        assert dataset["samples"] == 12
        assert dataset["label_counts"] == {"normal": 0, "defect": 12, "unlabeled": 0}
        assert (
            client.post("/api/reference-packs/register", json={"pack_keys": ["pku_pcb"]})
        ).status_code == 409

    kinds = [name for _, name in PCB_DEFECTS]
    with connection(settings.db_path) as conn:
        labels = [label.key for label in annotations_repo.list_labels(conn, dataset["id"])]
        images = images_repo.list_images_for_dataset(conn, dataset["id"])
        truth = resolve_box_truth(conn, dataset["id"], [image.id for image in images], labels)
        split = plan_class_stratified_split(
            conn, dataset["id"], seed=0, classes=labels, train_fraction=0.5, unlabeled_subset=None
        )
    assert labels == ["defect", *kinds]
    assert len(truth) == 12
    found = sorted((box.label_key, box.box) for item in truth.values() for box in load_boxes(item))
    # VOC's (3, 2)-(7, 5), 1-based and inclusive, is the pixel-edge box (2, 1)-(7, 5).
    assert found == sorted(
        (name, (2.0, 1.0, 6.0 + index, 5.0)) for name in kinds for index in (1, 2)
    )
    # One board of each kind trains and one tests: every class signature is split.
    assert list(split.values()).count(Subset.TRAIN) == 6


def test_pku_pcb_box_truth_keeps_what_an_image_already_has(tmp_path: Path) -> None:
    """A revision completed before the box truth is entered is not overwritten, and is counted."""
    references = tmp_path / "references"
    _pcb_tree(references)
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)
    apply_schema(settings.db_path)
    spec = next(pack for pack in pack_specs(settings) if pack.key == "pku_pcb").datasets[0]
    with connection(settings.db_path) as conn:
        dataset_id = commit_manifest(conn, settings, scan_spec(spec, lambda *_: None)).dataset_id
        first = images_repo.list_images_for_dataset(conn, dataset_id)[0]
    assert write_box_truth(settings, first.id, first.width, first.height, []).written

    # Registered, with its box truth unfinished: still pending, and registration finishes it.
    with TestClient(create_app(settings)) as client:
        catalog = client.get("/api/reference-packs").json()
        pcb = catalog["packs"][3]
        assert (pcb["status"], catalog["pending_datasets"]) == ("available", 1)
        assert pcb["datasets"][0]["registered_dataset_id"] == dataset_id
        started = client.post("/api/reference-packs/register", json={"pack_keys": ["pku_pcb"]})
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["registered"] == 0
        entered = job["result"]["box_truth"]["pku_pcb:pcb"]
        assert (entered["images"], entered["kept"]) == (11, 1)
        assert client.get("/api/reference-packs").json()["packs"][3]["status"] == "registered"
    with connection(settings.db_path) as conn:
        revision = annotations_repo.latest_revision(conn, first.id)
    assert revision is not None
    assert revision.document.shapes == []


def test_pku_pcb_refuses_a_class_its_taxonomy_does_not_name(tmp_path: Path) -> None:
    """Every file is read before anything is written, so a bad one changes nothing."""
    references = tmp_path / "references"
    root = _pcb_tree(references)
    directory, name = PCB_DEFECTS[-1]
    _write_voc(
        root / "Annotations" / directory / f"02_{name}_01.xml", (16, 12), [("scratch", 1, 1, 2, 2)]
    )
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)
    apply_schema(settings.db_path)
    spec = next(pack for pack in pack_specs(settings) if pack.key == "pku_pcb").datasets[0]
    with connection(settings.db_path) as conn:
        dataset_id = commit_manifest(conn, settings, scan_spec(spec, lambda *_: None)).dataset_id

    with pytest.raises(ValueError, match="scratch"):
        register_box_truth(settings, spec, dataset_id)
    with connection(settings.db_path) as conn:
        labels = annotations_repo.list_labels(conn, dataset_id)
        assert [label.key for label in labels] == ["defect"]
        images = images_repo.list_images_for_dataset(conn, dataset_id)
        assert all(annotations_repo.latest_revision(conn, image.id) is None for image in images)
