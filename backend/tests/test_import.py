"""Scan, review, commit and verify, end to end over the HTTP API.

The scan cases here run a real worker subprocess, so they also cover the job machinery's
success and progress paths, which `test_job_queue.py` deliberately leaves to a real
handler.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Label, LabelSource
from tests.conftest import write_image

TERMINAL_WAIT_SECONDS = 60.0

# What the fixture tree below calls its three illuminations.
FIXTURE_CHANNELS = ["bright", "dark", "dome"]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small tree in the shape real acquisition folders take.

    Two capture groups under two labels, three channels, plus one deliberately irregular
    group: two channels, and its channel name fused into its directory name.
    """
    root = tmp_path / "source"
    for group in ("set1", "set2"):
        for label in ("defect", "no-defect"):
            for external_id in ("1", "2"):
                for channel in ("Bright", "Dark", "Dome"):
                    write_image(root / group / label / channel / f"{external_id}.png")
    for external_id in ("1", "2"):
        for channel in ("Brightfield", "Darkfield"):
            write_image(root / "unsorted" / f"{channel} Bl7" / f"{external_id}.png")
    return root


def _await_job(client: TestClient, job_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    msg = f"job {job_id} never finished"
    raise AssertionError(msg)


def _scan(client: TestClient, root: Path, **body: Any) -> dict[str, Any]:
    """Scan the fixture tree, supplying the vocabulary its channel directories use.

    The adapter ships none: naming a dataset's channels is the operator's job, not an
    assumption baked into the application (ADR-0005). A caller that scans a channelled
    tree therefore has to say so, and that includes this helper.
    """
    options = {"channels": FIXTURE_CHANNELS, **body.pop("options", {})}
    response = client.post(
        "/api/import/scan",
        json={"root_path": str(root), "dataset_name": "fixture", "options": options, **body},
    )
    assert response.status_code == 200, response.text
    return _await_job(client, response.json()["id"])


def _manifest(client: TestClient, manifest_id: str) -> dict[str, Any]:
    response = client.get(f"/api/import/manifests/{manifest_id}")
    assert response.status_code == 200, response.text
    return dict(response.json())


def _commit(client: TestClient, manifest: dict[str, Any], **body: Any) -> dict[str, Any]:
    response = client.post("/api/import/commit", json={"manifest": manifest, **body})
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- adapters ------------------------------------------------------------------------


def test_adapters_are_listed_with_the_schema_their_form_is_built_from(
    client: TestClient,
) -> None:
    adapters = client.get("/api/import/adapters").json()

    names = {adapter["name"] for adapter in adapters}
    assert "channel_folders" in names
    schema = next(a for a in adapters if a["name"] == "channel_folders")["options_schema"]
    assert {"channels", "exclude", "channel_aliases"} <= set(schema["properties"])


# -- scan ----------------------------------------------------------------------------


def test_a_scan_proposes_without_writing_anything(
    client: TestClient, settings: Settings, tree: Path
) -> None:
    """Stage one of ADR-0006: the database is untouched until commit."""
    job = _scan(client, tree)

    assert job["status"] == "succeeded", job["error"]
    assert job["progress"] == 1.0
    assert job["result"]["samples"] == 10
    assert job["result"]["images"] == 28

    with connection(settings.db_path) as conn:
        assert datasets_repo.list_datasets(conn) == []


def test_the_manifest_is_readable_back_for_review(client: TestClient, tree: Path) -> None:
    job = _scan(client, tree)

    manifest = _manifest(client, job["result"]["manifest_id"])

    assert manifest["manifest_version"] == 1
    assert manifest["channels"] == ["bright", "dark", "dome"]
    assert manifest["root_path"] == str(tree)
    # The mapping table the operator can correct before committing.
    assert {row["source"] for row in manifest["channel_mapping"]} >= {"Bright", "Dark", "Dome"}


def test_the_irregular_group_is_grouped_and_warned_about_not_dropped(
    client: TestClient, tree: Path
) -> None:
    job = _scan(client, tree)
    manifest = _manifest(client, job["result"]["manifest_id"])

    two_channel = [s for s in manifest["samples"] if len(s["images"]) == 2]
    codes = {warning["code"] for warning in manifest["warnings"]}

    assert len(two_channel) == 2
    assert {s["group_key"] for s in two_channel} == {"unsorted/Bl7"}
    assert "variable_channel_count" in codes


def test_exclude_narrows_the_scan(client: TestClient, tree: Path) -> None:
    job = _scan(client, tree, options={"exclude": ["unsorted/*"]})
    manifest = _manifest(client, job["result"]["manifest_id"])

    assert job["result"]["samples"] == 8
    assert job["result"]["files_excluded"] == 4
    # Nothing *structural* is left to report: excluding `unsorted/` takes the two-channel
    # group with it. Asserted by code rather than as "no warnings at all", because the scan
    # also measures the pixels, and what those measurements say about the fixture images is
    # true whichever subtree was walked.
    codes = {warning["code"] for warning in manifest["warnings"]}
    assert "variable_channel_count" not in codes
    assert "unassigned_channel" not in codes
    assert "unknown_channel_name" not in codes


def test_one_tree_becomes_two_datasets_when_each_records_its_own_root(
    client: TestClient, tree: Path
) -> None:
    """A capture tree can hold several products, and each is its own dataset.

    `root_path` is `UNIQUE` and is what a re-import resolves against (ADR-0013), so two
    scans of one tree would otherwise collide into a single dataset holding both — which
    trains a normal-only method on two different parts and calls the mixture a baseline.
    """
    for name, keep, drop in (("set-one", "set1", "set2"), ("set-two", "set2", "set1")):
        job = _scan(
            client,
            tree,
            dataset_name=name,
            dataset_root=str(tree / keep),
            options={"exclude": [f"{drop}/*", "unsorted/*"]},
        )
        manifest = _manifest(client, job["result"]["manifest_id"])
        assert manifest["root_path"] == str(tree / keep)
        assert {sample["group_key"] for sample in manifest["samples"]} == {
            f"{keep}/defect",
            f"{keep}/no-defect",
        }
        committed = _commit(client, manifest)
        assert committed["dataset_created"] is True
        # Re-committing resolves by that recorded root, so a re-import still updates the
        # dataset it produced rather than making a third one.
        assert _commit(client, manifest)["dataset_id"] == committed["dataset_id"]

    with connection(client.app.state.settings.db_path) as conn:  # type: ignore[attr-defined]
        datasets = datasets_repo.list_datasets(conn)

    assert [dataset.name for dataset in datasets] == ["set-one", "set-two"]
    assert [Path(dataset.root_path).name for dataset in datasets] == ["set1", "set2"]


def test_a_dataset_root_outside_the_scan_root_is_refused(
    client: TestClient, tree: Path, tmp_path: Path
) -> None:
    """An identity has to name somewhere the images actually came from."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    response = client.post(
        "/api/import/scan",
        json={
            "root_path": str(tree),
            "dataset_name": "borrowed",
            "dataset_root": str(elsewhere),
        },
    )

    assert response.status_code == 422
    assert "not inside the scan root" in response.json()["detail"]


def test_scanning_a_directory_that_is_not_there_fails_before_a_job_is_made(
    client: TestClient, tmp_path: Path
) -> None:
    """An obvious mistake should be a message, not a job that starts and dies."""
    response = client.post(
        "/api/import/scan",
        json={"root_path": str(tmp_path / "nope"), "dataset_name": "d"},
    )

    assert response.status_code == 400
    assert client.get("/api/jobs").json() == []


def test_an_unknown_adapter_is_rejected_with_the_ones_that_exist(
    client: TestClient, tree: Path
) -> None:
    response = client.post(
        "/api/import/scan",
        json={"root_path": str(tree), "dataset_name": "d", "adapter": "nope"},
    )

    assert response.status_code == 400
    assert "channel_folders" in response.json()["detail"]


def test_a_manifest_id_cannot_escape_the_manifest_directory(client: TestClient) -> None:
    """Ids arrive in a URL path segment, so they are constrained, not merely checked."""
    assert client.get("/api/import/manifests/..%2F..%2Fapp.sqlite3").status_code in {404, 400}
    assert client.get("/api/import/manifests/nope").status_code == 404


# -- commit --------------------------------------------------------------------------


def test_commit_creates_the_catalog(client: TestClient, settings: Settings, tree: Path) -> None:
    job = _scan(client, tree)
    manifest = _manifest(client, job["result"]["manifest_id"])

    result = _commit(client, manifest)

    assert result["dataset_created"] is True
    assert result["samples_created"] == 10
    assert result["images_created"] == 28
    assert result["channels"] == 3
    assert result["missing_paths"] == []

    with connection(settings.db_path) as conn:
        dataset_id = result["dataset_id"]
        assert datasets_repo.label_counts(conn, dataset_id) == {
            Label.NORMAL: 4,
            Label.DEFECT: 4,
            Label.UNLABELED: 2,
        }
        assert [c.name for c in datasets_repo.list_channels(conn, dataset_id)] == [
            "bright",
            "dark",
            "dome",
        ]
        # The two-channel sample is an ordinary row, not a special case.
        two_channel = samples_repo.find_sample(
            conn, dataset_id, group_key="unsorted/Bl7", external_id="1"
        )
        assert two_channel is not None
        assert len(images_repo.list_images_for_sample(conn, two_channel.id)) == 2


def test_re_importing_the_same_tree_is_idempotent(
    client: TestClient, settings: Settings, tree: Path
) -> None:
    """M2 exit criterion: no duplicate samples on a second import."""
    first = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))
    second = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))

    assert second["dataset_id"] == first["dataset_id"]
    assert second["dataset_created"] is False
    assert second["samples_created"] == 0
    assert second["images_created"] == 0
    assert second["samples_updated"] == 10
    assert second["images_updated"] == 28

    with connection(settings.db_path) as conn:
        assert samples_repo.count_samples(conn, first["dataset_id"]) == 10
        assert datasets_repo.count_images(conn, first["dataset_id"]) == 28
        assert len(datasets_repo.list_datasets(conn)) == 1


def test_a_hand_corrected_label_survives_re_import(
    client: TestClient, settings: Settings, tree: Path
) -> None:
    """The whole point of `label_source` (ADR-0013)."""
    first = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))

    with connection(settings.db_path) as conn:
        sample = samples_repo.find_sample(
            conn, first["dataset_id"], group_key="set1/no-defect", external_id="1"
        )
        assert sample is not None
        samples_repo.set_label(conn, sample.id, Label.DEFECT)

    _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))

    with connection(settings.db_path) as conn:
        after = samples_repo.get_sample(conn, sample.id)
    assert after is not None
    assert after.label is Label.DEFECT
    assert after.label_source is LabelSource.MANUAL


def test_a_file_that_vanished_is_reported_never_deleted(
    client: TestClient, settings: Settings, tree: Path
) -> None:
    """An unmounted disk must not become data loss."""
    first = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))
    removed = tree / "set1" / "defect" / "Dome" / "1.png"
    removed.unlink()

    second = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))

    assert second["missing_paths"] == [str(removed)]
    with connection(settings.db_path) as conn:
        assert datasets_repo.count_images(conn, first["dataset_id"]) == 28


def test_an_edited_channel_mapping_is_what_gets_committed(
    client: TestClient, settings: Settings, tree: Path
) -> None:
    """Review is not decoration: what the operator accepts is what is written."""
    manifest = _manifest(
        client, _scan(client, tree, options={"exclude": ["unsorted/*"]})["result"]["manifest_id"]
    )
    manifest["channels"] = ["bright", "dark"]

    result = _commit(client, manifest)

    with connection(settings.db_path) as conn:
        names = [c.name for c in datasets_repo.list_channels(conn, result["dataset_id"])]
        dome_images = [
            image
            for image in images_repo.list_images_for_dataset(conn, result["dataset_id"])
            if "Dome" in image.path
        ]

    assert names == ["bright", "dark"]
    # A name the accepted dictionary does not contain means no channel, not a new one.
    assert dome_images and all(image.channel_id is None for image in dome_images)


def test_committing_into_a_dataset_that_does_not_exist_is_a_conflict(
    client: TestClient, tree: Path
) -> None:
    manifest = _manifest(client, _scan(client, tree)["result"]["manifest_id"])

    response = client.post("/api/import/commit", json={"manifest": manifest, "dataset_id": 4242})

    assert response.status_code == 409
    assert "4242" in response.json()["detail"]


def test_a_second_tree_cannot_reuse_a_dataset_name(
    client: TestClient, tree: Path, tmp_path: Path
) -> None:
    _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))
    other = tmp_path / "other"
    write_image(other / "g" / "Bright" / "1.png")

    manifest = _manifest(client, _scan(client, other)["result"]["manifest_id"])
    response = client.post("/api/import/commit", json={"manifest": manifest})

    assert response.status_code == 409
    assert "already named" in response.json()["detail"]


# -- verify --------------------------------------------------------------------------


def test_verify_confirms_an_untouched_dataset(client: TestClient, tree: Path) -> None:
    result = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))

    response = client.post("/api/import/verify", json={"dataset_id": result["dataset_id"]})
    job = _await_job(client, response.json()["id"])

    assert job["status"] == "succeeded"
    assert job["result"]["checked"] == 28
    assert job["result"]["verified"] == 28
    assert job["result"]["missing_count"] == 0


def test_verify_detects_a_moved_and_a_rewritten_file(client: TestClient, tree: Path) -> None:
    result = _commit(client, _manifest(client, _scan(client, tree)["result"]["manifest_id"]))
    (tree / "set1" / "defect" / "Bright" / "1.png").unlink()
    write_image(tree / "set1" / "defect" / "Dark" / "1.png", colour=7)

    response = client.post("/api/import/verify", json={"dataset_id": result["dataset_id"]})
    job = _await_job(client, response.json()["id"])

    assert job["result"]["missing_count"] == 1
    assert job["result"]["modified_count"] == 1
    assert job["result"]["verified"] == 26


def test_verifying_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    assert client.post("/api/import/verify", json={"dataset_id": 4242}).status_code == 404


# -- cancellation ---------------------------------------------------------------------


def test_a_running_scan_can_be_cancelled(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """Cancelling a running scan stops it and records `cancelled`, not `failed`.

    What this proves is the parent's half: the signal reaches the worker's process group
    and the job ends in the right terminal state with no partial dataset written. Whether
    the worker got far enough to unwind cooperatively or was still importing modules is
    deliberately not asserted — a scan of a small tree can finish starting up and finish
    working inside the same few milliseconds. The worker's own side of cancellation is
    tested deterministically in `test_worker.py`.
    """
    root = tmp_path / "big"
    for index in range(400):
        write_image(root / "g" / "Bright" / f"{index}.png")

    started = client.post(
        "/api/import/scan", json={"root_path": str(root), "dataset_name": "big"}
    ).json()
    job_id = started["id"]

    # `running` means a worker process exists, so the cancel below has something to
    # signal rather than racing the spawn.
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
            break
        time.sleep(0.02)
    else:  # pragma: no cover - only on a pathologically slow machine
        pytest.fail("the scan never reached `running`")

    client.post(f"/api/jobs/{job_id}/cancel")
    finished = _await_job(client, job_id)

    assert finished["status"] == "cancelled"
    assert finished["result"] == {}
    assert finished["finished_at"] is not None
    # Nothing was half-written: a cancelled scan leaves no manifest to commit and no rows.
    assert not list(settings.manifests_dir.glob("*.json"))
    with connection(settings.db_path) as conn:
        assert datasets_repo.list_datasets(conn) == []
