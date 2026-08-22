"""The resident inference worker, spiked against `pixel_reference` (ADR-0026).

Torch-free, and deliberately so: the baseline emits a per-image `z_map_raw` and imports
neither torch nor a network, so the whole round trip — spawn, request frame on stdin,
response on stdout, index merge, eviction — is provable in CI without the optional extra.

**The eviction path is the reason this file exists.** A resident holds a loaded model and,
on this machine, the MPS device; if one survived into a training run the failure would be
an out-of-memory error in an unrelated job, with nothing on screen connecting the two. It
is tested here rather than assumed, three times: that a job start evicts, that a hook which
*cannot* evict fails the job instead of starting it anyway, and that nothing outlives the
application.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx2
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker, generation_of

from .conftest import Fixture, create_experiment, run_handler

TERMINAL_WAIT_SECONDS = 60.0


def _resident(client: TestClient) -> ResidentWorker:
    app = cast(FastAPI, client.app)
    resident: ResidentWorker = app.state.resident
    return resident


def _diagnose(client: TestClient, experiment_id: int, image_id: int) -> httpx2.Response:
    return client.post(f"/api/experiments/{experiment_id}/diagnose", json={"image_id": image_id})


def _entries(client: TestClient, experiment_id: int) -> list[dict[str, Any]]:
    payload: dict[str, Any] = client.get(f"/api/experiments/{experiment_id}/diagnostics").json()
    entries: list[dict[str, Any]] = payload["entries"]
    return entries


# ------------------------------------------------------------------ the round trip


def test_an_image_outside_the_run_s_sample_can_still_be_asked_about(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """The whole point. A run records a bounded sample; this answers for anything in the split."""
    experiment = create_experiment(client, seeded)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
    run_handler(
        settings,
        JobKind.INFER,
        {"experiment_id": experiment["id"], "subsets": ["test"], "diagnostics": False},
    )
    assert not [e for e in _entries(client, experiment["id"]) if e["scope"] == "image"]

    image_id = seeded.defect_image_ids[0]
    response = _diagnose(client, experiment["id"], image_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["keys"] == ["z_map_raw"]
    assert body["warm"] is False

    recorded = [e for e in _entries(client, experiment["id"]) if e["image_id"] == image_id]
    assert [e["origin"] for e in recorded] == ["on_demand"]
    assert (
        settings.experiment_dir(experiment["id"])
        / "diagnostics"
        / "on-demand"
        / f"image-{image_id}"
        / "z_map_raw.npy"
    ).is_file()


def test_the_second_request_is_warm(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    """What a resident buys, and the only observable difference from a job per request."""
    first = _diagnose(client, scored["id"], seeded.defect_image_ids[0]).json()
    second = _diagnose(client, scored["id"], seeded.normal_image_ids[0]).json()

    assert first["warm"] is False
    assert second["warm"] is True
    assert _resident(client).snapshot() is not None


def test_a_request_leaves_the_stored_map_and_the_scores_alone(
    client: TestClient, settings: Settings, scored: dict[str, Any], seeded: Fixture
) -> None:
    """`predict` writes a map whether or not anybody wants one, which is why `maps_subdir` exists.

    Without it a browse request would replace a stored map under a `range.json` fitted by
    a different run, leaving that image's map a generation ahead of its own score row with
    nothing saying so.
    """
    image_id = seeded.defect_image_ids[0]
    artifact_dir = settings.experiment_dir(scored["id"])
    before = (artifact_dir / "maps" / f"{image_id}.npy").read_bytes()
    results = client.get(f"/api/experiments/{scored['id']}/results", params={"subset": "test"})

    _diagnose(client, scored["id"], image_id)

    assert (artifact_dir / "maps" / f"{image_id}.npy").read_bytes() == before
    assert not (artifact_dir / "scratch-maps").exists()
    after = client.get(f"/api/experiments/{scored['id']}/results", params={"subset": "test"})
    assert after.json() == results.json()


def test_asking_again_replaces_its_own_answer_and_leaves_the_run_s(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    """Both directions of handbook diagnostics.md's rule, on a real request rather than a writer.

    `scored` already sampled this image, so a run entry is there to be preserved — which
    is exactly the case a first draft of this test got wrong by counting entries without
    looking at their origin.
    """
    image_id = seeded.defect_image_ids[0]
    for _ in range(3):
        assert _diagnose(client, scored["id"], image_id).status_code == 200

    recorded = [e for e in _entries(client, scored["id"]) if e["image_id"] == image_id]
    origins = sorted(entry["origin"] for entry in recorded)
    assert origins == ["on_demand", "run"]


# ------------------------------------------------------------------ refusals


def test_an_image_of_another_split_is_refused_by_name(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """Not merely "no such image": one from another dataset exists and is still the wrong ask."""
    response = _diagnose(client, scored["id"], 999_999)

    assert response.status_code == 404
    assert "split" in response.text


def test_an_untrained_experiment_is_refused_as_a_request(
    client: TestClient, seeded: Fixture
) -> None:
    """422 now, rather than a 503 once a process has been spawned to discover it."""
    experiment = create_experiment(client, seeded)

    response = _diagnose(client, experiment["id"], seeded.defect_image_ids[0])

    assert response.status_code == 422
    assert "train it" in response.text
    assert _resident(client).snapshot() is None


def test_a_request_while_a_job_runs_is_refused_and_names_it(
    client: TestClient, settings: Settings, scored: dict[str, Any], seeded: Fixture
) -> None:
    """A browse request must not queue behind a two-hour train."""
    app = cast(FastAPI, client.app)
    queue: JobQueue = app.state.job_queue
    job = queue.enqueue(kind=JobKind.INFER, params={"experiment_id": scored["id"]})
    # The row is what the refusal reads. Marking it running is far more reliable than
    # racing a real worker in order to observe the application mid-flight.
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")

    response = _diagnose(client, scored["id"], seeded.defect_image_ids[0])

    assert response.status_code == 409
    assert "infer" in response.text
    assert str(job.id) in response.text


def test_clearing_diagnostics_is_refused_while_a_job_runs(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """An infer job's flush merges with disk, so a delete landing mid-run is undone."""
    app = cast(FastAPI, client.app)
    queue: JobQueue = app.state.job_queue
    job = queue.enqueue(kind=JobKind.INFER, params={"experiment_id": scored["id"]})
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")

    assert client.delete(f"/api/experiments/{scored['id']}/diagnostics").status_code == 409


# ------------------------------------------------------- staleness and eviction


def test_a_retrain_changes_the_fingerprint(tmp_path: Path) -> None:
    """What makes serving stale weights impossible rather than merely unlikely."""
    model_dir = tmp_path / MODEL_SUBDIR
    model_dir.mkdir()
    (model_dir / "model.npz").write_bytes(b"first")
    first = generation_of(model_dir)

    (model_dir / "model.npz").write_bytes(b"a second checkpoint, of a different size")

    assert generation_of(model_dir) != first
    assert generation_of(tmp_path / "absent") == "none"


def test_a_new_checkpoint_replaces_the_resident_rather_than_being_served_by_it(
    client: TestClient, settings: Settings, scored: dict[str, Any], seeded: Fixture
) -> None:
    image_id = seeded.defect_image_ids[0]
    assert _diagnose(client, scored["id"], image_id).json()["warm"] is False
    assert _diagnose(client, scored["id"], image_id).json()["warm"] is True

    run_handler(settings, JobKind.TRAIN, {"experiment_id": scored["id"]})

    # Cold again: what it had loaded is no longer what is on disk.
    assert _diagnose(client, scored["id"], image_id).json()["warm"] is False


def test_starting_a_job_evicts_the_resident(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    """The guarantee. One machine, one device — a resident must not still be holding it.

    Driven through the queue rather than by calling `evict`, because what is under test is
    that the queue *awaits the hook before it spawns*, not that the hook works when called.
    """
    _diagnose(client, scored["id"], seeded.defect_image_ids[0])
    assert _resident(client).snapshot() is not None

    response = client.post(
        f"/api/experiments/{scored['id']}/infer",
        json={"experiment_id": scored["id"], "subsets": ["test"]},
    )
    assert response.status_code == 200
    _await_terminal(client, response.json()["id"])

    assert _resident(client).snapshot() is None


def test_a_hook_that_cannot_evict_fails_the_job_instead_of_starting_it(
    settings: Settings,
) -> None:
    """ "Freeing the device failed, so we trained on top of it anyway" is the bad outcome."""

    apply_migrations(settings.db_path)

    async def scenario() -> str | None:
        async def refuses() -> None:
            msg = "the device could not be freed"
            raise RuntimeError(msg)

        queue = JobQueue(settings, before_spawn=refuses)
        await queue.start()
        try:
            job = queue.enqueue(kind=JobKind.PREWARM, params={})
            return await _await_job_error(settings, job.id)
        finally:
            await queue.stop()

    assert "could not be freed" in (asyncio.run(scenario()) or "")


def test_the_resident_does_not_outlive_the_application(
    settings: Settings, seeded: Fixture, client: TestClient
) -> None:
    """The sidecar is a child of the desktop shell; an orphan here is an orphan there.

    Driven by a second application's lifespan rather than by calling `stop` directly: the
    shutdown path is the thing under test, and a resident spawned on one event loop cannot
    be torn down from another — which is how a first draft of this test failed, for a
    reason that would equally be a bug if production code did it.
    """
    experiment = create_experiment(client, seeded)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})

    with TestClient(create_app(settings)) as own:
        _diagnose(own, experiment["id"], seeded.defect_image_ids[0])
        process = _resident(own)._process
        assert process is not None
        pid = process.pid
        assert _alive(pid)

    assert not _alive(pid)


# ------------------------------------------------------------------ helpers


def _await_terminal(client: TestClient, job_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    msg = f"job {job_id} did not finish within {TERMINAL_WAIT_SECONDS}s"
    raise AssertionError(msg)


async def _await_job_error(settings: Settings, job_id: int) -> str | None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TERMINAL_WAIT_SECONDS
    while loop.time() < deadline:
        with connection(settings.db_path) as conn:
            job = jobs_repo.get_job(conn, job_id)
        if job is not None and job.status.is_terminal:
            return job.error
        await asyncio.sleep(0.05)
    msg = f"job {job_id} did not finish within {TERMINAL_WAIT_SECONDS}s"
    raise AssertionError(msg)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
