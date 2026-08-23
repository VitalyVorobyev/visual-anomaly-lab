"""The vertical slice, end to end: create, train, score, evaluate, read.

Everything here runs on `pixel_reference`, which needs neither torch nor a network and
trains in milliseconds on 16x16 fixtures. That is exactly why it exists — the whole
results path is provable without the optional deep-learning group installed.

The fixtures are synthetic PNGs generated in code (ADR-0022): a smooth gradient with a
little noise for normals, and the same gradient with a bright square stamped in for
defects, plus matching ground-truth masks so the pixel metrics have something to measure.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import JobKind, JobStatus, Label
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.regions.preparation import run_region_prepare_job

from .conftest import FIXTURE_SIZE as SIZE
from .conftest import TEST_DEFECTS, TEST_NORMALS, TRAIN_NORMALS, Fixture
from .conftest import create_experiment as _create
from .conftest import run_handler as _run

# ----------------------------------------------------------------- the catalog


def test_the_method_catalog_carries_schemas_rather_than_field_lists(client: TestClient) -> None:
    """The claim ADR-0007 makes: the form is generated, so adding a field costs no UI."""
    payload = client.get("/api/experiments/model-types").json()

    keys = {method["key"] for method in payload["methods"]}
    assert "pixel_reference" in keys
    assert set(payload["preprocessing_schema"]["properties"]) == {"color"}
    assert "aggregation" in payload["evaluation_schema"]["properties"]

    baseline = next(m for m in payload["methods"] if m["key"] == "pixel_reference")
    assert baseline["config_schema"]["properties"]["smoothing_sigma"]["description"]
    assert baseline["capabilities"]["produces_anomaly_map"] is True
    assert baseline["capabilities"]["portable_formats"] == ["onnx"]
    assert baseline["availability"]["available"] is True


def test_every_method_option_carries_a_description(client: TestClient) -> None:
    """A generated form with no help text is a worse form than a hand-written one."""
    payload = client.get("/api/experiments/model-types").json()
    for method in payload["methods"]:
        for name, field in method["config_schema"]["properties"].items():
            assert field.get("description"), f"{method['key']}.{name} has no description"


# ----------------------------------------------------------------- creation


def test_creating_an_experiment_freezes_its_configuration(
    client: TestClient, seeded: Fixture
) -> None:
    created = _create(client, seeded)
    assert created["status"] == "draft"
    assert created["config"]["smoothing_sigma"] == 1.0
    # Defaults the caller never mentioned are resolved and stored, so the record is
    # complete rather than partial.
    assert created["config"]["score_percentile"] == 99.5
    assert created["preprocessing"]["color"] == "rgb"
    assert created["preprocessing"]["width"] == SIZE
    assert created["preprocessing"]["height"] == SIZE
    assert created["region_profile_id"] == seeded.region_profile_id
    assert len(created["region_manifest_sha256"]) == 64
    assert created["artifact_dir"].endswith(f"exp-{created['id']}")


def test_an_unbuilt_region_profile_cannot_become_experiment_input(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        profile = region_profiles_repo.create_revision(
            conn,
            dataset_id=seeded.dataset_id,
            name="not built",
            extractor_type="identity",
            extractor_config={},
            prepared_width=SIZE,
            prepared_height=SIZE,
            padding_fraction=0.0,
            seed=29,
        )

    response = client.post(
        "/api/experiments",
        json={
            "name": "missing pixels",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": profile.id,
            "model_type": "pixel_reference",
        },
    )

    assert response.status_code == 422
    assert "has not been built" in response.text


def test_an_invalid_configuration_is_refused_at_creation_not_at_job_time(
    client: TestClient, seeded: Fixture
) -> None:
    """A typo should be a 422 on the form, not a failed job discovered ten minutes later."""
    response = client.post(
        "/api/experiments",
        json={
            "name": "bad",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "pixel_reference",
            "config": {"score_percentile": 900},
        },
    )
    assert response.status_code == 422
    assert "score_percentile" in response.text


def test_an_unknown_method_is_refused_by_name(client: TestClient, seeded: Fixture) -> None:
    response = client.post(
        "/api/experiments",
        json={
            "name": "bad",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "not_a_method",
        },
    )
    assert response.status_code == 422
    assert "pixel_reference" in response.text


def test_a_split_from_another_dataset_is_refused(client: TestClient, seeded: Fixture) -> None:
    other = client.post(
        "/api/experiments",
        json={
            "name": "mismatched",
            "dataset_id": seeded.dataset_id + 999,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "pixel_reference",
        },
    )
    assert other.status_code == 404


# ----------------------------------------------------------------- training


def test_training_reports_what_it_learned_from(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    experiment = _create(client, seeded)
    result = _run(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})

    assert result["train_images"] == TRAIN_NORMALS
    assert result["excluded_images"] == 0
    assert result["device"] == "cpu"
    assert result["diagnostics"] > 0

    detail = client.get(f"/api/experiments/{experiment['id']}").json()
    assert detail["status"] == "trained"


def test_training_uses_normals_only_and_says_so(
    client: TestClient, settings: Settings, seeded: Fixture, migrated_db: sqlite3.Connection
) -> None:
    """A defect in the training set teaches the model that defects are normal."""
    with connection(settings.db_path) as conn:
        sample = samples_repo.find_sample(
            conn, seeded.dataset_id, group_key="all", external_id="train-0"
        )
        assert sample is not None
        samples_repo.set_label(conn, sample.id, Label.DEFECT)

    experiment = _create(client, seeded)
    result = _run(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})

    assert result["train_images"] == TRAIN_NORMALS - 1
    assert result["excluded_images"] == 1


def test_inference_before_training_refuses_rather_than_scoring_noise(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    experiment = _create(client, seeded)
    with pytest.raises(Exception, match="not trained"):
        _run(settings, JobKind.INFER, {"experiment_id": experiment["id"], "subsets": ["test"]})


def test_a_subset_with_no_samples_says_which_one(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """VisA's official protocol has no val subset; the message has to name that case."""
    experiment = _create(client, seeded)
    _run(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
    with pytest.raises(Exception, match="val"):
        _run(settings, JobKind.INFER, {"experiment_id": experiment["id"], "subsets": ["val"]})


# ----------------------------------------------------------------- results


def test_metrics_are_computed_and_stored_per_subset(
    client: TestClient, scored: dict[str, Any]
) -> None:
    detail = client.get(f"/api/experiments/{scored['id']}").json()
    by_subset = {entry["subset"]: entry["metrics"] for entry in detail["metrics"]}

    assert set(by_subset) == {"test"}
    test = by_subset["test"]
    assert test["sample_roc_auc"] == pytest.approx(1.0)
    assert test["image_roc_auc"] == pytest.approx(1.0)
    assert test["samples"] == {
        "total": TEST_NORMALS + TEST_DEFECTS,
        "normal": TEST_NORMALS,
        "defect": TEST_DEFECTS,
        "unlabeled": 0,
    }
    assert test["aggregation"] == "max"
    assert test["timing"]["mean_ms"] > 0
    assert detail["headline_roc_auc"] == pytest.approx(1.0)
    assert detail["metrics"][0]["ground_truth_digest"]
    assert detail["metrics"][0]["ground_truth_stale"] is False


def test_pixel_metrics_appear_because_this_dataset_has_masks(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """The capability ADR-0011 said might never exist, now that a dataset supplies masks."""
    detail = client.get(f"/api/experiments/{scored['id']}").json()
    pixel = next(e for e in detail["metrics"] if e["subset"] == "test")["metrics"]["pixel"]

    assert pixel["pixel_roc_auc"] is not None
    assert pixel["au_pro"] is not None
    assert pixel["mask_regions"] == TEST_DEFECTS
    # Normal images are folded in with an all-zero mask, which is the protocol published
    # numbers are computed under.
    assert pixel["mask_images"] == TEST_NORMALS + TEST_DEFECTS
    assert pixel["skipped_unannotated_defects"] == 0


def test_ranked_results_come_back_most_anomalous_first(
    client: TestClient, scored: dict[str, Any]
) -> None:
    page = client.get(f"/api/experiments/{scored['id']}/results?subset=test").json()

    scores = [sample["score"] for sample in page["samples"]]
    assert scores == sorted(scores, reverse=True)
    assert page["samples"][0]["label"] == "defect"
    assert page["threshold_rationale"]
    assert page["score_max"] >= page["score_min"]


def test_the_threshold_endpoint_recomputes_without_writing(
    client: TestClient, scored: dict[str, Any]
) -> None:
    page = client.get(f"/api/experiments/{scored['id']}/results?subset=test").json()
    suggested = page["suggested_threshold"]

    report = client.get(
        f"/api/experiments/{scored['id']}/threshold",
        params={"value": suggested, "subset": "test"},
    ).json()
    assert report["confusion"]["true_positive"] == TEST_DEFECTS
    assert report["confusion"]["false_positive"] == 0
    assert report["recall"] == pytest.approx(1.0)

    # A threshold below every score calls everything defective.
    everything = client.get(
        f"/api/experiments/{scored['id']}/threshold",
        params={"value": page["score_min"] - 1.0, "subset": "test"},
    ).json()
    assert everything["confusion"]["false_positive"] == TEST_NORMALS
    assert everything["confusion"]["true_negative"] == 0


def test_reevaluating_changes_the_reading_without_re_running_inference(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """The seam that makes aggregation a re-read: `mean` instead of `max`, no re-scoring."""
    experiment = _create(client, seeded, evaluation={"aggregation": "mean"})
    _run(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
    _run(settings, JobKind.INFER, {"experiment_id": experiment["id"], "subsets": ["test"]})

    metrics = client.post(f"/api/experiments/{experiment['id']}/reevaluate").json()
    assert metrics[0]["metrics"]["aggregation"] == "mean"


def test_per_image_scores_are_available_for_the_viewer(
    client: TestClient, scored: dict[str, Any], seeded: Fixture, settings: Settings
) -> None:
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, seeded.defect_image_ids[0])
    assert image is not None

    rows = client.get(f"/api/experiments/{scored['id']}/samples/{image.sample_id}/images").json()
    assert len(rows) == 1
    assert rows[0]["has_map"] is True
    assert rows[0]["has_mask"] is True
    assert rows[0]["inference_ms"] > 0


# ----------------------------------------------------------------- rendering


def test_an_anomaly_map_renders_at_the_source_resolution(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    """Alignment is the whole point of the overlay; a mismatched size cannot align."""
    image_id = seeded.defect_image_ids[0]
    response = client.get(
        f"/api/images/{image_id}/anomaly-map", params={"experiment_id": scored["id"]}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    import io

    rendered = Image.open(io.BytesIO(response.content))
    assert rendered.size == (SIZE, SIZE)


def test_the_map_is_transparent_where_the_model_found_nothing(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    """An opaque colormap over a photograph tints the whole frame.

    The regions where the model found nothing would then look as processed as the region
    where it found something, which is the opposite of what an overlay is for. So alpha
    follows the score: low means see-through, and the source image stays readable.
    """
    import io

    response = client.get(
        f"/api/images/{seeded.defect_image_ids[0]}/anomaly-map",
        params={"experiment_id": scored["id"]},
    )
    rendered = Image.open(io.BytesIO(response.content))
    assert rendered.mode == "RGBA"

    alpha = np.asarray(rendered)[:, :, 3]
    assert int(alpha.min()) == 0, "the quietest pixel must be fully transparent"
    assert int(alpha.max()) > 200, "the loudest pixel must be close to opaque"
    # A map is mostly quiet; if the median pixel were opaque the overlay would be a veil.
    assert int(np.median(alpha)) < 128


def test_a_ground_truth_mask_renders_as_a_transparent_outline(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    import io

    response = client.get(f"/api/images/{seeded.defect_image_ids[0]}/mask")
    assert response.status_code == 200

    rendered = Image.open(io.BytesIO(response.content))
    assert rendered.mode == "RGBA"
    alpha = np.asarray(rendered)[:, :, 3]
    # An outline, not a fill: some pixels opaque, most transparent.
    assert 0 < int((alpha > 0).sum()) < SIZE * SIZE // 2


def test_an_image_with_no_mask_says_so_rather_than_returning_an_empty_one(
    client: TestClient, scored: dict[str, Any], seeded: Fixture
) -> None:
    assert client.get(f"/api/images/{seeded.normal_image_ids[0]}/mask").status_code == 404


def test_asking_for_a_map_from_an_experiment_that_has_none_is_a_404(
    client: TestClient, seeded: Fixture
) -> None:
    experiment = _create(client, seeded)
    response = client.get(
        f"/api/images/{seeded.defect_image_ids[0]}/anomaly-map",
        params={"experiment_id": experiment["id"]},
    )
    assert response.status_code == 404


# ----------------------------------------------------------------- curves


def test_curves_are_served_for_the_benchmark_charts(
    client: TestClient, scored: dict[str, Any]
) -> None:
    payload = client.get(
        f"/api/experiments/{scored['id']}/curves", params={"subset": "test"}
    ).json()

    for key in ("sample_roc", "sample_pr", "image_roc", "image_pr"):
        curve = payload[key]
        assert curve is not None, key
        assert len(curve["x"]) == len(curve["y"])
        assert curve["dropped"] == 0

    roc = payload["sample_roc"]
    assert (roc["x"][0], roc["y"][0]) == (0.0, 0.0)
    assert (roc["x"][-1], roc["y"][-1]) == (1.0, 1.0)


def test_only_the_pr_curve_carries_thresholds(client: TestClient, scored: dict[str, Any]) -> None:
    """PR is drawn against the slider; ROC's synthetic endpoints have no finite score."""
    payload = client.get(
        f"/api/experiments/{scored['id']}/curves", params={"subset": "test"}
    ).json()

    for key in ("sample_pr", "image_pr"):
        curve = payload[key]
        assert len(curve["t"]) == len(curve["x"]), key
        # Descending, because the curve walks the score downwards one cut at a time.
        assert curve["t"] == sorted(curve["t"], reverse=True), key

    for key in ("sample_roc", "image_roc"):
        assert payload[key]["t"] == [], key


def test_a_curve_agrees_with_the_metric_it_is_drawn_beside(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """The chart and the number come from one implementation (§8)."""
    curves = client.get(f"/api/experiments/{scored['id']}/curves", params={"subset": "test"}).json()
    detail = client.get(f"/api/experiments/{scored['id']}").json()
    stored = next(entry for entry in detail["metrics"] if entry["subset"] == "test")

    area = float(np.trapezoid(curves["sample_roc"]["y"], curves["sample_roc"]["x"]))
    assert area == pytest.approx(stored["metrics"]["sample_roc_auc"], abs=1e-9)


def test_a_subset_with_one_class_has_no_curve_rather_than_an_empty_one(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """`train` is normals only by construction, so it cannot support a ROC curve."""
    payload = client.get(
        f"/api/experiments/{scored['id']}/curves", params={"subset": "train"}
    ).json()

    assert payload["sample_roc"] is None
    assert payload["sample_pr"] is None


# ----------------------------------------------------------------- diagnostics


def test_diagnostics_are_self_describing_and_never_keyed_by_method_name(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """The property M4's visualization depends on (ADR-0018)."""
    index = client.get(f"/api/experiments/{scored['id']}/diagnostics").json()

    kinds = {entry["kind"] for entry in index["entries"]}
    assert kinds <= {"map", "image", "grid", "graph", "table"}
    assert any(entry["scope"] == "image" for entry in index["entries"])
    for entry in index["entries"]:
        assert entry["title"]
        assert entry.get("path") or entry.get("payload") is not None


def _array_entries(client: TestClient, experiment_id: int) -> list[dict[str, Any]]:
    index = client.get(f"/api/experiments/{experiment_id}/diagnostics").json()
    return [entry for entry in index["entries"] if entry.get("path")]


def test_every_array_diagnostic_renders_as_a_png(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """One route, driven by `kind` — the property M6 inherits for free."""
    entries = _array_entries(client, scored["id"])
    assert entries

    for entry in entries:
        query: dict[str, Any] = {"key": entry["key"]}
        if entry.get("image_id") is not None:
            query["image_id"] = entry["image_id"]
        response = client.get(f"/api/experiments/{scored['id']}/diagnostics/payload", params=query)
        assert response.status_code == 200, entry
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_diagnostic_payload_is_addressed_through_the_index_not_by_path(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """There is no request that can name a file, so none can name the wrong one (§11)."""
    for key in ("../../../etc/passwd", "../diagnostics", "reference_median/../.."):
        response = client.get(
            f"/api/experiments/{scored['id']}/diagnostics/payload",
            params={"key": key},
        )
        assert response.status_code == 404


def test_an_unrecorded_diagnostic_is_a_404_naming_the_scope(
    client: TestClient, scored: dict[str, Any]
) -> None:
    response = client.get(
        f"/api/experiments/{scored['id']}/diagnostics/payload",
        params={"key": "no_such_key"},
    )
    assert response.status_code == 404
    assert "no_such_key" in response.text

    # A key that exists run-scoped is still absent for an image, and says so.
    scoped = client.get(
        f"/api/experiments/{scored['id']}/diagnostics/payload",
        params={"key": "reference_median", "image_id": 999999},
    )
    assert scoped.status_code == 404
    assert "image 999999" in scoped.text


def test_an_inline_kind_refuses_rather_than_inventing_a_second_source_of_truth(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    root = settings.experiment_dir(scored["id"]) / "diagnostics"
    index = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "key": "architecture",
            "title": "Arch",
            "kind": "graph",
            "scope": "model",
            "payload": {"nodes": [], "edges": []},
        }
    )
    (root / "diagnostics.json").write_text(json.dumps(index), encoding="utf-8")

    response = client.get(
        f"/api/experiments/{scored['id']}/diagnostics/payload",
        params={"key": "architecture"},
    )
    assert response.status_code == 400
    assert "inline" in response.text


def test_a_deleted_payload_reads_as_gone_rather_than_as_corruption(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """The artifact directory is deletable by design — the same 410 a missing map gets."""
    entry = _array_entries(client, scored["id"])[0]
    (settings.experiment_dir(scored["id"]) / "diagnostics" / entry["path"]).unlink()

    query: dict[str, Any] = {"key": entry["key"]}
    if entry.get("image_id") is not None:
        query["image_id"] = entry["image_id"]
    response = client.get(f"/api/experiments/{scored['id']}/diagnostics/payload", params=query)
    assert response.status_code == 410


def test_a_grid_frame_beyond_the_payload_is_a_404_naming_the_count(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    root = settings.experiment_dir(scored["id"]) / "diagnostics"
    np.save(root / "cells.npy", np.zeros((3, 4, 4), dtype=np.float32))
    index = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    index["entries"].append(
        {
            "key": "cells",
            "title": "Cells",
            "kind": "grid",
            "scope": "model",
            "path": "cells.npy",
            "shape": [3, 4, 4],
        }
    )
    (root / "diagnostics.json").write_text(json.dumps(index), encoding="utf-8")

    url = f"/api/experiments/{scored['id']}/diagnostics/payload"
    assert client.get(url, params={"key": "cells", "frame": 2}).status_code == 200

    missing = client.get(url, params={"key": "cells", "frame": 3})
    assert missing.status_code == 404
    assert "3 frame(s)" in missing.text


def _plane(payload: bytes) -> tuple[np.ndarray, int]:
    """The header the frontend decodes (handbook diagnostics.md), returning `(H, W, C)`."""
    assert payload[:4] == b"VAM1"
    width, height, stride, channels, _ = struct.unpack("<IIIII", payload[4:24])
    values = np.frombuffer(payload[24:], dtype="<f4")
    assert values.size == width * height * channels
    return np.moveaxis(values.reshape(channels, height, width), 0, -1), stride


def test_an_anomaly_map_serves_its_own_numbers_for_a_readout(
    client: TestClient, seeded: Fixture, scored: dict[str, Any]
) -> None:
    """A rendered map cannot be read back: the colormap clips and quantizes.

    See handbook diagnostics.md.
    """
    image_id = seeded.defect_image_ids[0]
    response = client.get(
        f"/api/images/{image_id}/anomaly-map/values",
        params={"experiment_id": scored["id"]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    plane, stride = _plane(response.content)
    assert stride == 1
    assert plane.shape == (SIZE, SIZE, 1)

    # These are the very floats the metrics were computed from, so the readout has to fall
    # inside the range the viewer prints beside the picture.
    detail = client.get(f"/api/experiments/{scored['id']}").json()
    assert detail["map_range"]["low"] <= float(plane.min())


def test_a_diagnostic_serves_raw_values_through_the_same_addressing(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """Which is why the per-branch panes inherit the readout with no per-method code."""
    entry = next(
        item for item in _array_entries(client, scored["id"]) if item["kind"] in {"map", "grid"}
    )
    query: dict[str, Any] = {"key": entry["key"], "format": "raw"}
    if entry.get("image_id") is not None:
        query["image_id"] = entry["image_id"]

    response = client.get(f"/api/experiments/{scored['id']}/diagnostics/payload", params=query)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    plane, _ = _plane(response.content)
    assert plane.shape[-1] == 1


def test_the_preprocessed_source_is_served_rather_than_the_display_tier(
    client: TestClient, seeded: Fixture, scored: dict[str, Any]
) -> None:
    """The number the *method* read, at the experiment's own size and colour mode.

    Every colour plane arrives in one response with the count in the header, so a client
    never has to know how many there are — which would be channel count as schema.
    """
    image_id = seeded.defect_image_ids[0]
    response = client.get(f"/api/experiments/{scored['id']}/images/{image_id}/source-values")

    assert response.status_code == 200
    plane, _ = _plane(response.content)
    assert plane.shape[:2] == (SIZE, SIZE)
    assert plane.shape[-1] >= 1
    # `load_array` normalizes to [0, 1]; an 8-bit display tier would not.
    assert float(plane.min()) >= 0.0
    assert float(plane.max()) <= 1.0
    # The fixture stamps a 255 square into the defect, so the peak is saturated.
    assert float(plane.max()) == pytest.approx(1.0)


def test_model_input_values_are_projected_back_into_source_coordinates(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """A crop-sized value plane cannot be indexed by a source-image pointer."""
    with connection(settings.db_path) as conn:
        profile = region_profiles_repo.create_revision(
            conn,
            dataset_id=seeded.dataset_id,
            name="coarse identity input",
            extractor_type="identity",
            extractor_config={},
            prepared_width=SIZE // 2,
            prepared_height=SIZE // 2,
            padding_fraction=0.0,
            seed=31,
        )
    run_region_prepare_job(
        JobContext(
            job_id=98,
            kind=JobKind.REGION_PREPARE,
            params={
                "dataset_id": seeded.dataset_id,
                "profile_id": profile.id,
                "mode": "build",
            },
            settings=settings,
        )
    )
    experiment = _create(client, seeded, region_profile_id=profile.id)

    response = client.get(
        f"/api/experiments/{experiment['id']}/images/{seeded.defect_image_ids[0]}/source-values"
    )

    assert response.status_code == 200
    plane, _ = _plane(response.content)
    assert plane.shape[:2] == (SIZE, SIZE)
    assert np.isfinite(plane).all()


def test_source_values_revalidate_on_the_frozen_preprocessing(
    client: TestClient, seeded: Fixture, scored: dict[str, Any]
) -> None:
    """Immutable in practice — the file is fixed and the config is frozen at creation."""
    url = f"/api/experiments/{scored['id']}/images/{seeded.defect_image_ids[0]}/source-values"
    first = client.get(url)
    again = client.get(url, headers={"if-none-match": first.headers["etag"]})
    assert again.status_code == 304


def test_the_payload_validator_covers_the_scale_it_was_drawn_on(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """The same array over a different span is a different picture.

    The validator hashed size and mtime only, so widening a key's range — re-inference, and
    now an on-demand diagnostic — left every already-fetched PNG cached at the old scale
    with nothing on screen to say so.
    """
    entry = next(item for item in _array_entries(client, scored["id"]) if item["kind"] == "map")
    query: dict[str, Any] = {"key": entry["key"]}
    if entry.get("image_id") is not None:
        query["image_id"] = entry["image_id"]

    url = f"/api/experiments/{scored['id']}/diagnostics/payload"
    before = client.get(url, params=query).headers["etag"]

    root = settings.experiment_dir(scored["id"]) / "diagnostics"
    index = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    index["ranges"][entry["key"]] = {"low": -99.0, "high": 99.0}
    (root / "diagnostics.json").write_text(json.dumps(index), encoding="utf-8")

    after = client.get(url, params=query)
    assert after.headers["etag"] != before
    # And the stale validator no longer wins a 304.
    assert client.get(url, params=query, headers={"if-none-match": before}).status_code == 200


def test_a_payload_revalidates_rather_than_being_cached_forever(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """Re-running inference overwrites an image's maps in place, so `immutable` would lie."""
    entry = _array_entries(client, scored["id"])[0]
    query: dict[str, Any] = {"key": entry["key"]}
    if entry.get("image_id") is not None:
        query["image_id"] = entry["image_id"]

    url = f"/api/experiments/{scored['id']}/diagnostics/payload"
    first = client.get(url, params=query)
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "no-cache"

    again = client.get(url, params=query, headers={"if-none-match": etag})
    assert again.status_code == 304


def test_every_image_of_one_key_is_drawn_on_the_run_s_scale(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """A recorded range is what makes two images comparable by eye (handbook diagnostics.md)."""
    root = settings.experiment_dir(scored["id"]) / "diagnostics"
    ranges = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))["ranges"]

    per_image = [entry for entry in _array_entries(client, scored["id"]) if entry.get("image_id")]
    assert per_image
    span = ranges[per_image[0]["key"]]
    assert span["high"] > span["low"]


# ----------------------------------------------------------------- lifecycle


def test_reopening_an_experiment_returns_identical_numbers(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """Nothing is recomputed on read, so 'identical' is structural rather than lucky."""
    first = client.get(f"/api/experiments/{scored['id']}").json()
    second = client.get(f"/api/experiments/{scored['id']}").json()
    assert first["metrics"] == second["metrics"]


def test_the_experiment_list_shows_the_headline_number(
    client: TestClient, scored: dict[str, Any]
) -> None:
    listed = client.get("/api/experiments").json()
    entry = next(item for item in listed if item["id"] == scored["id"])
    assert entry["headline_roc_auc"] == pytest.approx(1.0)


def test_the_experiment_catalog_combines_search_method_status_and_sort(
    client: TestClient, seeded: Fixture
) -> None:
    first = _create(client, seeded, name="Needle baseline")
    second = _create(client, seeded, name="Needle follow-up")
    _create(client, seeded, name="Unrelated")

    listed = client.get(
        "/api/experiments",
        params={
            "dataset_id": seeded.dataset_id,
            "model_type": "pixel_reference",
            "status": "draft",
            "q": "needle",
            "sort": "oldest",
        },
    )

    assert listed.status_code == 200
    assert [entry["id"] for entry in listed.json()] == [first["id"], second["id"]]


def test_the_experiment_catalog_rejects_unknown_filter_values(
    client: TestClient,
) -> None:
    assert client.get("/api/experiments", params={"status": "ghost"}).status_code == 422
    assert client.get("/api/experiments", params={"sort": "fastest"}).status_code == 422


def test_deleting_an_experiment_removes_its_artifacts_too(
    client: TestClient, scored: dict[str, Any]
) -> None:
    artifact_dir = Path(client.get(f"/api/experiments/{scored['id']}").json()["artifact_dir"])
    assert artifact_dir.is_dir()

    deleted = client.delete(f"/api/experiments/{scored['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert deleted.json()["artifacts_removed"] is True
    assert client.get(f"/api/experiments/{scored['id']}").status_code == 404
    assert not artifact_dir.exists()


def test_deletion_preview_reports_exact_app_owned_payload(
    client: TestClient, seeded: Fixture
) -> None:
    experiment = _create(client, seeded, name="preview me")
    artifact_dir = Path(experiment["artifact_dir"])
    (artifact_dir / "nested").mkdir()
    (artifact_dir / "model.bin").write_bytes(b"1234")
    (artifact_dir / "nested" / "metrics.json").write_bytes(b"567")

    preview = client.get(f"/api/experiments/{experiment['id']}/deletion-preview")

    assert preview.status_code == 200
    assert preview.json() == {
        "experiment_id": experiment["id"],
        "name": "preview me",
        "generated_files": 2,
        "generated_bytes": 7,
        "active_jobs": [],
        "resident_loaded": False,
        "artifact_location_safe": True,
        "can_delete": True,
        "blocker": None,
    }

    result = client.delete(f"/api/experiments/{experiment['id']}").json()
    assert result["freed_files"] == 2
    assert result["freed_bytes"] == 7


def test_active_work_blocks_experiment_deletion(
    client: TestClient, seeded: Fixture, migrated_db: sqlite3.Connection
) -> None:
    experiment = _create(client, seeded, name="busy")
    cursor = migrated_db.execute(
        "INSERT INTO job (kind, experiment_id, status, params) VALUES (?, ?, ?, '{}')",
        (JobKind.TRAIN.value, experiment["id"], JobStatus.RUNNING.value),
    )
    job_id = int(cursor.lastrowid or 0)

    preview = client.get(f"/api/experiments/{experiment['id']}/deletion-preview").json()
    assert preview["can_delete"] is False
    assert [job["id"] for job in preview["active_jobs"]] == [job_id]

    refused = client.delete(f"/api/experiments/{experiment['id']}")
    assert refused.status_code == 409
    assert client.get(f"/api/experiments/{experiment['id']}").status_code == 200


def test_deletion_never_follows_a_corrupt_artifact_path(
    client: TestClient,
    seeded: Fixture,
    migrated_db: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    experiment = _create(client, seeded, name="unsafe path")
    external = tmp_path / "external-source"
    external.mkdir()
    sentinel = external / "source-image-sentinel.png"
    sentinel.write_bytes(b"synthetic sentinel")
    migrated_db.execute(
        "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
        (str(external), experiment["id"]),
    )

    preview = client.get(f"/api/experiments/{experiment['id']}/deletion-preview").json()
    assert preview["artifact_location_safe"] is False
    assert preview["can_delete"] is False

    refused = client.delete(f"/api/experiments/{experiment['id']}")
    assert refused.status_code == 409
    assert sentinel.read_bytes() == b"synthetic sentinel"


def test_the_queue_carries_a_train_job_with_no_change_to_itself(
    client: TestClient, seeded: Fixture
) -> None:
    """ADR-0009's claim, tested through the real subprocess path.

    A `train` job is enqueued exactly like an import job, runs in its own worker, and
    reaches a terminal state — with no branch anywhere in the queue for what kind it is.
    """
    import time

    experiment = _create(client, seeded)
    queued = client.post(f"/api/experiments/{experiment['id']}/train").json()
    assert queued["kind"] == "train"
    assert queued["experiment_id"] == experiment["id"]

    deadline = time.monotonic() + 60.0
    payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{queued['id']}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.05)

    assert payload["status"] == "succeeded", payload.get("error")
    assert payload["result"]["train_images"] == TRAIN_NORMALS
    assert client.get(f"/api/experiments/{experiment['id']}").json()["status"] == "trained"


def test_an_experiment_left_mid_training_is_reconciled_at_startup(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """The sibling of the stale-job reconciliation, and needed for the same reason.

    A crash or a force-quit leaves the row reading `training`, which on screen is
    indistinguishable from a run that is genuinely in progress. Only one job runs at a
    time, so at startup nothing is — and anything still `training` is the record of a
    run that died.
    """
    from fastapi.testclient import TestClient as Client

    from anomaly_lab.api.app import create_app
    from anomaly_lab.domain.entities import ExperimentStatus

    experiment = _create(client, seeded)
    with connection(settings.db_path) as conn:
        experiments_repo.set_status(conn, experiment["id"], ExperimentStatus.TRAINING)

    with Client(create_app(settings)) as restarted:
        reopened = restarted.get(f"/api/experiments/{experiment['id']}").json()

    assert reopened["status"] == "failed"


# ------------------------------------------------------- continuing a run (handbook jobs.md)


def test_continuing_a_method_without_steps_is_refused_as_a_form(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """422, not a job that fails minutes later inside a worker.

    `pixel_reference` builds a median over the training set; it is either fitted or not,
    and there is no step to continue. The message says that rather than "invalid".
    """
    response = client.post(
        f"/api/experiments/{scored['id']}/train",
        json={"experiment_id": scored["id"], "additional_steps": 500},
    )

    assert response.status_code == 422
    assert "no notion of a training step" in response.text


def test_a_method_without_resume_reports_so_and_carries_no_training_state(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """`training_state` is absent rather than zero: a fabricated step count reads as fact."""
    detail = client.get(f"/api/experiments/{scored['id']}").json()

    assert detail["supports_resume"] is False
    assert detail["training_state"] is None
    assert detail["portable_formats"] == ["onnx"]


def test_export_is_refused_before_training(client: TestClient, seeded: Fixture) -> None:
    experiment = _create(client, seeded, name="not fitted")

    response = client.post(f"/api/experiments/{experiment['id']}/export")

    assert response.status_code == 422
    assert "train the experiment" in response.text


def test_export_uses_the_generic_job_queue(client: TestClient, scored: dict[str, Any]) -> None:
    queued = client.post(f"/api/experiments/{scored['id']}/export")

    assert queued.status_code == 200
    assert queued.json()["kind"] == "export"
    assert queued.json()["experiment_id"] == scored["id"]


def test_training_without_additional_steps_is_unaffected(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """The ordinary path must not have acquired a precondition."""
    response = client.post(
        f"/api/experiments/{scored['id']}/train", json={"experiment_id": scored["id"]}
    )
    assert response.status_code == 200


def test_additional_steps_stays_optional_on_the_wire(client: TestClient) -> None:
    """`default_factory`, not `= None`.

    A literal default emits `"default": null` into the schema and `openapi-typescript`
    then makes the property **required** — the trap that silently pinned a value for two
    milestones. The schema must therefore carry no default for this field at all.
    """
    schema = client.get("/openapi.json").json()
    train_params = schema["components"]["schemas"]["TrainParams"]

    assert "additional_steps" not in train_params.get("required", [])
    assert "default" not in train_params["properties"]["additional_steps"]


# ------------------------------------------------------- clearing diagnostics


def test_clearing_images_keeps_the_run_scoped_entries(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """What the Architecture and Inspector tabs draw must survive a routine disk clear."""
    before = client.get(f"/api/experiments/{scored['id']}/diagnostics").json()
    assert any(entry["scope"] == "image" for entry in before["entries"])
    assert any(entry["scope"] == "model" for entry in before["entries"])

    response = client.delete(f"/api/experiments/{scored['id']}/diagnostics")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["removed_entries"] > 0
    assert result["bytes_reclaimed"] > 0

    after = client.get(f"/api/experiments/{scored['id']}/diagnostics").json()
    assert not any(entry["scope"] == "image" for entry in after["entries"])
    assert any(entry["scope"] == "model" for entry in after["entries"])


def test_clearing_diagnostics_leaves_the_anomaly_maps_alone(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """Deleting a map orphans an `ImageResult` row and breaks the overlay silently."""
    maps = settings.experiment_dir(scored["id"]) / "maps"
    before = sorted(path.name for path in maps.iterdir())
    assert before

    client.delete(f"/api/experiments/{scored['id']}/diagnostics", params={"scope": "all"})

    assert sorted(path.name for path in maps.iterdir()) == before
    overlay = client.get(f"/api/experiments/{scored['id']}/results", params={"subset": "test"})
    assert overlay.status_code == 200


def test_clearing_an_experiment_that_recorded_nothing_is_not_an_error(
    client: TestClient, seeded: Fixture
) -> None:
    """The button exists before the first run does."""
    experiment = _create(client, seeded)

    response = client.delete(f"/api/experiments/{experiment['id']}/diagnostics")

    assert response.status_code == 200
    assert response.json()["removed_entries"] == 0


# ------------------------------------------------------ channel selection (ADR-0035)


def test_an_experiment_reads_every_channel_by_default(client: TestClient, seeded: Fixture) -> None:
    """Empty means all, which is what every experiment created before the column meant."""
    experiment = _create(client, seeded)

    assert experiment["channels"] == []


def test_selecting_a_channel_a_dataset_does_not_have_is_refused(
    client: TestClient, seeded: Fixture, settings: Settings
) -> None:
    """A run that quietly read nothing would look exactly like an empty split."""
    with connection(settings.db_path) as conn:
        datasets_repo.upsert_channel(conn, seeded.dataset_id, name="bright", position=0)

    response = client.post(
        "/api/experiments",
        json={
            "name": "typo",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "pixel_reference",
            "channels": ["birght"],
        },
    )

    assert response.status_code == 422
    assert "birght" in response.text
    assert "bright" in response.text  # it names what is actually available


def test_selecting_a_channel_on_a_dataset_that_has_none_is_refused(
    client: TestClient, seeded: Fixture
) -> None:
    response = client.post(
        "/api/experiments",
        json={
            "name": "nope",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "pixel_reference",
            "channels": ["bright"],
        },
    )

    assert response.status_code == 422
    assert "no channels" in response.text


def test_a_channel_selection_is_frozen_in_the_datasets_own_order(
    client: TestClient, seeded: Fixture, settings: Settings
) -> None:
    """Two identical requests must produce identical frozen records, so a diff between two
    experiments never shows a difference that is only a permutation."""
    with connection(settings.db_path) as conn:
        for index, name in enumerate(("bright", "dark", "dome")):
            datasets_repo.upsert_channel(conn, seeded.dataset_id, name=name, position=index)

    experiment = _create(client, seeded, name="picked", channels=["dome", "bright"])

    assert experiment["channels"] == ["bright", "dome"]


def test_reevaluate_rebuilds_the_sample_scores_it_reports_on(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """`reevaluate` promises to apply a changed `eval_config` without re-running the model.

    It used to refresh the metric sets while leaving `sample_result` at whatever the
    original run reduced, so the promise held only as long as nobody tested it. Editing the
    stored per-image scores and re-reading is the cheapest way to prove the derivation
    really runs again.
    """
    experiment_id = scored["id"]
    with connection(settings.db_path) as conn:
        before = {
            row["sample_id"]: row["agg_score"]
            for row in conn.execute(
                "SELECT sample_id, agg_score FROM sample_result WHERE experiment_id = ?",
                (experiment_id,),
            )
        }
        conn.execute(
            "UPDATE image_result SET score = score + 10.0 WHERE experiment_id = ?",
            (experiment_id,),
        )
        conn.commit()

    assert client.post(f"/api/experiments/{experiment_id}/reevaluate").status_code == 200

    with connection(settings.db_path) as conn:
        after = {
            row["sample_id"]: row["agg_score"]
            for row in conn.execute(
                "SELECT sample_id, agg_score FROM sample_result WHERE experiment_id = ?",
                (experiment_id,),
            )
        }

    assert after and after.keys() == before.keys()
    assert all(after[sample] == pytest.approx(before[sample] + 10.0) for sample in before)


def test_the_metric_set_records_how_the_channels_were_combined(
    client: TestClient, scored: dict[str, Any]
) -> None:
    """A sample-level number is uninterpretable without both halves of the decision."""
    metrics = client.get(f"/api/experiments/{scored['id']}").json()["metrics"]

    assert metrics
    for entry in metrics:
        assert entry["metrics"]["aggregation"] == "max"
        assert entry["metrics"]["channel_normalization"] == "none"


# ------------------------------------------------------- a method that no longer exists


def test_an_experiment_whose_method_was_removed_stays_readable(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """Retiring a method (e.g. `efficientad_anomalib`, ADR-0029) must not orphan its rows.

    `model_type` carries no foreign key, so an experiment created under a key that used to
    be registered stays in the database after the key is retired. This is what the five
    `except UnknownModelError` sites in `api/routers/experiments.py` are for: capabilities
    degrade to nothing rather than the request failing.

    Creation-time refusal (`create_experiment`, ~line 577) is exercised by
    `test_an_unknown_method_is_refused_by_name` instead — a *new* experiment cannot be
    created under an unregistered key at all, so that site is unreachable for a row that
    already exists, which is the case this test is about.

    A plain `POST .../train` (no `additional_steps`) is not covered here either: it
    enqueues a job unconditionally and only reads `model_type` inside the worker process,
    not synchronously in the router — a retired method fails the *job*, not the request.
    Only the resume path (`additional_steps` set) is checked synchronously, in
    `_refuse_impossible_resume`.
    """
    retired = "a_retired_method"
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.create_experiment(
            conn,
            name="orphaned",
            dataset_id=seeded.dataset_id,
            split_id=seeded.split_id,
            region_profile_id=seeded.region_profile_id,
            region_manifest_sha256="0" * 64,
            model_type=retired,
            model_config={},
            preprocessing_config={},
            eval_config={},
            artifact_dir=str(settings.data_dir / "experiments" / "exp-retired"),
        )

    detail = client.get(f"/api/experiments/{experiment.id}").json()
    assert detail["produces_anomaly_map"] is False
    assert detail["produces_diagnostics"] is False
    assert detail["supports_resume"] is False
    assert detail["portable_formats"] == []

    resumed = client.post(
        f"/api/experiments/{experiment.id}/train",
        json={"experiment_id": experiment.id, "additional_steps": 500},
    )
    assert resumed.status_code == 422
    assert retired in resumed.text

    diagnosed = client.post(f"/api/experiments/{experiment.id}/diagnose", json={"image_id": 1})
    assert diagnosed.status_code == 422
    assert retired in diagnosed.text

    exported = client.post(f"/api/experiments/{experiment.id}/export")
    assert exported.status_code == 422
    assert retired in exported.text
