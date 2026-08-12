"""`patchcore_anomalib` as a plugin: does it detect, is it bounded, can it be stopped.

**Nothing here downloads anything.** The backbone is `resnet18` with `pretrained_backbone`
off — a seeded random network — and that is not a compromise made to keep CI cheap. It was
measured first: on the stamped-defect fixture a random backbone separates by a factor of
forty, because PatchCore's mechanism is nearest-neighbour matching in *some* feature space
and an untrained convolutional stack is still a feature space. The ImageNet weights make it
better on real data and are not what this file is testing.

The centrepiece is `test_it_detects_a_stamped_defect`, for the reason M6 wrote down: a
method that runs and a method that detects are different claims, and only one of them gets
tested by accident. Beside it sits `test_the_greedy_selection_matches_anomalib`, which is
what licenses the module's central liberty — the coreset loop is ours, so it has to be
pinned against the one it replaced.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomaly_lab.eval.metrics import roc_auc  # noqa: E402
from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
    Prediction,
    TrainContext,
)
from anomaly_lab.models.diagnostics import (  # noqa: E402
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
from anomaly_lab.models.patchcore_anomalib import (  # noqa: E402
    STATE_FILENAME,
    LayerSet,
    PatchcoreAnomalibModel,
    PatchcoreConfig,
    greedy_select,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402

SIZE = 64
"""Small on purpose. The backbone strides by 8, so this is an 8x8 patch grid — enough
geometry for a coreset and a coverage map, cheap enough that a fit is under a second."""


def _config(**overrides: object) -> PatchcoreConfig:
    """A fit that needs no network and finishes immediately."""
    base: dict[str, object] = {
        "backbone": "resnet18",
        "pretrained_backbone": False,
        "max_bank_images": 8,
        "max_candidate_vectors": 400,
        "coreset_ratio": 0.25,
        "seed": 0,
    }
    base.update(overrides)
    return PatchcoreConfig(**base)  # type: ignore[arg-type]


def _write_normal(path: Path, rng: np.random.Generator) -> Path:
    """A textured normal. Flat grey would make every patch identical and the bank a point."""
    noise = rng.integers(96, 144, size=(SIZE, SIZE, 3), dtype=np.uint8)
    Image.fromarray(noise).save(path)
    return path


def _write_defect(path: Path, rng: np.random.Generator) -> Path:
    """The same texture with a stamped square — a patch appearance the bank has never seen."""
    noise = rng.integers(96, 144, size=(SIZE, SIZE, 3), dtype=np.uint8)
    noise[24:40, 24:40] = 255
    Image.fromarray(noise).save(path)
    return path


def _contexts(root: Path) -> tuple[TrainContext, InferContext]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig(width=SIZE, height=SIZE)
    diagnostics = artifacts / "diagnostics"

    train_ctx = TrainContext(
        artifact_dir=artifacts,
        cache_dir=cache,
        preprocessing=preprocessing,
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(diagnostics),
    )
    infer_ctx = InferContext(
        artifact_dir=artifacts,
        cache_dir=cache,
        preprocessing=preprocessing,
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(diagnostics),
    )
    return train_ctx, infer_ctx


@dataclass
class Fitted:
    """One fitted bank and everything the assertions read off it.

    A module fixture, because fitting is the expensive part and most of these tests only
    read. Adding a test that fits its own model is a decision about the `dl` job's runtime.
    """

    model: PatchcoreAnomalibModel
    predictions: list[Prediction]
    labels: np.ndarray
    probe: list[ImageRecord]
    train_ctx: TrainContext
    infer_ctx: InferContext
    root: Path


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Fitted:
    root = tmp_path_factory.mktemp("patchcore")
    rng = np.random.default_rng(0)
    train_ctx, infer_ctx = _contexts(root)

    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_write_normal(root / f"n{index}.png", rng)
        )
        for index in range(12)
    ]
    probe = [
        ImageRecord(
            image_id=100 + index,
            sample_id=100 + index,
            path=_write_normal(root / f"p{index}.png", rng),
        )
        for index in range(4)
    ] + [
        ImageRecord(
            image_id=200 + index,
            sample_id=200 + index,
            path=_write_defect(root / f"d{index}.png", rng),
        )
        for index in range(4)
    ]
    labels = np.array([False] * 4 + [True] * 4)

    model = PatchcoreAnomalibModel(_config())
    model.fit(train, train_ctx)
    train_ctx.diagnostics.flush()
    predictions = model.predict(probe, infer_ctx)
    infer_ctx.diagnostics.flush()

    return Fitted(
        model=model,
        predictions=predictions,
        labels=labels,
        probe=probe,
        train_ctx=train_ctx,
        infer_ctx=infer_ctx,
        root=root,
    )


# ------------------------------------------------------------------ the detection bar


def test_it_detects_a_stamped_defect(fitted: Fitted) -> None:
    """The test this method exists to pass.

    A hard bar on a deliberately easy task, in the shape M6 settled on: the measured margin
    is roughly 40x, so this passes comfortably or something is genuinely broken. A softer
    bar on a harder fixture would flake and teach us to lower it.
    """
    scores = np.array([prediction.score for prediction in fitted.predictions])
    labels = fitted.labels

    assert roc_auc(labels, scores) == pytest.approx(1.0)
    assert float(np.median(scores[labels])) > 5 * float(np.median(scores[~labels]))
    assert bool(np.isfinite(scores).all())


def test_the_defect_lands_where_the_defect_is(fitted: Fitted) -> None:
    """A score can be right for the wrong reason, and a map is how you tell.

    M6's own finding was that an image score is a maximum and is therefore decided by
    whatever is hottest, defect or not. Here the stamp is at rows and columns 24..40 of 64,
    so the map's maximum has to fall inside it — otherwise the separation above is a
    coincidence of this fixture rather than localization.
    """
    defect = fitted.predictions[-1]
    assert defect.anomaly_map is not None
    values = np.load(defect.anomaly_map)

    row, col = np.unravel_index(int(np.argmax(values)), values.shape)
    assert 24 <= row < 40
    assert 24 <= col < 40


# ------------------------------------------------------- the loop we took ownership of


def test_the_greedy_selection_matches_anomalib() -> None:
    """Our coreset loop is anomalib's rule, and this is what says so.

    The module replaces `KCenterGreedy.select_coreset_idxs` to get progress and
    cancellation, which is only defensible if the selection itself is unchanged.

    The random projection is deliberately taken out of the comparison. `KCenterGreedy`
    skips it when the embedding is not 2-D — it flattens instead — so both sides are handed
    the *same* feature matrix, and the only thing left to disagree about is the greedy rule.
    Leaving the projection in would have compared two random matrices as much as two loops,
    and the next test is where that randomness is examined on its own.

    The starting index is anomalib's own draw, passed to ours. What is compared is then the
    rule alone, and it has to agree index for index rather than approximately.
    """
    from anomalib.models.components.sampling import KCenterGreedy

    torch.manual_seed(1234)
    embedding = torch.randn(400, 48, 1)

    reference = KCenterGreedy(embedding=embedding, sampling_ratio=0.1).select_coreset_idxs()
    ours = greedy_select(embedding.reshape(400, -1), len(reference), start=reference[0])

    assert ours.tolist() == reference


def test_the_bank_is_reproducible_from_its_seed(tmp_path: Path) -> None:
    """Two runs of one configuration have to be one experiment.

    This is M6's lesson arriving in a different library. `SparseRandomProjection` draws its
    sparsity pattern through scikit-learn's `sample_without_replacement`, whose
    `random_state` defaults to `None` — numpy's global RNG, untouched by
    `torch.manual_seed` — and anomalib constructs it with no `random_state` at all. So
    **anomalib's own coreset is not reproducible**: same seed, same data, different bank,
    and nothing anywhere says so. A `seed` field on the experiment form that did not
    control the result would be a lie, and every comparison built on it would carry noise
    nobody could account for.

    Asserting both directions, because pinning a seed is only meaningful if a different one
    actually changes the answer.
    """
    rng = np.random.default_rng(5)
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_write_normal(tmp_path / f"n{i}.png", rng))
        for i in range(6)
    ]

    def bank(seed: int, tag: str) -> object:
        ctx, _ = _contexts(tmp_path / tag)
        model = PatchcoreAnomalibModel(_config(seed=seed))
        model.fit(train, ctx)
        return model._model.memory_bank.clone()

    first = bank(0, "a")
    again = bank(0, "b")
    different = bank(1, "c")

    assert torch.equal(first, again)
    assert not torch.equal(first, different)


def test_the_selection_stops_when_the_job_is_cancelled(tmp_path: Path) -> None:
    """The property owning the loop was for.

    Anomalib's selection returns when it is finished and not before — at defaults ten
    thousand iterations, at a raised cap half an hour — while every other long operation in
    this workbench stops within one step. The reporter here trips as soon as the selection
    announces itself, so what is asserted is that cancellation lands *inside* the loop and
    that nothing half-built is left behind claiming to be a model.
    """

    class CancelAtSelection(NullReporter):
        def __init__(self) -> None:
            self.armed = False

        def progress(self, fraction: float, message: str | None = None) -> None:
            if message and "projecting" in message:
                self.armed = True

        def should_cancel(self) -> bool:
            return self.armed

    rng = np.random.default_rng(1)
    train_ctx, _ = _contexts(tmp_path)
    reporter = CancelAtSelection()
    train_ctx.reporter = reporter

    train = [
        ImageRecord(image_id=i, sample_id=i, path=_write_normal(tmp_path / f"n{i}.png", rng))
        for i in range(8)
    ]
    model = PatchcoreAnomalibModel(_config())

    with pytest.raises(ModelCancelledError):
        model.fit(train, train_ctx)

    assert model._model is None
    with pytest.raises(RuntimeError, match="before it was fitted or loaded"):
        model.predict(train[:1], _contexts(tmp_path)[1])


# ----------------------------------------------------------------------- the bounding


def test_both_caps_bind_and_the_table_says_what_was_dropped(fitted: Fitted) -> None:
    """Exit criterion one is not "it fits", it is "it fits and says what it cost".

    The fixture is configured so both caps bind at once: twelve images against a limit of
    eight, and 8x64 patches against a pool of 400. The bank has to reflect both, and the
    diagnostic table has to carry the numbers rather than leaving them in a log line that
    scrolls away.
    """
    plan = fitted.model._plan
    assert plan is not None
    assert plan.images_used == 8 and plan.images_available == 12
    assert plan.patches_kept_per_image == 50 and plan.patches_per_image == 64
    assert plan.candidates_kept == 400
    assert plan.coreset_size == 100
    assert fitted.model._model.memory_bank.shape[0] == 100

    index = load_index(fitted.train_ctx.diagnostics.root)
    table = next(entry for entry in index.entries if entry.key == "memory_bank")
    assert table.kind is DiagnosticKind.TABLE
    assert table.payload is not None
    rows = {row[0]: row[1] for row in table.payload["rows"]}

    assert rows["images used"] == "8 of 12"
    assert rows["patches per image"] == "50 of 64"
    assert rows["bank vectors"] == "100"
    assert rows["backbone"] == "resnet18"
    assert rows["layers"] == "layer2+layer3"
    # The number the caps exist for: what an uncapped run would have held.
    assert rows["uncapped pool would have been"] == "768 vectors"


def test_a_bank_that_fits_uncapped_drops_nothing(tmp_path: Path) -> None:
    """The caps must be bounds, not a tax every run pays.

    A training set small enough to fit under both limits has to produce every patch of
    every image — otherwise the defaults would be quietly discarding data on the datasets
    that need it least.
    """
    rng = np.random.default_rng(2)
    train_ctx, _ = _contexts(tmp_path)
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_write_normal(tmp_path / f"n{i}.png", rng))
        for i in range(4)
    ]

    model = PatchcoreAnomalibModel(_config(max_bank_images=64, max_candidate_vectors=100_000))
    model.fit(train, train_ctx)

    plan = model._plan
    assert plan is not None
    assert plan.images_used == 4
    assert plan.patches_kept_per_image == plan.patches_per_image
    assert plan.candidates_kept == plan.candidates_generated == 4 * 64
    assert plan.images_dropped == 0 and plan.patches_dropped_per_image == 0


# -------------------------------------------------------------------------- the bank


def test_a_saved_model_reloads_to_the_same_scores(fitted: Fitted) -> None:
    fitted.model.save(fitted.train_ctx.artifact_dir)

    reloaded = PatchcoreAnomalibModel(_config())
    reloaded.load(fitted.train_ctx.artifact_dir)
    again = reloaded.predict(fitted.probe, fitted.infer_ctx)

    for before, after in zip(fitted.predictions, again, strict=True):
        assert after.score == pytest.approx(before.score, rel=1e-6)


def test_fitted_bank_exports_with_map_and_paper_score_parity(
    fitted: Fitted, tmp_path: Path
) -> None:
    """The reweighted score is exported as a graph output, not guessed from its map."""
    destination = tmp_path / "patchcore.onnx"
    contract = fitted.model.export_onnx(destination, fitted.infer_ctx.preprocessing)
    fixture = np.linspace(0.0, 1.0, 3 * SIZE * SIZE, dtype=np.float32).reshape(1, 3, SIZE, SIZE)
    expected_map, expected_score = fitted.model.portable_reference(fixture)
    ort: Any = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    values = session.run(None, {contract.input_name: fixture})
    actual_map = np.asarray(values[0])
    actual_score = np.asarray(values[1])

    assert contract.score.kind == "tensor"
    np.testing.assert_allclose(actual_map[0, 0], expected_map, atol=1e-4, rtol=1e-4)
    assert abs(float(actual_score[0]) - expected_score) <= 1e-4


def test_a_changed_backbone_is_refused_by_name(fitted: Fitted, tmp_path: Path) -> None:
    """The weights are an input to the experiment, so a run has to notice them changing.

    timm resolves pretrained weights through a hub that can move underneath a checkout.
    The bank was selected in the old feature space, so distances against a new one are not
    comparable — and the failure mode this guards against is that nothing would say so: the
    checkpoint loads, inference runs, and the number is plausible and wrong.
    """
    fitted.model.save(tmp_path)
    stored = torch.load(tmp_path / STATE_FILENAME, map_location="cpu", weights_only=False)
    stored["backbone_fingerprint"] = "0" * 64
    torch.save(stored, tmp_path / STATE_FILENAME)

    reloaded = PatchcoreAnomalibModel(_config())
    with pytest.raises(RuntimeError, match="resnet18"):
        reloaded.load(tmp_path)


def test_predicting_before_fitting_is_an_error_not_a_zero(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path)
    rng = np.random.default_rng(3)
    record = ImageRecord(image_id=1, sample_id=1, path=_write_normal(tmp_path / "n.png", rng))

    model = PatchcoreAnomalibModel(_config())
    with pytest.raises(RuntimeError, match="before it was fitted or loaded"):
        model.predict([record], infer_ctx)


def test_the_checkpoint_does_not_carry_the_backbone(fitted: Fitted, tmp_path: Path) -> None:
    """A fingerprint instead of 260 MB, and this is the assertion that keeps it that way.

    The bank at this fixture's size is a few hundred kilobytes. If somebody later "fixes"
    the fingerprint check by storing the weights, every experiment in a comparison grows by
    the size of a backbone and this fails loudly rather than the disk filling quietly.
    """
    fitted.model.save(tmp_path)
    stored = torch.load(tmp_path / STATE_FILENAME, map_location="cpu", weights_only=False)

    assert isinstance(stored["backbone_fingerprint"], str)
    assert len(stored["backbone_fingerprint"]) == 64
    assert not any(key.startswith("feature_extractor") for key in stored)
    assert (tmp_path / STATE_FILENAME).stat().st_size < 2_000_000


# ------------------------------------------------------------ M4's views, inherited


def test_every_m4_view_works_with_no_new_code(fitted: Fitted) -> None:
    """ADR-0018's prediction, tested against a method the interface was not designed around.

    M6 checked this for a second EfficientAD, which shares its shape with the first.
    PatchCore trains nothing, has no steps, resumes nothing and holds a bank instead of
    weights — and still has to reach every view through the same self-describing index,
    rendered by `kind` and never by method name.
    """
    index = load_index(fitted.train_ctx.diagnostics.root)
    by_key = {entry.key: entry for entry in index.entries if entry.image_id is None}

    assert by_key["architecture"].kind is DiagnosticKind.GRAPH
    assert by_key["memory_bank"].kind is DiagnosticKind.TABLE
    assert by_key["embedding_pca"].kind is DiagnosticKind.IMAGE
    assert by_key["embedding_magnitude"].kind is DiagnosticKind.MAP
    assert by_key["coreset_coverage"].kind is DiagnosticKind.MAP
    assert all(entry.scope is DiagnosticScope.MODEL for entry in by_key.values())

    per_image = {entry.key for entry in index.entries if entry.image_id is not None}
    assert per_image == {"patch_scores", "map_unblurred"}

    # A colormapped key is drawn on one run-wide scale or two images cannot be compared by
    # eye; an `image` payload is already in [0, 1] and correctly has none.
    assert "coreset_coverage" in index.ranges
    assert "patch_scores" in index.ranges
    assert "embedding_pca" not in index.ranges


def test_the_architecture_tree_has_real_shapes_and_no_invented_wiring(fitted: Fitted) -> None:
    """Nodes from a real forward pass, and an edge list that stays honest.

    `introspect` records nodes finely and leaves wiring to the caller so a view never draws
    what it did not measure. PatchCore is one backbone forward followed by a search that is
    not a module, so there is nothing branch-level to draw and the edge list is empty. That
    is the contract working, not a gap.
    """
    index = load_index(fitted.train_ctx.diagnostics.root)
    payload = next(e for e in index.entries if e.key == "architecture").payload
    assert payload is not None

    nodes = payload["nodes"]
    assert len(nodes) > 20
    assert payload["edges"] == []
    assert payload["total_parameters"] > 0

    executed = [node for node in nodes if node["executed"]]
    assert len(executed) == len(nodes), "features_only prunes unused layers, so all execute"
    assert all(node["output_shape"] for node in executed)
    # The root is the extractor itself, which returns a dict of layer tensors rather than
    # one tensor — the case `introspect._shape_of` learned to read for this method.
    root = next(node for node in nodes if node["parent"] is None)
    assert root["output_shape"] is not None


def test_the_coverage_map_is_the_patch_grid(fitted: Fitted) -> None:
    """The coverage diagnostic is per patch position, so its shape is the grid, not the image."""
    index = load_index(fitted.train_ctx.diagnostics.root)
    entry = next(e for e in index.entries if e.key == "coreset_coverage")
    assert entry.path is not None

    counts = np.load(fitted.train_ctx.diagnostics.root / entry.path)
    assert counts.shape == (SIZE // 8, SIZE // 8)
    assert counts.sum() == pytest.approx(100.0)


def test_the_unblurred_map_is_sharper_than_the_stored_one(fitted: Fitted) -> None:
    """`blur_sigma` is a field, and the diagnostic is what makes its effect visible.

    Blurring cannot raise the maximum of a map, so the unsmoothed one has to peak at least
    as high. Asserting the direction rather than a magnitude keeps this a statement about
    what smoothing does rather than about this fixture's numbers.
    """
    index = load_index(fitted.infer_ctx.diagnostics.root)
    defect_id = fitted.probe[-1].image_id
    entry = next(e for e in index.entries if e.key == "map_unblurred" and e.image_id == defect_id)
    assert entry.path is not None

    unblurred = np.load(fitted.infer_ctx.diagnostics.root / entry.path)
    map_path = next(p.anomaly_map for p in fitted.predictions if p.image_id == defect_id)
    assert map_path is not None
    stored = np.load(map_path)

    assert unblurred.shape == stored.shape
    assert float(unblurred.max()) >= float(stored.max())


# ---------------------------------------------------------------------- the config


def test_a_layer_set_resolves_to_real_layer_names() -> None:
    """The picker's values are the layer names timm is actually asked for."""
    assert LayerSet.LAYER2_LAYER3.layers == ("layer2", "layer3")
    assert LayerSet.LAYER2_LAYER3_LAYER4.layers == ("layer2", "layer3", "layer4")


def test_a_deeper_layer_set_changes_the_embedding_width(tmp_path: Path) -> None:
    """The layer set is not cosmetic: it decides the width of every stored vector.

    Worth pinning because the whole memory calculation is images x patches x **width**, and
    a field that silently did nothing would make the plan's arithmetic a fiction.
    """
    rng = np.random.default_rng(4)
    train = [
        ImageRecord(image_id=i, sample_id=i, path=_write_normal(tmp_path / f"n{i}.png", rng))
        for i in range(4)
    ]

    widths = {}
    for layer_set in (LayerSet.LAYER1_LAYER2, LayerSet.LAYER2_LAYER3):
        ctx, _ = _contexts(tmp_path / layer_set.name)
        model = PatchcoreAnomalibModel(_config(layer_set=layer_set))
        model.fit(train, ctx)
        assert model._plan is not None
        widths[layer_set] = model._plan.embedding_dim

    # resnet18: layer1+layer2 is 64+128, layer2+layer3 is 128+256.
    assert widths[LayerSet.LAYER1_LAYER2] == 192
    assert widths[LayerSet.LAYER2_LAYER3] == 384


def test_downloads_can_be_refused_by_name() -> None:
    """`allow_downloads=False` must fail naming the backbone, not reach for the network.

    A tool that claims to be local-only should not quietly fetch 260 MB, and the error has
    to say which knob turns it back on — the same contract `efficientad_anomalib` keeps for
    its teacher and penalty set.
    """
    model = PatchcoreAnomalibModel(
        _config(backbone="not_a_real_backbone_xyz", pretrained_backbone=True, allow_downloads=False)
    )
    with pytest.raises(RuntimeError, match="not_a_real_backbone_xyz"):
        model._build_model("cpu")
