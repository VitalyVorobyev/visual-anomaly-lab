# Model plugin interface

Every anomaly-detection method is a plugin behind one interface (ADR-0007). The rest of the application knows
only this interface and the registry key.

```python
# backend/src/anomaly_lab/models/base.py

class Capabilities(BaseModel):
    requires_training: bool          # PatchCore/EfficientAD yes; a pure-reference method may say no
    produces_anomaly_map: bool       # drives whether the UI offers overlay controls
    produces_diagnostics: bool       # drives whether the UI offers the inspector views (ADR-0018)
    channel_aware: bool              # model consumes channel metadata internally
    dataset_specific: bool           # True for classical_circular — surfaced as a UI warning
    preferred_device: Literal["cpu", "mps", "cuda"]


class ImageRecord(BaseModel):
    image_id: int
    sample_id: int
    channel: str | None              # canonical channel name, None for single-view datasets
    path: Path                       # absolute path to the pinned prepared PNG, read-only


class Prediction(BaseModel):
    image_id: int
    score: float                     # higher = more anomalous
    anomaly_map: Path | None         # float32 .npy written into ctx.artifact_dir / "maps"
    inference_ms: float


class AnomalyModel(Protocol):
    @classmethod
    def config_model(cls) -> type[BaseModel]: ...
    @classmethod
    def capabilities(cls) -> Capabilities: ...

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None: ...
    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]: ...
    def save(self, artifact_dir: Path) -> None: ...
    def load(self, artifact_dir: Path) -> None: ...


# The registry is a table of *lazy loaders*, so opening the method picker does not import
# torch. That only holds while each plugin keeps its heavy imports inside its functions.
LOADERS: dict[str, Callable[[], type[AnomalyModel]]] = {
    "pixel_reference":       _pixel_reference,       # numpy + Pillow, the floor
    "efficientad_anomalib":  _efficientad_anomalib,  # the wrapper, and the baseline (ADR-0029)
    "efficientad_custom":    _efficientad_custom,    # ours, same interface (ADR-0008)
    "patchcore_anomalib":    _patchcore_anomalib,    # bounded memory-bank reference
    "dinomaly_anomalib":     _dinomaly_anomalib,     # transformer-reconstruction reference
    "glass_anomalib":        _glass_anomalib,        # experimental learned synthesis
    # "classical_circular":  M8, optional (ADR-0015)
}
```

A method whose optional dependencies are missing reports `availability.available = false` with the
command that installs them, and is **listed rather than hidden**: "why can't I pick EfficientAD"
should be answerable from the screen.

## Schema-driven configuration

`config_model()` returns a **pydantic model**, which the API exposes as JSON Schema at
`GET /api/experiments/model-types`. The frontend renders the experiment configuration form **directly from
that schema** — field types, defaults, ranges and descriptions all come from the Python side. Adding a
hyperparameter to a model therefore requires no frontend change at all, which is what makes "add a method
without touching the rest of the app" true in practice rather than aspirational.

## Contexts

`TrainContext` and `InferContext` carry everything a long-running plugin needs and must not invent for itself:

- **`artifact_dir`** — the experiment's directory; the only place a model writes its own outputs;
- **`cache_dir`** — shared, app-managed storage for downloaded assets (pretrained weights, the ImageNette
  penalty set). Separate from `artifact_dir` because these belong to the *method* and are reused across every
  experiment that runs it — a copy per run would be absurd;
- **`preprocessing`** — the prepared size and colour policy every method is made to share (below);
- **`progress(fraction, message)`** — forwarded to the job event stream ([the job system](jobs.md));
- **`metric(name, value, step)`** — scalar series; becomes a `metric` event, which is what per-epoch losses use;
- **`should_cancel()` / `raise_if_cancelled()`** — cooperative cancellation, polled at batch boundaries;
- **`log`** — structured logger whose records become `log` events in the job stream;
- **`emit_diagnostic(...)`** — the diagnostics contract (below, ADR-0018);
- `TrainContext.val` — the held-out normals, **empty when the split has no `val` subset**;
- `InferContext.write_map(image_id, array)` — projects through the pinned image transform, persists one
  source-frame float32 map and accumulates the finite run display range.

Everything above is *injected* (ADR-0014). Models never touch SQLite, never read application settings, and
never write outside `artifact_dir` and `cache_dir` — which is also what makes a plugin unit-testable with a
`NullReporter` and no job system at all.

## Spatial input is configuration of the experiment, not of the model

A comparison between two methods only means something if they were shown the same pixels. Left to themselves
the libraries disagree, and the resulting difference in AUROC would partly measure the resize.

Every `Experiment` pins a complete `RegionProfileRevision` build by profile id and manifest digest.
`SpatialTransform` records a clipped half-open source crop, the actual integer contain-resize, and symmetric
edge padding. All plugins receive the build's lossless prepared PNG paths and decode them through
`load_array`; that function applies colour policy and verifies the frozen size but is forbidden to resize.
A model that opens another path is a bug, not a variation.

Plugins emit maps in prepared coordinates. The injected `InferContext.write_map` projects them through the
image's recorded transform before persistence, so stored maps, source masks and UI overlays share source
coordinates. Pixels outside the selected crop are `NaN`: rendering makes them transparent, while evaluation
keeps them in the denominator at the score floor and reports their defect/normal counts. A crop cannot improve
its metric by hiding a defect. There is no mixed rollout and no method-specific localisation code.

## Region extractor plugins

Spatial localisation has its own lazy `RegionExtractor` registry rather than becoming a model option. An
extractor receives one source RGB array and returns one source pixel-edge box or an explicit failure. Its
pydantic configuration schema drives the client exactly as model schemas do; adding an extractor is one
module and one registry entry. The three current entries represent the value test rather than three promises
of equal quality:

- `identity` is the full-source control;
- `foreground_threshold` estimates background luminance from the border, thresholds absolute contrast on a
  bounded analysis grid, and returns the largest connected component;
- `mobile_sam` uses the verified TinyViT checkpoint and a bounded automatic prompt grid, then selects the
  largest mask inside configured area/quality limits. Torch and MobileSAM imports remain inside construction.

Extractor confidence is method-specific and cannot be compared between registry entries. Profile preview
reports coverage, box geometry, runtime and failures; it does not turn those confidence values into a shared
score they are not.

Preview selects at most 24 images evenly across the dataset and writes no prepared pixels. Full build is a
single cancellable `region_prepare` job: it writes into a job-specific staging directory, records successful
and failed images in a deterministic JSON-lines manifest, then atomically publishes the build. A completed
materialisation is immutable; rebuilding requires a new profile revision. Each successful entry pins the
source digest, realised transform, extractor metadata and prepared-image digest. The summary retains a bounded
visual-audit sample and failure examples;
the dataset-local **Prepare** screen overlays each source crop and can switch to the materialised pixels.
MobileSAM attempts MPS for the first real image and transparently reconstructs on CPU after an MPS runtime
failure; that chosen device is reported as extractor metadata rather than hidden.

### Standardizing for a backbone is the model's business, not the bridge's

The bridge decides the pinned spatial artifact and colour, and stops there. What a method then does with those pixels is
part of the method: `efficientad_nets.imagenet_normalize` applies ImageNet statistics inside `forward`, and
that is not a second preprocessing, it is the network's first layer.

The seam matters because **the two libraries put it in different places**, and one of them is invisible when
it is wrong. anomalib's EfficientAD normalizes inside the model, so the wrapper hands it `load_array`'s
`[0, 1]` array unchanged. anomalib's PatchCore does **not** — `Patchcore.configure_pre_processor` puts the
`Normalize` in the Lightning pre-processor, which none of these wrappers use — so `patchcore_anomalib`
applies it itself, from `IMAGENET_MEAN` / `IMAGENET_STD` in `preprocessing.py`. Feeding an ImageNet backbone
unnormalized pixels does not fail: it runs, produces maps, and quietly scores features from outside the
distribution the backbone was trained on.

## Bounding a memory-bank method

`pixel_reference` caps how many normals build its median and `efficientad_*` cap how many fit their
quantiles. Both are one number over a pass whose value saturates. **PatchCore is the case where the bound is
the design**, because its cost is not a step budget at all — it holds every training patch at once and then
runs a selection whose loop count is the size of the bank.

Measured by `scripts/patchcore-smoke-test.py` at 256×256 with `wide_resnet50_2` and `layer2+layer3`, read
from a real forward pass rather than derived from an assumed stride:

| quantity | measured |
| --- | --- |
| patch grid / embedding width | 32×32 = 1024 patches, 1536 dims |
| per image | 6.29 MB |
| a ~900-image VisA class | 921 600 patches = **5.66 GB** before any selection |
| backbone forward | **7.0 ms/image on MPS**, 19.3 on CPU |
| greedy iteration at N=100 000 | 7.16 ms on MPS, **2.46 ms on CPU** |
| greedy total at `coreset_ratio` 0.1 | 1.2 s at N=25 000, 24.6 s at 100 000, 150.8 s at 250 000 |
| nearest-neighbour search, 10 000-vector bank | 14.5 ms/image on MPS, 13.1 on CPU |

Three things follow, and each is a rule rather than a tuning.

**Two independent caps, applied in a fixed order.** `max_bank_images` bounds the backbone pass;
`max_candidate_vectors` bounds the store and, with it, the selection — whose total cost is *quadratic*, since
both the iteration count and the work per iteration grow with N. Images are dropped first and patches thinned
second, never the reverse: patches inside one image overlap through the 3×3 pooling and are largely
redundant, while two images differ by whatever the process actually varies. `plan_bank` resolves both before
the pass and its numbers go into the job log, so a footprint is known before it is paid for. It is pure and
torch-free for the same reason `introspect.build_tree` is: the arithmetic that decides whether the machine
survives is checked by the CI job that has no torch.

**The selection runs on the CPU even when the rest runs on MPS.** The loop is one norm over a tall thin
tensor, an argmax and a scatter — too little arithmetic to cover per-iteration dispatch — so the device that
wins the forward pass loses the selection threefold. Nothing in the application would have shown this: the
run finishes and the bank is correct, it is merely three times slower than it needed to be.

**The loop itself is ours, and the rule is not.** anomalib's `KCenterGreedy` returns when it is finished and
not before, so a job reports no progress and ignores cancellation for as long as it runs. Every other long
operation here stops within one step. So the projection and the distance stay anomalib's, the iteration order
is identical, and `test_dl_patchcore.py::test_the_greedy_selection_matches_anomalib` pins that the same
features and the same starting point produce the same indices.

### A seed only means something if it reaches every stream

Writing that pin found the bug worth finding. `SparseRandomProjection` draws its sparsity pattern through
scikit-learn's `sample_without_replacement`, whose `random_state` defaults to `None` — **numpy's global RNG,
which `torch.manual_seed` does not touch** — and anomalib constructs it with no `random_state` at all. Its
coreset is therefore not reproducible: same seed, same data, a different memory bank, and nothing anywhere
saying so.

This is M6's finding arriving in a different library. There it was weight initialisation drawing from torch's
global stream, so `seed` controlled the training order over *different* initial weights and two runs of one
configuration were not one experiment. The shape is the same and so is the consequence: a `seed` field on the
experiment form that does not control the result puts unattributable noise under every comparison built on
it. `patchcore_anomalib` pins both streams, and a test asserts the bank is identical across two fits at one
seed and different at another — both directions, because pinning a seed is only meaningful if changing it
changes the answer.

## A reconstruction method needs a fixed training horizon

`dinomaly_anomalib` freezes a small registered DINOv2 encoder and trains anomalib's bottleneck and
eight-layer decoder against normal feature maps. Its plugin owns the bounded batch-1 loop while anomalib
owns the network, loss, optimizer and anomaly-map arithmetic. Prepared dimensions must be divisible by
the encoder's 14-pixel patch size; the plugin rejects an incompatible experiment before allocating the
model.

The exposed controls are deliberately small: finite steps, seed and the first-download policy. Decoder
depth is fixed at eight because anomalib's fusion topology indexes eight outputs even when its constructor
accepts a shallower value. Learning rate follows one fixed 5,000-step warm-cosine schedule independent of
where a run is paused. Consequently, 1,000 steps plus a 1,000-step continuation is the same experiment as
2,000 uninterrupted steps, rather than a schedule whose past changes when the user resumes it.

The checkpoint stores the bottleneck, decoder, optimizer, CPU/MPS random streams, NumPy image-order stream
and completed step. The frozen encoder remains in the app-managed asset cache and is represented by its
name, exact tensor fingerprint and dependency versions. Reload refuses a changed encoder rather than
silently attaching old decoder weights to a different feature space. Public VisA evidence and the exact
resource protocol are recorded in [`measurements-dinomaly.md`](../measurements-dinomaly.md).

## A learned-synthesis method needs a bounded reference frame

`glass_anomalib` keeps anomalib's GLASS network, Perlin local synthesis, Gaussian global
synthesis, gradient-ascent mining, losses and anomaly-map rule. The plugin supplies the
finite batch-1 loop, cancellation, shared prepared pixels and exact continuation. The
pretrained WRN-50 is frozen; only the feature projection and discriminator are optimized.

Global synthesis is defined relative to a centre in the projected normal-feature space.
Upstream recomputes that centre over the entire training loader at every epoch boundary,
which makes an otherwise finite step budget hide a dataset-sized pass. Here
`center_images` bounds the pass and `evenly_spaced` preserves acquisition-range coverage;
the plan prints both retained and omitted counts before loading torch.
`center_refresh_steps` is an absolute step schedule. A continuation at step 650 therefore
uses the saved centre until the same next refresh as an uninterrupted run, while a
continuation exactly on a refresh boundary recomputes it. The checkpoint carries both
optimizers, the current centre, CPU/MPS synthesis RNG, NumPy image-order stream and the
completed step; a test pins 2+1 steps to the exact state of three uninterrupted steps.

The optional Describable Textures Dataset is not an implicit dependency. Built-in Perlin
synthesis is the first default and needs no corpus. DTD has a named public URL and SHA-256
in the measurement record, but becomes app-managed storage only if a paired public-data
ablation shows value. Likewise, the upstream per-category `svd` switch is exposed only as
the generic `synthesis_anchor` experiment control; no dataset name enters the method.
Resource evidence and the external-asset policy are recorded in
[`measurements-glass.md`](../measurements-glass.md). The bounded paired public gate missed
its image-level quality floor, so the method remains available as an explicitly experimental
comparison rather than the recommended learned-synthesis reference.

## A downloaded asset can be a hyperparameter

EfficientAD cannot train from a dataset alone: it needs a **pretrained teacher**, distilled from a
WideResNet on ImageNet. That looks like a fixed public constant, and it was treated as one. It is not.

A teacher is the *output of somebody's distillation run*, and two people who each did that honestly
ship different weights. Measured: anomalib's `pretrained_teacher_small.pth` and the teacher bundled
with [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD) have identical architecture
and identical tensor shapes, and differ element by element by up to **1.4 in absolute value**. They
are two different networks, not one file under two names — and the second is the one whose repository
reports reproducing the paper (MVTec AD 99.1, VisA 98.2).

So which teacher is loaded is **configuration of the experiment**, `teacher_source`, and both URLs
live in `efficientad_assets.py` under separate cache subdirectories so a run can be repeated against
either without a refetch. Two consequences worth stating:

- **The two published files are keyed differently**, because the two reference codebases build the
  same network differently: anomalib names its layers (`conv1.weight` …) and nelson1425 builds an
  `nn.Sequential`, so its file is keyed by *position* (`0`, `3`, `6`, `8` — the gaps are ReLUs and
  pools). `load_pdn_weights` maps the positional layout **by order of appearance**, never by parsing
  the indices, and validates every shape before loading. A shuffled mapping would load without
  complaint and produce a plausible, wrong teacher.
- **Matching shapes is not the same claim as being the same network**, so the architecture itself is
  pinned: `test_our_pdn_is_the_reference_pdn` builds the reference's `nn.Sequential` from its own
  source, loads one set of weights into both, and compares outputs — for both widths and with
  padding on and off. Two networks can share every parameter shape and differ in where the ReLUs and
  pools sit, and that mistake is invisible from the outside. The same test pins the one genuine
  difference: **ours standardizes with ImageNet statistics inside `forward` and the reference does it
  in its dataset transform**. Same function, different seam — and a teacher fed unnormalized pixels
  would also load without complaint.
- **The reproduction's URL is pinned to a commit**, not to `main`. An asset that changes upstream
  must be a checksum failure naming the file, not a silent change of teacher between two runs that
  read as comparable.

This is the general shape, not a special case: anything downloaded that a number depends on is an
input to the experiment, and belongs in its configuration where the comparison screen can show it.

**PatchCore inherits the whole argument.** Its backbone is timm's pretrained `wide_resnet50_2`, resolved
through the HuggingFace hub and equally somebody's training run, so `backbone` and `pretrained_backbone` are
configuration and `allow_downloads` refuses a fetch by name. The difference is what gets stored: EfficientAD's
teacher is small enough to live in the checkpoint, while a backbone is 260 MB per experiment and a comparison
holds several. So `patchcore.pt` carries a **sha256 fingerprint** of the backbone instead, and `load` refuses
a mismatch naming the backbone. The bank was selected in the old feature space, so distances against new
weights are not comparable — and without the check, the checkpoint would load, inference would run, and the
number would be plausible and wrong.

## Diagnostics

A method that declares `produces_diagnostics` can show what it did as well as what it concluded, by
pushing into a self-describing index the UI renders **by `kind`, never by method name**. That is one
capability flag and one context method; the whole contract — authoring, the two read paths, the
architecture tree, on-demand entries and deletion — is on its own page:
**[diagnostics](diagnostics.md)**.

## Contract: scores are per-image

> **Models emit per-image scores. Cross-channel aggregation belongs to the evaluation layer ([the evaluation layer](evaluation.md)).**

This is the seam that keeps evaluation model-independent. A model *may* use channel metadata internally —
`ImageRecord.channel` is provided, and the classical baseline relies on it to keep one reference statistic per
channel (ADR-0010) — but it still returns one `Prediction` per input image. No model decides how a part's
three views combine into a sample-level verdict; that policy lives in one place and is applied identically
to every method.

## Anomaly map storage

Anomaly maps are written as **float32 `.npy`** arrays — the source of truth, lossless, directly usable for
recomputing statistics or (later) pixel-level metrics against masks. The API renders **colormapped PNGs on
demand** at `GET /api/images/{image_id}/anomaly-map?experiment_id=…`, caching the rendered PNG. Overlay
opacity is applied in CSS by the UI, never baked into the served image, so the opacity slider is instant and
requires no server round-trip.

## Device policy

Defaults target Apple Silicon: `preferred_device = "mps"` for the DL adapters, `"cpu"` for the classical
baseline (ADR-0008). Device is resolved at job start with a graceful fallback to CPU when MPS is unavailable
or an operator is unimplemented, and the resolved device is recorded in the job log.

**A method's preferred device is not necessarily right for every stage inside it.** `patchcore_anomalib`
prefers MPS and keeps its backbone there, and pins its coreset selection to the CPU regardless — measured
three times faster, because a loop that does too little arithmetic per iteration is dominated by dispatch
rather than by compute. The rule this generalizes to: `preferred_device` is where the *tensor work* goes, and
a tight Python-driven loop over small kernels is a reason to measure rather than to inherit.

## Classical baseline

`classical_circular` is the non-neural reference method. It was originally planned as the vertical slice's
first model, on the grounds that it needs no training infrastructure, no GPU and no external framework. That
ordering has been **superseded**: making the *showcase-specific* method the first one contradicted the
universal goal, so the slice is now proven with a dataset-agnostic method and a dataset-agnostic floor
baseline, and this method is scheduled later as an optional milestone. In outline (ADR-0010): a **circle
fit** on the part boundary with a **prior-based fallback** when the fit is poor; the resulting geometry is
**shared across all channels of a sample**, since the views are near-simultaneous images of the same physical
object; a **polar transform** about the fitted centre turns rotation into translation; **FFT angular
correlation** recovers orientation; a **per-channel median/MAD reference** is built from the training normals;
and scoring is a **percentile of the per-pixel z-score** map. It runs in seconds per sample on CPU. This
method is explicitly `dataset_specific = True` (above) — it is showcase-dataset-specific (circular parts),
exploiting the part's circular geometry, which the deep methods must not. The full algorithm, its parameters
and its failure modes are in **ADR-0010**.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
