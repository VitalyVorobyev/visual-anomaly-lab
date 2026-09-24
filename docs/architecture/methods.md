# Methods

Every anomaly-detection method is a plugin behind one interface (ADR-0007). The rest of the application
knows only this interface and the registry key; a new method is one module and one entry in
`models/registry.py`, with no route, schema or TypeScript.

## Plugin interface

```python
# backend/src/anomaly_lab/models/base.py

class Capabilities(BaseModel):
    tasks: list[Task] = [Task.ANOMALY]  # ADR-0039; an experiment is refused for an unlisted task
    requires_training: bool = True      # a fitted memory counts as training
    produces_anomaly_map: bool = True   # drives whether the UI offers overlay controls
    produces_diagnostics: bool = False  # drives the inspector views (diagnostics.md)
    channel_aware: bool = False         # the model *may* read ImageRecord.channel
    dataset_specific: bool = False      # surfaced as a UI warning
    supports_resume: bool = False       # must agree with the SupportsResume protocol
    portable_formats: list[PortableFormat] = []  # only parity-proven exporters (ADR-0034)
    preferred_device: Device = Device.CPU

class ImageRecord(BaseModel):
    image_id: int
    sample_id: int
    channel: str | None        # canonical channel name, None for single-view datasets
    path: Path                 # absolute path to the pinned prepared PNG, read-only

class Prediction(BaseModel):
    image_id: int
    score: float               # higher = more anomalous
    anomaly_map: Path | None   # float32 .npy
    inference_ms: float

class AnomalyModel(ABC):
    title: ClassVar[str]; summary: ClassVar[str]
    @classmethod
    def config_model(cls) -> type[BaseModel]: ...
    @classmethod
    def capabilities(cls) -> Capabilities: ...
    @classmethod
    def availability(cls) -> Availability: ...   # available by default
    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None: ...
    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]: ...
    def save(self, artifact_dir: Path) -> None: ...
    def load(self, artifact_dir: Path) -> None: ...
```

Two optional structural protocols sit beside the ABC rather than on it, so no method carries a stub it
cannot honestly implement:

- **`SupportsResume`** (`completed_steps`, `fit_more`) — continued training ([jobs](jobs.md)). The train
  handler checks it agrees with `supports_resume` and refuses a mismatch by name. A method that cannot
  resume exactly (no optimizer state) refuses rather than restarting its optimizer.
- **`SupportsOnnxExport`** (`deployment/protocol.py`) — a class lists `onnx` only when its fitted
  instance implements this and its graph has passed the generic Python-versus-runtime parity gate. The
  API and UI branch on the capability, never on the registry key ([deployment](deployment.md)). A format
  that holds for only some configurations is not claimed, because the offer is made from the registry
  before any configuration is read.

### Registry and availability

`LOADERS` in `models/registry.py` is a table of **lazy loaders**, so opening the method picker does not
import torch. That holds only while every plugin keeps its heavy imports inside its functions.

A method whose optional dependencies (the `dl` extra) are missing reports `availability.available = false`
with the command that installs them, and is **listed, not hidden**, so "why can't I pick this method" is
answerable from the screen.

### Schema-driven configuration

`config_model()` returns a pydantic model exposed as JSON Schema at `GET /api/experiments/model-types`.
The experiment form is rendered from that schema — types, defaults, bounds, descriptions — so a new
hyperparameter needs no frontend change ([frontend](frontend.md)). A default lives in Python alone: an
untouched field is sent as unset.

### Contexts

`TrainContext` and `InferContext` inject everything a plugin must not invent for itself:

- `artifact_dir` — the experiment's directory, the only place a model writes its own outputs;
- `cache_dir` — shared app-managed storage for downloaded assets (weights, the ImageNette penalty set),
  reused by every experiment of a method;
- `preprocessing` — the prepared size and colour policy (below);
- `progress(fraction, message)`, `metric(name, value, step)`, `log` — become job events
  ([jobs](jobs.md));
- `should_cancel()` / `raise_if_cancelled()` — cooperative cancellation, polled at batch boundaries;
- `emit_diagnostic(...)` — the [diagnostics](diagnostics.md) contract (ADR-0018);
- `TrainContext.val` — held-out normals, empty when the split has no `val` subset;
- `TrainContext.targets` — the only path ground truth takes into a plugin (ADR-0039). It is `None` for
  `anomaly`, so an anomaly method cannot see a defect mask by construction. For a targeted task it is a
  `TargetProvider`: `label_key`, and `mask(image_id)`, the class's region as a boolean array in the
  prepared frame (`experiments/targets.py`);
- `InferContext.write_map(image_id, array)` — projects a prepared-frame map to source coordinates,
  persists it as float32 and accumulates the run's finite display range;
- `InferContext.write_mask(image_id, mask)` — a method's own foreground decision, projected nearest
  into the source frame and stored as a 0/255 PNG beside the map (`maps/<id>.mask.png`). For a
  targeted task the map is the foreground probability, and a run that writes no mask is read by
  thresholding it.

**What `fit` is given is the task's** (`experiments/policy.py`). `anomaly` fits on the train subset's
normals and calibrates on the val subset's. `few_shot_segmentation` fits on every train-subset image whose
truth answers for the target class, present or absent (`annotations/class_truth.py`), and has no val. The
handler logs how many images it left out and why.

Models never touch SQLite, never read application settings, and never write outside `artifact_dir` and
`cache_dir`, which is what makes a plugin testable with a `NullReporter` and no job system.

## Preprocessing

**Spatial input is configuration of the experiment, not of the model.** A comparison means something only
if both methods saw the same pixels.

Every `Experiment` pins a complete `RegionProfileRevision` build by profile id and manifest digest.
`SpatialTransform` records a clipped half-open source crop, an integer contain-resize and symmetric edge
padding. Every plugin receives the build's lossless prepared PNG paths and decodes them through
`models/preprocessing.load_array`, which applies the colour policy and verifies the frozen size but never
resizes. A model that opens another path is a bug.

Plugins emit maps in prepared coordinates; `InferContext.write_map` projects them through the image's
recorded transform, so stored maps, source masks and overlays share source coordinates. Pixels outside the
crop are `NaN`: rendered transparent, and kept by evaluation in the denominator at the score floor with
their defect/normal counts reported — a crop cannot improve its metric by hiding a defect.

**Backbone standardization belongs to the model.** ImageNet normalization is the network's first layer
(`efficientad_nets.imagenet_normalize` inside `forward`; `patchcore_anomalib` applies `IMAGENET_MEAN` /
`IMAGENET_STD` from `preprocessing.py` itself, because anomalib puts it in a Lightning pre-processor these
wrappers do not use). Unnormalized pixels do not fail — they score features from outside the backbone's
distribution.

**`expand_planes` is on the model's side too.** Under `color=grayscale`, `load_array` returns one plane and
each three-channel backbone calls `expand_planes(chw, 3)` itself. Expanding in `load_array` would make
`grayscale` and `rgb` identical for a mono file and give `pixel_reference` three copies of every image.
Exported ONNX graphs replicate internally, so a bundle's declared input is `preprocessing.channels`.

### Region extractors

Localisation has its own lazy `RegionExtractor` registry (`regions/registry.py`) rather than being a model
option. An extractor receives one source RGB array and returns one source pixel-edge box or an explicit
failure; its pydantic schema drives the client as model schemas do.

- `identity` — the full-source control;
- `center_crop` — a fixed fractional window around a configured centre, content-free by design and the
  only extractor identical across the channels of one grouped sample by construction; content-based
  extraction can disagree between channels and misregister per-position channel fusion;
- `foreground_threshold` — border-estimated background, absolute-contrast threshold on a bounded grid,
  largest connected component;
- `mobile_sam` — the verified TinyViT checkpoint with a bounded automatic prompt grid, largest mask
  within area/quality limits. It tries MPS and falls back to CPU after an MPS runtime failure, reporting
  the chosen device as extractor metadata.

Extractor confidence is method-specific and not comparable between entries. Preview samples at most 24
images evenly and writes no pixels. A full build is one cancellable `region_prepare` job that writes to a
job-specific staging directory, records successes and failures in a deterministic JSON-lines manifest, and
publishes atomically. A build is immutable; rebuilding needs a new profile revision. Each entry pins the
source digest, realised transform, extractor metadata and prepared-image digest. The dataset's
**Prepare** screen overlays each crop and can switch to the prepared pixels. The default-profile verdict
is in [measurements](../measurements.md).

## Seeding

A `seed` field that does not control the result puts unattributable noise under every comparison. Seeds
must reach **every** random stream, including those libraries hide: scikit-learn's
`SparseRandomProjection` draws from numpy's global RNG, which `torch.manual_seed` does not touch, and
torch's global stream drives weight init. Every method pins all its streams, and its tests assert
reproducibility in both directions — same seed identical, different seed different.

## Memory planning

Anything whose cost is linear in the dataset and whose value saturates is capped, sampled with
`models.base.evenly_spaced` (never the first N), and the cap is logged with what was dropped.

For memory-bank methods the bound is the design. A plan is resolved **before** the pass by a pure,
torch-free function — `plan_bank` (PatchCore) and `plan_memory` (`dino_memory`) — whose `describe()` goes
into the job log, so the footprint is known before it is paid for and the arithmetic is tested by the
torch-free CI job. Caps compose in a fixed order: **images (units) first, then patches within each
surviving image**, because patches inside one image overlap and are redundant while images differ by
what the process varies.

Greedy coreset selection is quadratic in the candidate pool. The loop lives in `models/coreset.py`
(torch-only, anomalib-free, so in-house methods reuse it), reports progress and honours cancellation, and
keeps anomalib's projection, distance and iteration order —
`test_dl_patchcore.py::test_the_greedy_selection_matches_anomalib` pins identical indices. It runs on
**CPU** even when the backbone is on MPS: a tight loop of small kernels is dispatch-bound. Footprint
figures are in [measurements](../measurements.md).

## Device policy

`preferred_device` is where the tensor work goes: `mps` for the deep methods, `cpu` for `pixel_reference`
(ADR-0008). The device resolves at job start with a CPU fallback when MPS is unavailable or an operator is
missing, and is recorded in the job log. A stage inside a method may be placed elsewhere when a smoke
test (`scripts/mps-smoke-test.py`, `scripts/patchcore-smoke-test.py`,
`scripts/dino-memory-smoke-test.py`) says so; nothing in the application reveals a mis-placed stage,
because the run finishes with correct numbers either way.

**A probe runs before a plugin is written** (ADR-0008). Before wrapper or method code targets a new
library, or a new stage targets the accelerator, a standalone `scripts/<name>-smoke-test.py` exercises it;
what it finds (an operator missing on MPS, a stage faster on CPU, a footprint) becomes the plugin's
defaults, device placement and caps.

## Downloaded assets and licences

Anything downloaded that a number depends on is an input to the experiment and belongs in its
configuration. Pretrained weights are somebody's training run: two published EfficientAD teachers share
architecture and shapes and still differ element-wise, so `teacher_source` is a field. `allow_downloads`
refuses a fetch by name. URLs are pinned to a commit and checksummed, so an upstream change fails naming
the file.

A backbone too large to store per experiment travels as a **sha256 fingerprint** in the checkpoint, and
`load` refuses a mismatch naming the backbone — a bank selected in one feature space is meaningless
against other weights.

**A licence is not a config field.** The two DINOv3 entries of the shared `DinoBackbone` table
(`models/dino_backbone.py`) resolve to gated weights; access reaches the method as an ambient `HF_TOKEN`,
never as a form field (which would store a credential in the database, logs and exports).
`dino_backbone._gated_failure_message` turns a 401 into which weights are gated, where to request access,
and which ungated Apache-2.0 encoders need no account. The token value is never read, stored or printed.
Defaults are ungated, and `test_models.py` asserts which entries are.

## Per-image scores

> **Models emit per-image scores. Cross-channel aggregation belongs to the [evaluation layer](evaluation.md).**

A model may read `ImageRecord.channel`, but it returns one `Prediction` per input image, and no model
decides how a part's views combine into a sample verdict. A channel-aware method that is asked to score a
channel it was not fitted on **refuses by name**; falling back to another channel's reference would
produce a confident map of the difference between two illuminations.

## Anomaly maps

Maps are stored as **float32 `.npy`** — lossless, the source of truth for statistics and pixel metrics.
The API renders colormapped PNGs on demand at `GET /api/images/{image_id}/anomaly-map?experiment_id=…`
and caches them. Overlay opacity is applied in CSS, never baked into the image ([media](media.md)).

## Diagnostics

A method declaring `produces_diagnostics` pushes entries into a self-describing index the UI renders by
`kind`, never by method name. The whole contract is on **[diagnostics](diagnostics.md)**.

## Frozen-DINO building blocks

Shared by every method that reads a frozen DINOv2/DINOv3 encoder, so two methods that differ in what they
do with patch features do not also differ in how they got them.

- `dino_backbone.image_patch_features` is the one encoding path: pixels through `load_array`, ImageNet
  standardisation, the chosen blocks with each layer L2-normalised on its own, then the concatenation
  normalised, returned as `(N, P, D)` on the CPU.
- `positional.py` is INSID3's positional debiasing. `dino_backbone.noise_patch_features` passes one seeded
  standard-normal image at the prepared size through the encoder. The top `s` right singular vectors of its
  `(P, D)` features span where position lives, and `debias` projects features onto their complement and
  renormalises. INSID3 fixes `s = 500` for ViT-L. The rank is clipped to half of `min(P, D)`, because a
  small grid would otherwise lose every direction the noise spans.
- `refine.py` brings a patch-grid probability to prepared-pixel resolution: `bilinear`, or `guided` (He et
  al.'s guided filter, with the grey image as guide), which is edge-preserving smoothing. It concentrates a
  transition on the image's own edge and keeps the local mean in flat regions; it is not a threshold. numpy
  only.

## Shipped methods

| key | family | trains | resume | channel-aware | ONNX | device |
| --- | --- | --- | --- | --- | --- | --- |
| `pixel_reference` | per-pixel median/MAD | fit | no | yes | yes | cpu |
| `efficientad_custom` | student–teacher + autoencoder | yes | yes | no | yes | mps |
| `patchcore_anomalib` | coreset memory bank | fit | no | no | yes | mps |
| `dinomaly_custom` | feature reconstruction | yes | yes | no | no | mps |
| `glass_anomalib` | learned anomaly synthesis | yes | yes | no | yes | mps |
| `dino_memory` | frozen DINO patch memory | fit | no | yes | no | mps |
| `subspace_ad` | PCA residual over frozen DINO | fit | no | yes | no | mps |
| `color_prototype` | few-shot: fg/bg colour Gaussians | fit | no | no | no | cpu |
| `fss_dino` | few-shot: FSSDINO prototypes + Gram | fit | no | no | no | mps |

`color_prototype` and `fss_dino` declare `few_shot_segmentation` alone; every other method declares
`anomaly`. Gate verdicts for each are in [measurements](../measurements.md).

### `pixel_reference`

The floor: numpy + Pillow, no torch. A per-pixel median and MAD over the training normals; a test image
becomes a z-map, smoothed, and scored by a percentile.

- `reference_scope` — `channel` (default) fits one reference per channel; `dataset` pools. Pooling
  illuminations inflates the MAD with the difference *between* channels and flattens real defects to noise.
- `max_reference_images` (128), `smoothing_sigma`, `score_percentile` (99.5), `mad_floor`.
- ONNX: one static graph carries one reference, so `export_onnx` refuses a multi-reference bank by name
  inside the plugin.

### `efficientad_custom`

EfficientAD, implemented in-house (`efficientad_custom.py` for config, loop and checkpoint;
`efficientad_nets.py` for modules). A PDN student distils a frozen pretrained teacher, plus an autoencoder
branch; defaults reproduce the published algorithm, and each departure is a field (ADR-0028).

- `model_size` (`small`/`medium`), `max_steps`, `learning_rate`, `weight_decay`, `seed`.
- `teacher_source` — `nelson1425` (default), `distilled` (a teacher produced by a `distill` job, named in
  `distilled_teacher`; see [teacher distillation](https://github.com/VitalyVorobyev/visual-anomaly-lab/blob/main/book/src/teacher.md)), or `anomalib` (kept so
  recorded runs stay reproducible, listed last). Assets live in `efficientad_assets.py` under separate cache
  subdirectories. The positional nelson1425 layout is mapped by order of appearance with every shape
  validated, and `test_our_pdn_is_the_reference_pdn` pins the architecture against the reference source.
- `use_penalty`, `hard_quantile`, `student_teacher_weight`, `score_reduction` / `score_top_k`.
- **A distilled teacher is a model asset, not an experiment.** The `distill` job (`models/distill.py`)
  trains a PDN by MSE against a frozen source patch-aggregated to the PDN's output width and grid; the
  source is a `FeatureSource` (`models/teacher_distill.py`), of which `WideResNet101Source` is the only
  implementation. It is resumable, and writes weights, the source's feature-normalization statistics and
  its config. `fit_more` refuses a checkpoint whose recorded teacher differs from the configured one.
- Bounds: `quantile_images` and `quantile_pixel_budget` cap the score-normalization fit;
  `calibration_holdout` fits it on held-out normals. 256 px is a hard floor, checked before training.
- The diagnostic keys are a frozen set the views depend on,
  `test_dl_efficientad_custom.py::test_the_diagnostic_keys_are_the_ones_the_views_expect`.

### `patchcore_anomalib`

anomalib's PatchCore: pretrained backbone features in a coreset memory bank; nothing is trained, and
there are no steps to continue.

- `backbone` (`wide_resnet50_2`), `layer_set`, `pretrained_backbone`, `allow_downloads`, `num_neighbors`,
  `blur_sigma`, `seed` (coreset start, random projection, both RNG streams).
- Bounds: `max_bank_images` caps the backbone pass, `max_candidate_vectors` the store and the quadratic
  selection; `coreset_ratio` sets the bank size. `plan_bank` prints both before the pass.
- The checkpoint `patchcore.pt` stores the bank and a backbone fingerprint, not the backbone.

### `dinomaly_custom`

Dinomaly, implemented in-house: a frozen DINO encoder, a trainable bottleneck and decoder that reconstruct
the encoder's grouped intermediate layers, and a map from the reconstruction error.
`dinomaly_custom.py` holds config, plan, pass and checkpoint; `dinomaly_nets.py` the modules, hard-mined
loss, StableAdamW and map rule. **Neither imports anomalib.**

- `encoder` — any `DinoBackbone` entry (default DINOv2 ViT-S/14-reg4). Encoder tokens come from
  `dino_backbone.extract_layer_tokens`: no final norm, prefix tokens kept, no per-layer L2 normalisation.
- `decoder_depth` (2–12, default 8) — `fuse_groups(count)` derives the decoder's fusion groups; the
  encoder always contributes eight target layers.
- `max_steps` (5000, a fixed horizon) with a 100-step warm-up to `learning_rate` and cosine decay;
  `batch_size`, `weight_decay`, `hard_mining_fraction`, `dropout`, `map_blur_sigma`,
  `pretrained_encoder`, `allow_downloads`, `seed`.
- Score: the map is resampled to 256², Gaussian-smoothed (σ = 4) and the hottest one percent averaged.
- `DinomalyNet.trainable_parameters` takes the bottleneck and decoder by name, so the frozen encoder never
  reaches the optimizer.
- Bit-exact pins (`atol=0`) against the shared backbone and anomalib's step and map are replaced, never
  loosened, when a divergence is chosen deliberately.
- ONNX: not yet; the parity gate is on [backlog.md](../backlog.md).

### `glass_anomalib`

GLASS: anomalib's network, Perlin local synthesis, Gaussian global synthesis with gradient-ascent mining,
losses and map rule, driven by this plugin's finite batch-1 loop. The pretrained WRN-50 is frozen; only the
projection and discriminator train. **Experimental**: it missed its image-level gate
([measurements](../measurements.md)).

- `max_steps`, `mining_steps`, `learning_rate`, `allow_downloads`, `seed`.
- `synthesis_anchor` — the generic replacement for upstream's per-category `svd` switch; no dataset name
  enters the method.
- Bounds: the global-synthesis centre is computed over `center_images` normals (evenly spaced) rather than
  the whole training set, and refreshed on an absolute `center_refresh_steps` schedule, so a continuation
  refreshes exactly where an uninterrupted run would. The checkpoint carries both optimizers, the centre,
  the synthesis and image-order RNG states and the completed step.
- The Describable Textures Dataset is not a dependency; built-in Perlin synthesis needs no corpus.
- ONNX: the graph embeds ImageNet normalization, projection and discriminator and emits both the map and
  GLASS's own image score; runtime parity pins both.

### `dino_memory`

A frozen DINOv2/DINOv3 encoder whose L2-normalized patch features for the training normals become a
memory; a test patch scores by its distance to it (ADR-0037). Nothing is trained.

- `backbone` (default ungated DINOv2 ViT-S/14-reg4), `layers` (`last_two`), `pretrained_backbone`,
  `allow_downloads`, `blur_sigma`, `score_percentile`, `feature_batch_size`, `seed`.
- `scoring` — one enum, because a layout × distance product has an invalid cell (a global Gaussian models
  the dataset's marginal, not normality):

  | value | the memory | sees |
  | --- | --- | --- |
  | `global_knn` | one coreset bank over all positions | position-blind (PatchCore's rule) |
  | `local_knn` | one bank per patch position, searched over `window_radius` | a normal pattern in the wrong place |
  | `local_gaussian` | one shrunk Gaussian per position, Mahalanobis distance | PaDiM's rule |

  `test_a_per_position_bank_finds_what_a_global_bank_explains_away` keeps the axis honest.
- `local_knn` loops over `(2r+1)²` offsets with one einsum each rather than materialising every window;
  a border position falls back to its own offset.
- `local_gaussian` fits a covariance of `mahalanobis_dims` from `per_position_images` samples — singular in
  the ordinary case, recorded as `MemoryPlan.sample_deficit`. `shrinkage` is Ledoit-Wolf or a ridge whose
  `ridge_epsilon` is a fraction of the trace (an absolute ridge would swamp unit-vector covariances and turn
  Mahalanobis into Euclidean). The fit runs in CPU float64.
- `channel_fusion` — `per_image` (default) pools every channel image into one memory and scores each image;
  `feature_concat` banks a **sample**: per-channel vectors concatenated in `channel_order` (sorted names)
  and re-normalized, so each channel contributes `1/√C` — the fused squared distance is the *mean* of the
  per-channel ones, so the score scale does not depend on channel count — and a one-channel group equals
  `per_image`. `channel_aware` is declared unconditionally. A sample
  missing or adding a channel refuses by name. `diagnose` scores a single record, so a multi-channel
  `feature_concat` model refuses a single-image diagnosis (backlog).
- Bounds: `plan_memory` derives grid and width from arithmetic before the encoder is built, then the first
  batch verifies both. `max_bank_images`, `per_position_images`, `max_candidate_vectors`, `coreset_ratio`.
  Neighbour selection runs on CPU (MPS `topk` is slower and breaks ties differently); only the encoder
  forward uses MPS.
- ONNX: none, unconditionally — `feature_concat` has no single-input graph, and the export offer is read
  from the registry before any config exists.

### `subspace_ad`

The same frozen encoders asked a different question: patch tokens are mean-pooled over a band of blocks,
PCA is fitted to the normals, and a patch scores the squared residual the leading subspace cannot
reconstruct. A fitted model is a mean and an orthonormal basis. **Experimental**: its defaults come from a
sweep run outside the application (ADR-0038) and its promotion gate is open.

- `backbone` (default DINOv2 ViT-L/14), `pretrained_backbone`, `allow_downloads`, `seed`.
- `layers` — a `LayerWindow` (default `upper_half`) expressed as a fraction of depth
  (`dino_backbone.LayerBand`), because a fixed block count picks a different window on every encoder.
- `variance` (0.99) and `tail_fraction` (the image score is the mean of the top fraction of patch scores),
  `smoothing_sigma`. The arithmetic in `models/subspace.py` and `models/score_map.py` is shared with the
  sweep, so a stored fit answers every `variance` by a `cumsum` and every `tail_fraction` by a sort.
- `rotations` (30) random rotations per training image with `rotation_fill` for the invented corners;
  `rotations=0` for parts whose orientation carries meaning.
- Bounds: `max_fit_images` (16, evenly spaced, drops logged); a fit costs `(1 + rotations)` forwards per
  image.
- One subspace per channel; an unfitted channel refuses by name. It needs pretrained weights — a random
  encoder gives it no meaningful variance directions — so plugin tests cover plumbing and accuracy is
  asserted in `test_subspace_ad_math.py`.
- ONNX: none.

### `color_prototype`

The few-shot floor (ADR-0040): numpy + Pillow, no torch. The references' target masks split their pixels
into the class and everything else, one Gaussian is fitted to the colour of each, and a query pixel's map
value is the class's posterior under equal priors. The presence score is a high percentile of that map. It
writes no mask, so the evaluator cuts the map at its rule.

- `color_space` — `lab` (default) or `rgb`; a grey image is modelled on intensity either way.
- `max_pixels_per_class` (100 000), sampled with `evenly_spaced` across the references and logged when it
  bites. There is no RNG, so the same references always give the same models.
- `smoothing_sigma`, `presence_percentile` (99.9).
- It refuses to fit without targets, or when the references hold no pixel of the class or of the
  background.
- It knows only colour. It proves the task's slice in the torch-free CI job, and a deep method that does
  not beat it has learned nothing about shape or texture.
- ONNX: none.

### `fss_dino`

A reproduction of FSSDINO (arXiv 2602.07550) for one class, training-free, on the shared encoding path.
The references' last-block patch features are split by their masks: a patch is the class when at least
half of it is covered after bilinear downsampling. Each side gets `prototypes_per_class` (5) prototypes by
seeded cosine k-means, and a Gram matrix. A query patch's cosine map per prototype, and its Gram energy
(min-max normalised per image), are upsampled; each side combines them as `mean * max`, and a pixel goes
to the higher side. The arithmetic is `models/prototypes.py`, in numpy.

- It writes that argmax as its own mask. Its map is the foreground's share of the two scores, floored at
  zero, so `>= 0.5` agrees with the argmax. Presence is a high percentile of the map; the paper has none.
- The default encoder is the ungated DINOv2 ViT-B/14. The paper's DINOv3 ViT-B/16 is one field away and
  licence-gated. The prepared size is the experiment's, not the paper's 512 px.
- `max_features_per_class` (20 000) caps k-means and the Gram matrix, sampled evenly and logged. `seed`
  reaches k-means and, without pretrained weights, the encoder; the tests assert both directions.
- The encoder is not saved. `load` restores the prototypes, and the first prediction refuses an encoder
  whose fingerprint moved.
- Accuracy is the public gate's question; the plugin tests run a seeded random ViT. ONNX: none.

### `classical_circular` (optional, not built)

The one method allowed to assume the showcase dataset's geometry, declared `dataset_specific = True`: a
circle fit on the part boundary with a prior-based fallback, geometry shared across a sample's channels, a
polar transform about the centre, FFT angular correlation for orientation, a per-channel median/MAD
reference, and a percentile of the per-pixel z-score as the score. CPU only.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
