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
    score: float               # higher = more anomalous; for segmentation, the share given a class;
                               # for detection, the highest confidence
    anomaly_map: Path | None   # float32 .npy
    label_map: Path | None     # semantic segmentation: 8-bit class-index PNG (write_label_map)
    instances: Path | None     # object detection: boxes as JSON (write_instances)
    inference_ms: float

class AnomalyModel(ABC):
    title: ClassVar[str]; summary: ClassVar[str]
    @classmethod
    def config_model(cls) -> type[BaseModel]: ...
    @classmethod
    def capabilities(cls) -> Capabilities: ...
    @classmethod
    def availability(cls) -> Availability: ...   # available by default
    @classmethod
    def check_input(cls, config, preprocessing) -> None: ...   # any input by default
    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None: ...
    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]: ...
    def save(self, artifact_dir: Path) -> None: ...
    def load(self, artifact_dir: Path) -> None: ...
```

**`check_input` refuses at creation what could only fail at fit.** `create_experiment` and the reference
studio's preview call it with the frozen config and prepared size, and a `ValueError` becomes a 422 that
names the reason. The frozen-DINO methods use it for a patch size the prepared frame does not divide.

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
- `TrainContext.label_targets` — its sibling for `semantic_segmentation`, `None` for every other task: a
  `LabelTargetProvider` with the run's pinned `classes` and `labels(image_id)`, a `uint8` label map in the
  prepared frame where `classes[i]` is `i + 1`, 0 is background and `IGNORE_INDEX` (255) is letterbox
  padding or a class the run did not pin — fitted on by nobody;
- `TrainContext.box_targets` — the third sibling, for `object_detection`: a `BoxTargetProvider` with the
  pinned `classes` and `boxes(image_id)`, the image's object instances of those classes as `TargetBox`es
  (`label_key`, `box`) in the prepared frame, clipped to the region crop. Boxes are pixel-edge
  `(x0, y0, x1, y1)` with `x1`/`y1` exclusive. An empty list is a confirmed absence;
- `InferContext.write_map(image_id, array)` — projects a prepared-frame map to source coordinates,
  persists it as float32 and accumulates the run's finite display range;
- `InferContext.write_mask(image_id, mask)` — a method's own foreground decision, projected nearest
  into the source frame and stored as a 0/255 PNG beside the map (`maps/<id>.mask.png`). For a
  targeted task the map is the foreground probability, and a run that writes no mask is read by
  thresholding it.
- `InferContext.write_label_map(image_id, labels, classes=n)` — a supervised segmentation method's class
  per pixel, checked to lie in `0..n`, projected **nearest** into the source frame (background outside
  the crop) and stored as an 8-bit PNG (`maps/<id>.labels.png`). A class index is a name and is never
  interpolated. The `score` beside it is the share of the prepared image given a class other than
  background, so every segmentation method's score means the same thing and needs no calibrated
  probability.
- `InferContext.write_instances(image_id, instances, classes=keys)` — a detection method's
  `PredictedInstance`s (`label_key`, `box`, `confidence`), in the prepared frame. Each is checked — a
  pinned class, a finite confidence, a box with area — and projected into the source frame through the
  pinned transform (`SpatialTransform.project_box`), clipped to the crop; a box wholly in the letterbox
  padding is dropped. Stored most confident first as `maps/<id>.instances.json`, an empty list
  included. More than `MAX_INSTANCES_PER_IMAGE` (100, COCO's cap) is refused rather than trimmed: a
  method keeps its most confident. The `score` beside it is the highest confidence, 0 when it found
  nothing.

**What `fit` is given is the task's** (`experiments/policy.py`). `anomaly` fits on the train subset's
normals and calibrates on the val subset's. `few_shot_segmentation` fits on every train-subset image whose
truth answers for the target class, present or absent (`annotations/class_truth.py`), and has no val.
`semantic_segmentation` and `object_detection` fit on every train-subset image whose truth answers for
all of the run's pinned classes, and have no val. The handler logs how many images it left out and why.

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
- `center_crop` — a fixed fractional window around a configured centre, content-free by design and so
  identical across the channels of one grouped sample by construction; content-based extraction can
  disagree between channels, which a profile's `sample_alignment = union` resolves by giving the whole
  sample the union of its crops ([domain model](domain-model.md#regionprofilerevision));
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
- `dino_backbone.FrozenEncoder` builds a method's encoder once and stores its fingerprint; after `load`,
  the first rebuild must match it. The weights themselves are never saved with a fitted model.
- `refine.py` brings a patch-grid probability to prepared-pixel resolution: `bilinear`, or `guided` (He et
  al.'s guided filter, with the grey image as guide), which is edge-preserving smoothing. It concentrates a
  transition on the image's own edge and keeps the local mean in flat regions; it is not a threshold. numpy
  only.

### Calibrating the foreground probability

A few-shot method's map is a probability by construction — a softmax share, a posterior under equal
priors, a ratio of two scores — and not by measurement, so the evaluator's fixed `>= 0.5` cut means
something different on every class. `calibration.py` is the shared fix, a `calibration` field on
`color_prototype`, `fss_dino` and `proto_seg`:

- **`leave_one_out`** holds each reference out in turn, fits the method on the others, and scores the one
  it left out. The held-out (probability, truth) pixel pairs are pooled, at most 262 144 of them, evenly
  spaced within each reference and shared equally between them, and the cap is logged. A Platt scale —
  logistic regression on the probability's logit, with Platt's smoothed targets so perfectly separated
  pairs stay finite — is fitted to them, and the log says where 0.5 now falls on the unscaled map.
- **The fitted model does not change.** The final bank is built first from the seed alone; each fold
  draws its own stream from `[seed, fold]`. Only the scale differs from an uncalibrated run, and it is
  saved with the model (a checkpoint without one loads as unscaled).
- **Monotone by refusal.** A scale whose slope is not positive, or pairs of one class only, leave the
  probability unscaled and say so. The scale is applied to the whole map and to the presence score after
  that score is computed, so neither pixel ranking nor presence ranking can move.
- **One reference cannot be left out**, and a fold whose other references show no pixel of one side — an
  absent reference beside a single present one — is skipped and counted. With no fold scored, the
  probability stays unscaled.
- **The references' prior, not the queries'.** Every fold is fitted on references, which usually all show
  the class; a query set that is mostly absent sees a scale fitted where the class is common.

The default is `none` for all three, by the few-shot gate's calibration leg
([measurements.md](../measurements.md)): calibrated, absent images stop being flagged and IoU rises, but
on a small defect class the scaled map clears 0.5 on too few of the images that show it.

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
| `proto_seg` | few-shot: debiased prototype bank / probe | fit | no | no | no | mps |
| `color_classifier` | segmentation: per-class colour Gaussians | fit | no | no | no | cpu |
| `dino_linear_seg` | segmentation: softmax head on frozen DINO | yes | no | no | no | mps |
| `color_detector` | detection: colour components, boxed | fit | no | no | no | cpu |
| `dino_linear_det` | detection: DINO linear head, components boxed | yes | no | no | no | mps |

`color_prototype`, `fss_dino` and `proto_seg` declare `few_shot_segmentation` alone, `color_classifier`
and `dino_linear_seg` declare `semantic_segmentation` alone, `color_detector` and `dino_linear_det`
declare `object_detection` alone, and every other method declares `anomaly`. Gate verdicts for each are in [measurements](../measurements.md).

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
- `calibration` — `none` (default) or `leave_one_out`
  ([calibrating](#calibrating-the-foreground-probability)); the colour models of each fold are
  refitted from the other references' pixels.
- It refuses to fit without targets, or when the references hold no pixel of the class or of the
  background.
- It knows only colour. It proves the task's slice in the torch-free CI job, and a deep method that does
  not beat it has learned nothing about shape or texture.
- ONNX: none.

### `color_classifier`

The supervised segmentation floor (ADR-0039): numpy + Pillow, no torch, sharing `color_prototype`'s
colour model. Background and each pinned class get one Gaussian over the colour of their training
pixels; a pixel's class is the highest posterior under equal priors, after each posterior is smoothed.
It writes the label map, a map of the probability of anything but background, and the foreground-share
score.

- `color_space` — `lab` (default) or `rgb`.
- `max_pixels_per_class` (100 000), sampled with `evenly_spaced` across the whole pool of each class's
  training pixels and logged when it bites. Two passes — count, then read only the chosen pixels — keep
  memory bounded by the cap rather than the training set. There is no RNG.
- `smoothing_sigma` (1.0), applied to each class's posterior before the argmax.
- A class with no training pixel is not modelled, never predicted, and named in a warning. It refuses to
  fit without label targets, or when the training images hold no pixel of any class.
- ONNX: none.

### `color_detector`

The detection floor (ADR-0039): `color_classifier` read as boxes, numpy + Pillow, no torch. Training
paints each truth box's interior with its class and everything outside every box as background — a
pixel is inside when its centre is, and a smaller box is painted over a larger one — and fits
`color_classifier`'s Gaussians to that. At inference every 8-connected component of a class's argmax is
one detection: its tight box, and as confidence the mean smoothed posterior of the class over it. It
writes the boxes, the probability of anything but background as its map, and the top confidence as its
score.

- The `color_classifier` fields, plus `min_area` (4 prepared pixels; a smaller component is not a
  detection) and `max_detections` (100, the most confident kept).
- A box is not an outline, so each class's colour model also learns the background its boxes enclose.
  It knows only colour: touching objects of one class are one detection. What a deep detector has to
  beat, not a candidate.
- Components use `eval/pixel.py`'s union-find, a Python loop over foreground pixels, through
  `component_boxes`, which `dino_linear_det` decodes with too.
- It refuses to fit without box targets. ONNX: none.

### `dino_linear_seg`

The first deep supervised segmentation method (ADR-0039): a softmax classifier over background and the
pinned classes — a 1x1 convolution on the patch grid — on the shared frozen encoding path, fitted through
`label_targets`. It writes the label map, the probability of anything but background as its map, and the
foreground-share score, like `color_classifier`.

- **Trained at pixels, predicted at pixels.** The head is linear, so the logits of an interpolated
  feature are the interpolated logits. A sampled pixel's training feature is the patch grid interpolated
  at that pixel (`pixel_features`, which reproduces `refine.upsample` exactly — the torch-free test pins
  it), and prediction upsamples the grid's logits bilinearly before the argmax. The loss is taken on the
  function the label map is drawn from, and a pixel is never simply given its patch's label.
- **Bounded before encoding.** `plan_pixels` divides `max_training_pixels` (131 072) among the training
  images, at most `pixels_per_image` (1 024) labelled pixels each; images are dropped, evenly spaced,
  only when there are more of them than the total, since each keeps a pixel; the plan and its float32
  footprint are logged, and a plan above 4 GiB is refused naming the knobs. `IGNORE_INDEX` pixels are
  never sampled, and only the chosen images are encoded.
- **`pixel_sampling` decides how an image's budget is spent.** `raster` spaces it evenly over the
  image's labelled pixels whatever their class, which gives a defect covering a fraction of a percent of
  the frame a pixel or two. `per_class` (default) has `allocate_pixels` split the budget equally among the
  classes present in the image, visiting the smallest first so that a class with fewer pixels than its
  share takes them all and leaves the rest to the others, and `sample_pixels` spaces each class's take
  evenly over that class's own pixels. Nothing is random either way. The fit logs the rule and, per
  class, the pixels taken against the labelled pixels the chosen images held. On its own `per_class`
  measured lower than `raster` on defect IoU: with thousands of defect pixels under `inverse_frequency`,
  the head moves its boundary further into background and labels a few percent of every image defect.
  With `held_out_iou` the fitted constant pulls that boundary back, and the defect's larger sample is
  what the constant is fitted on ([measurements](../measurements.md)).
- **The head fits on the CPU** with AdamW (`learning_rate` 1e-3, `weight_decay` 1e-4) for `epochs` (10)
  over shuffled minibatches of `batch_size` (1 024), with cross-entropy weighted by `class_balancing`:
  `inverse_frequency` (default) gives every sampled class the same total weight, `none` counts every pixel
  once. Weighting and sampling do different work: per-class sampling balances the classes *within* an
  image, and the weights balance what remains *across* images, because an image without the class still
  contributes its budget to background. Either way `inverse_frequency` trains the head as if every class
  were equally common. One seeded generator draws the initial weights and the batch order, so nothing
  depends on torch's global stream; the tests assert the seed in both directions. The per-epoch loss is a
  metric.
- **`logit_bias` is one constant per class, added to the logits before the argmax**, because a head
  trained as if every class were equally common labels a class that covers a fraction of a percent of the
  pixels as readily as the background around it. The head is fitted identically whatever the field says;
  the constants are saved beside it, and a checkpoint asked for a constant it was not fitted with refuses
  by name. `none` is the head's own answer. `training_prior` is the standard logit adjustment
  (`prior_shift`): `log p_c − log q_c`, with `p` the class's share of the chosen images' labelled pixels
  and `q` its share of the fit's loss weight (sampled count times class weight), which makes the argmax
  the Bayes answer for pixel accuracy — and on a VisA defect, at a shift near −8, that answer is almost
  never the class ([measurements](../measurements.md)). `held_out_iou` fits the constant for the measure
  the task is read by: the training images with sampled pixels are split into `BIAS_FOLDS` (3) folds by
  position, a head fitted on the other folds (same rule, a seed derived from `seed` and the fold) scores
  each fold's sampled pixels, and `pixel_weights` makes each pixel stand for `available / sampled` pixels
  of its class in its image, so the pooled sample reads as the training frames themselves; it is the
  default.
  `fit_class_bias` then visits each class but background once, holding the constants already fitted,
  sorts the held-out margins and takes the cut with the highest weighted IoU; a class no cut overlaps
  keeps zero and says so. It costs `BIAS_FOLDS` more head fits on the CPU and no second encoding pass.
  Both are torch-free and tested without the `dl` extra. `guided` refinement filters probabilities after
  the constant, so the fitted cut is exact only under `bilinear`.
- `layers` defaults to the last block; `refine` is `bilinear` (the function the head was trained on) or
  `guided`, which filters each class's probability against the image before the argmax.
- A class with no sampled pixel is never predicted and is named in a warning. It refuses to fit without
  label targets, or when the sampled pixels hold no class at all. The encoder is not saved: `save` writes
  the head (each file whole, then renamed into place) and the encoder's fingerprint, and `load` refuses
  a different backbone or layer set.
- **Supported**, by the logit-bias gate: under `per_class` sampling and `held_out_iou` it beat the floor by
  the predeclared margin on both VisA classes, where neither pixel sampling alone had
  ([measurements](../measurements.md)). The mask it draws of a VisA defect is usable, not good. ONNX:
  none.

### `dino_linear_det`

The first deep detector (ADR-0039), and `dino_linear_seg` read as boxes the way `color_detector` is
`color_classifier` read as boxes. Training paints box interiors exactly as the floor does
(`PaintedBoxes`) and fits `dino_linear_seg`'s head on that — its pixel plan, logged before anything is
encoded, its per-class sampling, its seeded CPU fit and its `held_out_iou` constant, all unchanged and
all its fields. At inference the head's probabilities (`DinoLinearSegModel.probabilities`, the function
the segmenter's label map is drawn from) are argmaxed, and `component_boxes` makes every 8-connected
component of a class one detection with the mean probability of its class as confidence. It writes the
boxes, the probability of anything but background as its map, and the top confidence as its score.

- **Why components and not box regression.** It adds no trained part the workbench has not measured: the
  head cleared its segmentation gate and the decoding is the floor's, so the detection gate reads one
  difference — frozen DINO features against colour. It keeps the floor's weakness: touching objects of
  one class are one detection, and a box is as tight as the component the upsampled logits draw. A
  regression head with non-maximum suppression is the next step if the gate says the features are worth
  it.
- `held_out_iou` fits each class's constant for the IoU of painted box interiors — the pixel-level shadow
  of a box's IoU. AP depends only on how detections rank, and a component's mean class probability puts
  a confident, separated region above a faint one.
- `min_area` (4 prepared pixels) and `max_detections` (100), as for the floor.
- It refuses to fit without box targets. `save` writes the head's files and a class list, each whole or
  not at all; the encoder is not saved and `load` refuses a different backbone or layer set.
- **Experimental** until the public detection gate. ONNX: none.

### `fss_dino`

A reproduction of FSSDINO (arXiv 2602.07550) for one class, training-free, on the shared encoding path.
The references' last-block patch features are split by their masks after bilinear downsampling
(`prototypes.split_patches`): a patch is the class when half of it is covered, or when it is the most covered
patch of a region smaller than that, which small defects are at ViT patch sizes; it is background only when
the mask does not touch it, so mixed patches teach neither side. Each side gets `prototypes_per_class` (5) prototypes by
seeded cosine k-means, and a Gram matrix. A query patch's cosine map per prototype, and its Gram energy
(min-max normalised per image), are upsampled; each side combines them as `mean * max`, and a pixel goes
to the higher side. The arithmetic is `models/prototypes.py`, in numpy.

- It writes that argmax as its own mask. Its map is the foreground's share of the two scores, floored at
  zero, so `>= 0.5` agrees with the argmax. Presence is a high percentile of the map; the paper has none.
- `calibration` — `none` (default) or `leave_one_out`
  ([calibrating](#calibrating-the-foreground-probability)). Calibrated, the mask is the scaled map at
  0.5 rather than the argmax.
- The default encoder is the ungated DINOv2 ViT-B/14. The paper's DINOv3 ViT-B/16 is one field away and
  licence-gated. The prepared size is the experiment's, not the paper's 512 px.
- `max_features_per_class` (20 000) caps k-means and the Gram matrix, sampled evenly and logged. `seed`
  reaches k-means and, without pretrained weights, the encoder; the tests assert both directions.
- The encoder is not saved. `load` restores the prototypes, and the first prediction refuses an encoder
  whose fingerprint moved.
- Accuracy is the public gate's question; the plugin tests run a seeded random ViT. ONNX: none.

### `proto_seg`

Ours (ADR-0040), training-free by default, built from the shared blocks so that each design choice is a
field the gate can measure. It is the default few-shot method by the public gate's predeclared rule
([measurements.md](../measurements.md)):

- **Debiased features** (`positional_debias`, `positional_rank` 500 as in INSID3, clipped and logged)
  from `layers` (default the last two blocks).
- **A hybrid bank per side**: the mean direction plus `clusters_per_class` (8) seeded cosine k-means
  prototypes. Both sides get the same cluster count, because the LSE score sums over a side's prototypes
  and would otherwise favour the side that has more.
- **LSE scoring**: a patch's probability is the class's share of a softmax over every prototype at
  `temperature` (0.1).
- **`adaptation`**: `training_free` scores with the bank; `linear_adapt` fits a class-balanced,
  L2-regularised logistic probe on the references' patches by deterministic full-batch descent, and scores
  with it.
- **`refine`**: `guided` (default) or `bilinear`, from the patch grid to pixels.
- **Presence**: the mean of the `presence_patches` (4) most confident patches.
- **`calibration`**: `none` (default) or `leave_one_out`
  ([calibrating](#calibrating-the-foreground-probability)); each fold rebuilds the bank, and the probe
  under `linear_adapt`, from the other references' patches.
- It writes no mask, so the evaluator cuts its map at 0.5. `max_features_per_class` (20 000) bounds the
  bank and the probe. The seed reaches k-means, the noise image and, without pretrained weights, the
  encoder.
- No mask of a small defect it draws is usable yet; the gate that made it the default says so
  ([measurements.md](../measurements.md)). ONNX: none.

### `classical_circular` (optional, not built)

The one method allowed to assume the showcase dataset's geometry, declared `dataset_specific = True`: a
circle fit on the part boundary with a prior-based fallback, geometry shared across a sample's channels, a
polar transform about the centre, FFT angular correlation for orientation, a per-channel median/MAD
reference, and a percentile of the per-pixel z-score as the score. CPU only.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
