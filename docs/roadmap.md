# Roadmap

**visual-anomaly-lab** is a local desktop workbench for training, evaluating and comparing visual
anomaly-detection methods on arbitrary image datasets, validated against public benchmarks whose
published numbers can be checked against our own (**ADR-0015**). The private showcase dataset —
images of a manufactured circular part — is one reference dataset among them, and only the classical
baseline is showcase-specific (**ADR-0010**).

Delivery order: plan first, then one working vertical slice end to end with a single method, then
depth and the remaining methods behind the same interface. **Nothing from M5 on may change the
application outside the model plugin boundary** (**ADR-0007**); if it does, the slice was not
actually vertical.

Sizing is honest, not aspirational: one developer plus Claude Code on an Apple Silicon Mac, evenings
and weekends. Milestones are sequential, and the [backlog](backlog.md) is re-triaged after each one.

**Completed milestones below are summaries.** Each keeps only what still constrains future work; the
narrative of how each finding was reached lives in the git history and in the ADRs it produced.

---

## Done

### M0 — Repo safety + foundation docs · complete 2026-08-06

Layered ignore rules, `scripts/check-repo-safety.sh`, and ADRs 0001–0011 written before any
application code, so the vertical slice was implemented against a decided design.

**Still binds:** the private-data arrangement was reconsidered in **ADR-0022** — source images now
live *outside* the working tree, so the guards are defence in depth rather than the only control.

### M1 — Walking skeleton · complete 2026-08-06

Tauri shell spawning a FastAPI sidecar on an OS-chosen port, announced back over stdout; health
screen; the job queue and its WebSocket; CI.

**Still binds:** the browser path is first class — the same UI runs against a manually started
backend, and that is where most debugging happens.

### M2 — Import + browse · complete 2026-08-06

Two-phase import (scan → reviewable manifest → commit), the media tier cache, sample browsing and
labelling, seeded sample-level splits.

**Still binds:** import runs on the *same* job machinery as training, which is why training needed no
second progress mechanism. Adapter option forms are generated from each adapter's JSON Schema.

### M3 — Universal vertical slice · complete 2026-08-07

The loop closes: import a directory tree or a public benchmark, browse and label it, adopt or draw a
split, train, score, and read image- and pixel-level metrics with a working overlay. Two methods
ship: `pixel_reference` and `efficientad_anomalib`. The diagnostics contract (**ADR-0018**) and the
evaluation layer (**ADR-0011**, **ADR-0017**) both date from here.

**Still binds:**

- **CI installs without `--extra dl`.** The torch-free environment demonstrably runs ruff, format,
  `mypy --strict` and the suite, so the `dl` boundary is measured rather than asserted. Anything
  added to the evaluation path must stay torch-free. Found when CI ran for the first time, having
  been dead since M1 behind a tag that resolved to nothing — the lesson being that "verification is
  green" was asserted from local runs while the authoritative one was red.
- **Score normalization moves ROC-AUC**, despite the metric being threshold-free. EfficientAD scores
  `max_p [ w_st·map_st[p] + w_ae·map_stae[p] ]` and the two weights come from *different* quantile
  pairs, so the fit sets their **ratio** — the relative influence of the two branches before the max
  — and images reorder. Measured at a fixed 4 000 steps on a byte-identical test set: fitting the
  quantiles on 90 genuinely held-out normals rather than on the training normals moved sample
  ROC-AUC 0.744 → **0.769**, pixel ROC-AUC 0.853 → **0.873**, AU-PRO 0.571 → **0.595** — and that
  understates it, since the holdout also costs 90 training images.
- **The EfficientAD wrapper does not use Lightning.** `EfficientAd.on_train_start` reads
  `self.trainer.datamodule`, so the Lightning path means adopting anomalib's preprocessing and
  breaking the property that makes any comparison meaningful. anomalib supplies the architecture,
  losses, maps, pretrained teacher and statistics; the training loop is ours. The cost is that our
  loop can drift from theirs; the benefit is that cancellation lands within one step — and that
  warm-starting a run is tractable at all.
- **The published EfficientAD figure was not reproduced.** The gap is measured and broken down into
  what was ruled out and what was left untested by decision, rather than asserted away. Default
  `max_steps` is 4 000 against the paper's 70 000, which any comparison against a published number
  inherits.

### M4 — The researcher's workbench UI · complete 2026-08-07

Architecture view from a real forward pass, teacher inspector, live training charts, benchmark
charts, per-branch diagnostic overlays. Every view renders by diagnostic `kind` and never by method
name, so a future method inherits all of them (**ADR-0018**). Charts are hand-rolled SVG, not a
library.

**Still binds:**

- **`GET /api/jobs/{id}/metrics` replays scalar series from the job log** (**ADR-0020**), because
  `metric` events are streamed and stored in no column. The read is linear in the log's size on
  every request — the first thing to look at if a comparison view ever asks for several runs at once.
- **Val AUROC per epoch was not delivered, by decision.** `TrainContext.val` is a bare sequence with
  no labels and the train handler filters it to normals (**ADR-0011**), so `roc_auc` correctly
  returns `None`. Delivering it means giving `TrainContext` labelled validation data, which is an
  **ADR-0007** change and a decision of its own. In the backlog.
- **Pixel-level ROC and PR curves cannot be drawn.** The pixel accumulator streams its histograms and
  discards them by design (**ADR-0017**), so the curve would mean re-reading every anomaly map. The
  benchmark tab says so on screen. The pixel ROC-AUC and AU-PRO themselves are unaffected.

### M4.5 — The UI/UX pass · complete 2026-08-07

Unplanned, taken before M5 on the grounds that a comparison screen built in the old idiom would have
to be rebuilt in the new one. A semantic token layer, light and dark themes with a three-state
toggle, a primitive set under `components/ui/`, and a visible focus ring the application did not
previously have anywhere (**ADR-0021**).

**Still binds:** the chrome carries no saturation, so the data can be loud — full colour belongs to
charts and images. `enum` is read before `type` and `$ref` resolves through `$defs`, so every
pydantic shape reaches the right control. **An empty control means unset**, and defaults live in
Python alone.

### M4.6 — Reachability · complete 2026-08-08

Unplanned, taken before M5 because the workbench had built views nobody could reach. A finished run
did not refresh its own screen, so the Benchmark tab stayed disabled and the run list stayed on
`queued` until the user navigated away and back. Everything else followed from looking at the
running application.

What shipped: a **Samples** gallery ranking every scored sample as a picture, filterable by outcome
(*mistakes* = FP + FN); the sample page rebuilt around a full-height canvas with prev/next, zoom and
pan; an **anomaly segmentation** overlay that can be laid against the ground-truth outline; the
artifact directory surfaced with a *Reveal in Finder* shell capability (**ADR-0014**'s anticipated
second one); and the per-image diagnostics budget spread across the run instead of taking its first
twelve images.

**Still binds:**

- **A working overlay can read as a missing feature.** `alpha_follows_score` is correct and is what
  keeps a map from tinting the whole photograph, but the run-wide high end is set by the hottest
  image in the run, so a mid-scoring image is legitimately almost transparent. The segmentation
  layer and the printed scale exist because of this.
- **A segmentation cut is a *display* decision** in map units, taken as a fraction of the run-wide
  range so it is the same cut on every image. It feeds no metric, and is deliberately never
  conflated with the sample-score threshold or with the pixel metrics, which integrate over every
  threshold.
- **A rank order can be inverted and look right.** "Most anomalous" opened the grid on the cleanest
  samples in the run for as long as it took to read the scores under the tiles. Pinned by a test.
- **`openapi-typescript` emits a property with a literal `default` as required**, which forces every
  caller to restate it — which is how the frontend came to pin a value that silently overrode the
  Python default for two milestones. `default_factory` keeps a default in Python and off the wire.

---

### M4.7 — The workbench you can iterate in · complete 2026-08-08

Unplanned, taken before M5 for the third time in a row, and for the same reason each time: looking at
the running application. M4.6 made the workbench's output *reachable*; running it showed that the
workbench was not something you could **iterate in**. Training restarted from zero. Per-branch
diagnostics existed only for the images an inference job happened to sample. The architecture view
was three boxes. The Overview tab opened on a job list and a log tail, with the metrics below the
fold.

What shipped: **Overview** rebuilt around results, metrics and configuration, with runs, logs and
files moved to a **Jobs & files** tab and the run controls to a bar above every tab; precision,
recall and F1 drawn against the threshold beside its slider; thumbnails instead of paths in the
verdict table. The sample viewer returns to the tab it came from, reports the **map and source
values under the cursor** (**ADR-0023**), and zooms anchored on the pointer with a fit toggle. The
Architecture tab shows the **real module hierarchy** from forward hooks in a shared helper
(**ADR-0024**). Training can be **continued** for further steps from a checkpoint carrying the
optimizer, the schedule, the step counter and both RNG streams (**ADR-0025**). And any scored image
can be **diagnosed on demand** from a resident inference worker (**ADR-0026**), whose output is
first-class in the index and deletable (**ADR-0027**).

**Still binds:**

- **A resident inference process is kept off the accelerator by a lock, not by a check.** `evict` and
  a request take the same lock, and the job queue awaits `before_spawn` before starting a worker, so
  the two cannot coexist. A job can therefore be delayed by one in-flight request, and that delay
  *is* the guarantee — do not make the hook non-blocking to avoid it.
- **`max_steps` is a per-run budget, so the learning rate visibly goes back up at a resume point.**
  Both facts are deliberate and printed on screen. Continuing through disk is bit-identical to
  continuing in process; 10 + 10 is *not* 20, and a test asserts that difference so nobody later
  mistakes it for a bug and removes it.
- **`named_modules()` sees modules, not wiring.** `F.relu`, `torch.cat` and arithmetic inside
  `forward` are invisible, so `nodes` is fine-grained while `edges` stays branch-level. The caption
  says so: the UI must not draw wiring it did not measure.
- **Two producers write the diagnostics index, and every merge rule is scoped by `origin`.** A run's
  per-image entries are a *sample* and supersede each other wholesale; an on-demand entry is a
  question somebody asked. Neither may sweep the other away, and an on-demand emission may only
  *widen* a display range. Counting the two together yields a number above the stated budget, which
  reads as a cap that does not work.
- **A status check is not a check when it runs in a threadpool.** `current_job_id` had to become a
  claim held across the whole of execution: the window between the pre-spawn hook finishing and the
  row being written was one in which a request would see an idle queue and start a resident against
  a worker that was already launching.
- **`mypy --strict src` is not what CI runs.** `pyproject` sets `files = ["src", "tests"]`, and
  checking only `src` is how five type errors in a test file reached CI. Run bare `uv run mypy`.
- **The `dl`-gated tests now run in their own CI job**, at about eight minutes on CPU. ADR-0025
  recorded their absence as an accepted cost; that cost has been paid off, and the MPS path is still
  local-only.

---

### M5 — Comparison UI · complete 2026-08-08

The milestone the workbench exists for, and it turned on a question the plan had not asked: **what
does it mean to compare two runs whose scores are not in the same units?** On the reference data
`pixel_reference` sits at a threshold of 14.28 and `efficientad_anomalib` at 0.065 — a factor of two
hundred, both correct, and one slider across the pair would have printed two true confusion matrices
at operating points nobody chose. **ADR-0028** answers it and was written before the screen.

- [x] `pixel_reference` and `efficientad_anomalib` are comparable side by side on the same split
      under the same evaluation protocol, with identical preprocessing — and the Configuration tab
      is what shows the preprocessing to be identical rather than asserting it.
- [x] Any sample can be opened in a side-by-side map view, N panes wide.
- [x] Every requirement in the brief's UI list is implemented; the comparison view was the last one.

What shipped: `GET /api/compare` and `eval/compare.py` — the operating-point rules and the
per-sample agreement table, beside `threshold.py` and importing no model. A screen with its own run
picker constrained to one dataset and one split, two metric tables split by whether a number depends
on a threshold, overlaid ROC and PR curves, a configuration diff that calls out preprocessing
loudly, a per-sample table filtered to **where the methods disagree**, and a side-by-side sample
view sharing one cut fraction and one zoom across every pane.

**Still binds:**

- **Nothing is compared in score units, ever.** Threshold-independent metrics are functions of the
  ranking and compare directly. Everything else is resolved per run by one shared *rule*, and each
  run's own threshold is printed beside its confusion matrix with the sentence that produced it.
  Remove either and the table becomes the misleading one ADR-0028 rejected — it looks identical.
- **What transfers across runs is a fraction of a range, never a value.** One cut slider drives N
  segmentations, resolving to 42.299 in one run and 0.230 in another, which is what makes it the
  same operating point in both. No map is renormalized into another's scale, and no future screen
  may stack two methods' maps into a difference image.
- **The `f1` rule delegates to `suggest_threshold`.** It is quadratic in the number of samples and
  that cost is accepted: a second F1-optimal search would be free to drift from the one the results
  screen opens its slider at, and a comparison that disagrees with the screen it was reached from is
  worse than a slow one. The same reasoning made `outcome_of` public.
- **A different dataset or split is refused; different preprocessing is warned.** The first is a
  different question and the numbers are not commensurable. The second is a legitimate experiment
  whose AUROC difference is then partly a measurement of the resize — a caveat, stated above the
  table rather than under it.
- **The metric rows are the single-run builders, transposed.** A comparison cannot print a number
  the experiment screen disagrees with, and a metric one run lacks is a dash in that column rather
  than a dropped row that would shift every other column's meaning.
- **A dynamic set of runs means `useQueries`, not a hook in a `map`.** The premise of the screen is
  that a column can be added, and the number of hooks a component calls may not change between
  renders.
- **A cancelled training run leaves the checkpoint untouched**, because `ModelCancelledError`
  propagates before `model.save`. That is why the stale-metrics warning counts only *succeeded*
  train jobs, and why it correctly stays silent on the reference experiments.

---

## Next

### M6 — Custom EfficientAD · engineering complete 2026-08-12

The milestone turned on a question the plan had not asked, and the answer reshaped it: **what is a
second implementation actually for?** Measuring our own against the wrapper's number makes the
wrapper the specification and the goal a second copy of something already known. **ADR-0029** answers
it — anomalib is the floor this method beats or does not — and it was written before the measuring
started, because a rule invented afterwards is not a rule.

**Shipped.** `efficientad_custom`: the PDN in both published widths, the autoencoder, the three
losses, teacher statistics, quantile calibration, an MPS training loop, ADR-0025 resume, and asset
acquisition of its own so the method needs **torch alone, not anomalib**. One registry entry and one
module — no route, no schema, no TypeScript, which is the prediction ADR-0007 made and the first time
it has been tested by a method the interface was not designed around.

**Exit criteria**

- [x] `efficientad_custom` trains and infers on MPS and produces maps and scores through the standard interface. 130 ms/step, 25.8 ms/image.
- [x] The comparison view shows both implementations side by side, and the measured gap is bounded honestly. It is measured on `candle` at three seeds; the long official-protocol sweep remains a research follow-up rather than a product dependency.
- [x] Every M4 visualization works on it with no new code. Checked against two real runs, not a fixture: identical diagnostic keys, kinds and scopes, 37 architecture nodes each with real shapes, same edges. Nothing needed a special case.

**Still binds:**

- **A method that runs and a method that detects are different claims, and only one of them was
  being tested.** Before M6 nothing in the suite asserted that any EfficientAD detects anything —
  the `dl`-gated files covered checkpoint exactness and introspection, and neither called `predict`.
  A model that trained, wrote maps, saved, reloaded and separated nothing would have passed
  everything. The detection test trains for real and asserts a ROC-AUC bar; two assertions beside it
  carry more weight than the AUROC does, because each branch has to separate *on its own* and the
  error on held-out normals has to actually fall.
- **Writing that second assertion found the bug worth finding.** Weight initialisation draws from
  torch's global stream, so `seed` controlled the training order over *different* initial weights.
  Two runs of one configuration were not one experiment, which would have put noise under every
  number this milestone exists to produce.
- **The two implementations agree to 0.002 sample ROC-AUC at 50 steps, and are both inverted there.**
  Independent code arriving at the same number is the strongest validation available; both scoring
  defects *below* normals is a fact about EfficientAD's undertrained regime, not about either
  implementation. A run stopped early is not a weaker detector, it is a backwards one.
- **Only the speed result is claimed: 25.8 ms/image against 46.7.** The wrapper computes the branch
  maps twice, once for the score and again for the diagnostics. On accuracy the seed spread swallows
  a +0.033 median gap, and **the rule set in ADR-0029 before the runs says that is not evidence** —
  even though all three of our runs beat all three of the wrapper's. More seeds, not a softer rule.
- **Our seed spread is twice the wrapper's, and it is our own default's fault.** The quantile fit
  samples half of each calibration map at random where an exact fit would have fitted under
  `torch.quantile`'s limit. Fixing that is the next measurement, and it should narrow the spread
  rather than move the median.
- **`total_parameters` differs by exactly 772** — anomalib counts its `mean_std` and `quantiles`
  `ParameterDict`s as parameters. They are fitted statistics; ours are buffers. A correctness
  decision that turns out to be visible in the Architecture tab.

- **4000 steps is not a budget at which a model can be judged.** Both implementations sit *below* the
  numpy floor there (0.727 and 0.764 against 0.790) where the paper reports 0.975 at 70 000, and the
  seed spread (0.041) is larger than every effect the remaining hypotheses chase (0.01–0.03). The
  head-to-head is fair *at budget 4000* and says nothing about the converged ranking. **A
  step-budget curve to 70 000 therefore runs before any other hypothesis**, as one resumed trajectory
  read at six points rather than five separate runs.
- **Resume had been silently wrong since it was written, in both plugins.** `StepLR.get_lr`
  multiplies the param group's *current* rate and `Adam.load_state_dict` restores the decayed one, so
  every continuation started a tenfold low and dropped again — 1e-5 instead of 1e-4 on the first
  resume, 1e-9 by the fifth. Nothing in the application would have shown it: the loss curve
  continues, the step counter is right, the run finishes. It took a measurement that *depended* on
  the learning rate being correct to make it visible, sixty seconds into the first leg.

- **The curve answered its question at 30 000 and the answer was large.** Sample ROC-AUC 0.723 →
  0.835 and AU-PRO 0.680 → 0.833 between 16 000 and 30 000 steps, on one trajectory where seed noise
  cannot explain it. The 4000-step rows above are not weak measurements of this method, they are
  measurements of an unconverged one — including the comparison against the numpy floor, which
  EfficientAD now clears on both metrics.
- **The failure at low budget is diagnosable, and the diagnostic is worth keeping.** The map's
  maximum lands inside the ground-truth defect on 16% of defect images at 4000 steps and 24% at
  16 000, with the mean maximum *outside* the mask exceeding the one inside it. That single number
  explains why pixel metrics climb while sample ROC-AUC sits flat — an image score is a maximum, so
  it is decided by whatever is hottest, defect or not — and it moves earlier than sample ROC-AUC
  does. Reduction, border artifacts, branch weighting and calibration source were each swept and
  each ruled out first, at a cost of no GPU time, against stored maps.
- **The pretrained teacher is not a public constant, and treating it as one was an unexamined
  assumption.** anomalib's teacher and the one bundled with nelson1425/EfficientAD — the
  reproduction reporting the paper's numbers — have identical architecture and shapes and differ
  tensor by tensor by up to 1.4. "Two independent implementations agree, so it is not the code" was
  sound and also incomplete: the teacher is the *other* thing both share. It is now `teacher_source`,
  a field of the experiment, with both layouts loadable — anomalib names its layers, nelson1425 keys
  by position in an `nn.Sequential`.

**The milestone grew a second half, and the measurement is what grew it.** M6 set out to reimplement
the method. It has now established that the largest lever measured anywhere in this project is not in
the method at all — it is **the teacher**, a weight file nobody here produced. Reimplementing the
student while accepting somebody else's teacher leaves the most important input outside the
workbench. So M6 continues into **distilling the teacher ourselves**:

- **A frozen source model, distilled into the compact PDN.** `wide_resnet101_2` (`IMAGENET1K_V1`),
  `layer2` + `layer3`, patch-aggregated to 384 channels at 64×64, MSE into the PDN. The source model
  is **training-only** — what ships is the same 2.7M-parameter PDN, so inference cost does not move.
- **Staged, and cheap by default.** Imagenette is the smoke corpus and is already on disk; ImageNet-1K
  is an opt-in path that is never the default. The reference recipe is 60 000 steps at batch 16 with a
  WideResNet-101 forward at 512×512, which is not an overnight job here — the step cost gets measured
  and the extrapolation written down before anything long starts.
- **The feature source is a protocol from the first commit.** `features(batch)` plus the preprocessing
  it requires. That is the entire cost of making a frozen DINOv2-S a second implementation later
  rather than a rewrite, and paying it now is nearly free.
- **The expected result is stated in advance.** Imagenette is 13 394 images across 10 classes against
  ImageNet's 1.28M across 1000, so a smoke-distilled teacher should *lose* to `nelson1425`. That is
  the pipeline working, not a regression, and saying so before the run is what makes it evidence.
- **Nothing gets compared to a published number until the official protocol runs.** Every number in
  this milestone so far is on a generated split.

**Where it stopped, and what is deliberately not done yet.** The teacher comparison is replicated
across three seeds, the default has moved, the distillation stage is built and smoke-tested, and the
official one-class split is imported and verified. Three things are **postponed rather than
forgotten**, each because it is machine time rather than design:

- **The protocol sweep at 30 000 steps.** `protocol.py` takes the budget as an argument; the three
  runs are about three and a half hours.
- **A real distilled teacher.** The stage runs end to end; phase 1 is 10 000 steps on Imagenette,
  and the phase-2 corpus is an open choice — ImageNet-1K needs an account and more disk than this
  machine has free.
- **The curve past 30 000.** It ended at 0.955 sample ROC-AUC and 0.994 pixel, with AU-PRO
  plateaued at 16 000 and the per-leg sample gain shrinking (+0.030, +0.027, +0.012) — a
  quantity approaching a limit rather than one that has reached it. `extend.py` forks a
  finished run, so 70 000 is one command rather than a restart.

**New runs no longer use the anomalib teacher.** It stays a value of `teacher_source` only so the
five experiments recorded against it remain loadable. ADR-0029 still makes the wrapper the baseline
this method was measured against; it is now a baseline the method has left behind rather than one it
is tracked against run by run.

---

### M7 — PatchCore · complete 2026-08-09

**Goal.** A third method, and the first one whose resource profile is genuinely awkward.

The awkwardness was the milestone, and it was larger than "needs a config rather than defaults". At
256×256 with `wide_resnet50_2` a ~900-image VisA class generates **921 600 patches — 5.66 GB — before
any selection**, and anomalib's coreset then runs 92 160 Python iterations over all of it. The
measurement came first (`scripts/patchcore-smoke-test.py`, ADR-0008's rule), and it decided the
defaults rather than confirming them.

**Exit criteria**

- [x] PatchCore trains and infers without exhausting memory, and its memory-bank configuration is
      documented. Two independent caps resolved by a pure `plan_bank` **before** the pass, its numbers
      in the log and in a `memory_bank` diagnostic table.
- [x] It appears in the comparison view alongside the others with no changes outside the plugin.
      Verified against a real run: regenerating the API contract after adding the method produces
      **no diff at all**.

**The first real run, on `candle`, split `default`, defaults untouched.** Both caps bound as planned —
512 of 600 images, 195 of 1024 patches each, a 99 840-vector pool at 613 MB, a **9 984-vector bank at
61 MB**, selected in 21 s on the CPU while the embedding pass ran on MPS. Every figure the probe
predicted held to within a few percent, which is what makes the sizing a measurement rather than a
guess. Inference is 27.7 ms/image against `efficientad_custom`'s 26.3 on the same images.

| test subset | `patchcore_anomalib` | `efficientad_custom` (`candle-nelson-curve-s0`) |
| --- | --- | --- |
| sample ROC-AUC | 0.908 | **0.955** |
| AU-PRO | 0.912 | **0.943** |
| pixel ROC-AUC | 0.974 | **0.994** |
| resolved threshold (`f1`) | 37.23 | 0.236 |

**EfficientAD wins on this split, and the comparison is not yet a fair one.** PatchCore ran at
untouched defaults on its first attempt; the EfficientAD run beside it is the product of a milestone
of tuning, on a teacher chosen because it measured better. What the pair establishes is that the
screen works with a structurally different method in it — 26 of 270 samples disagree, and the two
thresholds differ by a factor of 158, which is exactly the situation ADR-0028 exists for. Reading the
0.047 sample ROC-AUC gap as a result about the *methods* would repeat the mistake ADR-0029 was written
to prevent.

**Still binds:**

- **The cost is quadratic in the candidate pool, and the second cap is what bounds it.**
  `max_bank_images` bounds the backbone pass; `max_candidate_vectors` bounds the store *and* the
  selection, since both the iteration count and the work per iteration grow with N. Measured: 1.2 s at
  25 000 candidates, 24.6 s at 100 000, 150.8 s at 250 000, about half an hour at a full class. Images
  are dropped before patches are thinned, because patches inside one image overlap through the 3×3
  pooling and are largely redundant while two images differ by whatever the process actually varies.
- **A method's preferred device can be wrong for a stage inside it.** The backbone forward is 2.8×
  faster on MPS and stays there; the greedy loop is **3× faster on CPU** (2.46 ms/iteration against
  7.16 at N=100 000) because it does too little arithmetic per iteration to cover dispatch. Nothing in
  the application would have shown it — the run finishes and the bank is correct, it is merely three
  times slower than it needed to be.
- **anomalib's coreset is not reproducible, and `torch.manual_seed` does not fix it.**
  `SparseRandomProjection` draws its sparsity pattern through scikit-learn's
  `sample_without_replacement` with `random_state=None` — numpy's global stream. Same seed, same data,
  a different bank, nothing saying so. This is **M6's finding in a different library**: there it was
  weight init drawing from torch's global stream, and the consequence is identical — a `seed` field
  that does not control the result puts unattributable noise under every comparison built on it. Both
  streams are now pinned, asserted in both directions.
- **The two libraries put ImageNet normalization in different places, and one of them is silent.**
  anomalib's EfficientAD normalizes inside `forward`, so the wrapper passes `[0, 1]` through
  unchanged; anomalib's PatchCore puts it in the Lightning pre-processor, which none of these wrappers
  use. Unnormalized pixels do not fail — they run, produce maps, and quietly score an off-distribution
  backbone.
- **The backbone is an experiment input, stored as a fingerprint rather than as weights.** 260 MB per
  experiment is not affordable across a comparison, so `patchcore.pt` carries a sha256 of the backbone
  and `load` refuses a mismatch by name. Without it the checkpoint loads, inference runs, and the
  number is plausible and wrong.
- **ADR-0007's prediction held against a method the interface was not designed around.** One registry
  entry and one module, no route, no schema, no TypeScript, `generated.ts` unchanged. M6 tested this
  with a second EfficientAD, which shares its shape with the first; PatchCore has no steps, no
  gradients and no resume, holds a bank instead of weights, and still cost exactly one entry.
- **The `dl` CI glob was a naming contract spelled after one method.** `test_efficientad_*.py` matched
  nothing for PatchCore, so a new dl-gated file would have skipped in the torch-free job and run
  nowhere at all — the failure that glob's own comment describes. Renamed to `test_dl_*.py`.
  `test_efficientad_assets.py` is deliberately excluded: it is torch-free and had been running in both
  jobs for no reason.

**Size:** small-to-medium — the wrapper reuses M3's integration path.

---

### M8 — Dataset-first workbench + lifecycle · in progress

**Goal.** Make a dataset the stable place where work begins: browse it, manage its splits, create an
experiment and read its history without crossing between unrelated top-level screens.

**Scope**

- Three explicit route layouts (`ReadingLayout`, `WorkspaceLayout`, `CanvasLayout`) with one owner
  for vertical scrolling; images, maps, charts and tables remain the largest region.
- Dataset-local navigation for browse, annotate, splits and experiments. Experiment creation moves
  to a dedicated dataset route; the global experiment catalogue remains a secondary searchable view.
- Server-side experiment search, method/dataset/status/date filters, sortable columns and URL-backed
  query state. Selected compatible experiments can be sent to Compare.
- Previewed deletion of experiments and datasets, covering only application-owned records,
  annotations, caches and artifacts. Referenced source images and source masks are immutable.
- Local reference-pack discovery for VisA and GKN, with one atomic, idempotent `Register all` action.

**Exit criteria**

- [ ] Dataset browse has exactly one visible vertical scrollbar at 1440×900 and 1024×768.
- [x] A new experiment starts inside one dataset and its history is visible beside that action.
- [x] `method=…` returns every matching experiment from SQLite and survives reload/back/forward.
- [ ] Destructive previews name every application-owned consequence, and source trees survive tests.
- [ ] A present VisA/GKN pack registers in one action; a missing pack is an instructional state.

---

### M9 — Annotation system + editor

**Goal.** Turn source masks into immutable provenance and make the application's own editable,
versioned defect annotation the ground truth used for future evaluation.

**Scope**

- Dataset label taxonomy, per-image draft document, immutable completed revisions, checksummed
  derived masks and a ground-truth digest carried by metrics.
- PNG mask, LabelMe and COCO polygon/RLE import/export without changing source annotations.
- Full-height editor: polygon/vertices, brush/eraser, pan/zoom, undo/redo, autosave with conflict
  detection, completion state, keyboard queue and MobileSAM point/box assistance.

**Exit criteria**

- [ ] Imported source masks remain byte-for-byte untouched while their app-owned revisions can be edited.
- [ ] Evaluation uses one completed revision and marks older metric digests stale.
- [ ] PNG, LabelMe and COCO round trips preserve the binary evaluation mask.
- [ ] A keyboard-only annotation pass can move through a queue, edit, save and complete an image.

---

### M10 — Spatial input pipeline + region profiles

**Goal.** Test, rather than assume, whether localising one dominant object before anomaly detection
improves the result while keeping annotations and maps in source-image coordinates.

**Scope**

- Dataset-owned immutable `RegionProfileRevision`s and a `RegionExtractor` registry with `identity`,
  a threshold baseline and MobileSAM.
- One shared source → crop → contain-resize → pad transform and its exact inverse; no model opens an
  image outside this bridge.
- Preview on evenly spaced calibration images, bounded full-dataset preparation, explicit failures,
  CPU fallback and MPS only after a smoke test.
- Paired identity/localised experiments on public data with quality, latency, memory and failure-rate
  reporting. A negative result still closes the milestone.

**Exit criteria**

- [ ] Points and masks round-trip through every transform within the stated synthetic tolerance.
- [ ] Every method consumes identical prepared pixels and persisted maps align with source masks.
- [ ] A profile revision is previewed, built, pinned and reproducible; no failure silently becomes identity.
- [ ] Evidence, not preference, decides whether localisation becomes a default.

---

### M11 — Modern reference methods

**Goal.** Add a small set of methods that represent different anomaly-detection principles and are
useful quality references on Apple Silicon.

**Order:** SuperADD → Dinomaly → GLASS → one zero/few-shot VLM method. Each remains one lazy model
module plus one registry entry, with resource planning, shared preprocessing, CPU/MPS smoke tests,
same/different-seed assertions and a public benchmark. AnomalyDINO stays lower priority because its
principle overlaps more with the existing memory-bank methods.

**Exit criteria**

- [ ] At least three distinct method families run through the unchanged vertical slice.
- [ ] Every unbounded candidate pool or external asset is planned and printed before execution.
- [ ] Results record the exact package, weights, preprocessing and public protocol used.

---

### M12 — Polish + reproducible onboarding

**Goal.** Make the finished workbench coherent, accessible and reproducible from a fresh machine.

**Scope:** responsive light/dark visual QA, loading/empty/error states, keyboard and focus pass,
performance, README setup and reference-data credits, method-extension guide and documentation audit.

**Exit criteria**

- [ ] A fresh public-data workspace reaches register/import → annotate → split → experiment → compare from the README alone.
- [ ] Key screens pass visual review at both target sizes and in both themes.
- [ ] The method-extension guide has been followed end to end, and the handbook matches the code.
