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

## Next

### M5 — Comparison UI

**Goal.** Several methods, one split, one evaluation protocol, compared directly. This is what makes the workbench worth having.

**Scope**

- Multi-experiment metric table: sample / image / pixel ROC-AUC, PRO, confusion at a chosen threshold, timing.
- A/B overlay showing two methods' anomaly maps on the same sample, to see where they disagree.
- TP / FP / TN / FN filtering polish across the results and comparison screens.

**Exit criteria**

- [ ] `pixel_reference` and `efficientad_anomalib` are comparable side by side on the same split under the same evaluation protocol, with identical preprocessing.
- [ ] Any sample can be opened in A/B overlay.
- [ ] Every requirement in the brief's UI list is implemented.

**Size:** medium — a focused piece of frontend work over an evaluation layer that already produces the numbers.

---

### M6 — Custom EfficientAD

**Goal.** Reimplement EfficientAD from the paper (arXiv:2303.14535) in PyTorch, behind the same interface, with the anomalib version's number as its yardstick — the research payoff of having built the workbench.

**Scope**

- PDN student/teacher architecture and distillation loss; the autoencoder branch; quantile-based score normalization; an MPS training loop.
- Registered as `efficientad_custom` behind the unchanged plugin interface (**ADR-0007**), emitting the same diagnostics as the wrapper so M4's views work on it unchanged.

**Exit criteria**

- [ ] `efficientad_custom` trains and infers on MPS and produces maps and scores through the standard interface.
- [ ] The comparison view shows both implementations side by side, and the gap between them is measured and explained (a gap is an acceptable outcome; an unexplained gap is not).
- [ ] Every M4 visualization works on it with no new code — if any needed a special case, that is a finding about the diagnostics contract and gets an ADR.

**Size:** large, but **isolated and low-risk to the rest of the system**. Split it into subtasks (backbone → distillation → autoencoder branch → normalization → training loop) when it is actually scheduled.

---

### M7 — PatchCore

**Goal.** A third method, and the first one whose resource profile is genuinely awkward.

**Scope**

- `patchcore_anomalib` wrapper, with explicit attention to memory: a coreset memory bank over a full training set at native resolution needs a sized, documented configuration rather than library defaults.

**Exit criteria**

- [ ] PatchCore trains and infers without exhausting memory, and its memory-bank configuration is documented.
- [ ] It appears in the comparison view alongside the others with no changes outside the plugin.

**Size:** small-to-medium — the wrapper reuses M3's integration path.

---

### M8 — `classical_circular` (optional)

**Goal.** The showcase-specific baseline, if it is still wanted once the universal tool exists.

**Scope**

- Circle detection (Hough seed → radial-ray subpixel edges → robust circle fit, with a median-prior fallback), polar transform, FFT angular-correlation orientation alignment against a bootstrapped reference, per-channel median/MAD reference build, and a predict path producing a z-map → smoothing → inverse-polar warp → percentile score (**ADR-0010**).
- Built as a circle-fit front-end onto `pixel_reference`, which is already the geometry-free core of the same algorithm.

**Exit criteria**

- [ ] It runs on the showcase dataset through the identical interface, and its inverse-polar warp is verified rather than assumed.
- [ ] It is the *only* component that assumes anything about the dataset's geometry.

**Size:** medium. Deferring an optional milestone in a spare-time project is close to cancelling it, and **ADR-0015** says so outright.

---

### M9 — Polish + README

**Goal.** Make the project reproducible by someone who is not the author (including the author six months later), and bring the documentation back in line with what was actually built.

**Scope**

- Full README: setup from a fresh machine, how to obtain and import each reference dataset, architecture overview, and a **"how to add a new anomaly-detection method"** guide walking through the plugin interface with a working example.
- UX polish pass: loading and empty states, error surfaces, keyboard navigation, consistent layout across screens — it should read as an engineering tool, not a debug panel.
- Documentation refresh: `system-design.md` and the ADRs updated where implementation diverged from the decision (with amendments, not silent edits).
- Backlog re-triage: drop what no longer matters, promote what the work revealed.

**Exit criteria**

- [ ] Following the README alone on a fresh machine reaches full M3 functionality on a **public** dataset — download, import, train, view results — with no undocumented steps and no private data required.
- [ ] The "add a new method" guide has been followed end-to-end at least once (M6 counts if it was written first and used as the recipe).
- [ ] Documentation matches reality: no ADR describes a decision that was silently reversed.
- [ ] Backlog re-triaged and the "Later / ideas" list refreshed.

**Size:** medium. README and docs work expands to fill the time available; timebox it.
