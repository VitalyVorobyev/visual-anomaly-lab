# Backlog

Working task list, keyed to the milestones in [roadmap.md](roadmap.md). The roadmap says *what* each
milestone must achieve; this says *what to actually do next*.

**Sizes.** `S` ≈ half a day. `M` ≈ one focused day. `L` = multi-day, and should be split before it is
started rather than after. Sizes include reading the generated code properly — a task is not done
until its output has been reviewed.

**Re-triage at the end of every milestone**: close what shipped, delete what stopped mattering, split
any `L` that is next up, and promote anything the milestone revealed out of *Later / ideas*.

---

## Shipped

Kept as one line each. What still constrains future work is in the roadmap's per-milestone summaries;
the detail is in the git history and the ADRs.

| Epic | Milestone | What landed |
| --- | --- | --- |
| E1 | M0 | Layered ignore rules, the safety guard, system design, roadmap, backlog, ADRs 0001–0011 |
| E2 | M1 | Backend scaffold, migration runner and schema v1, frontend scaffold, Tauri shell with port handover, the WebSocket, CI and the pre-commit hook |
| E3 | M2 | Manifest schema, the `channel_folders` adapter, scan/commit, the import review UI with its warnings panel, dataset browser, grouped sample viewer, seeded sample-level splits, `verify` |
| E4 | M2 | Thumbnail and preview cache with content-derived ETags, a pre-warm job, an uncached full-resolution tier |
| E5 | M3 | `AnomalyModel` and the lazy registry, the subprocess worker and its JSON-lines protocol, the FIFO queue with crash recovery, `train` and `infer` handlers, the preprocessing bridge, the diagnostics contract |
| E7 | M3 | Aggregation, threshold-independent metrics, on-demand threshold outputs, rankings, timing, the pixel accumulator at constant memory |
| E8 | M3 | Experiment create with a schema-generated config form, the training console, the results screen with its overlay |
| E9 | M3 | `pixel_reference` and `efficientad_anomalib`, the MPS smoke test, the ImageNette penalty set |
| E13 | M4 | The diagnostic payload route, run-wide display ranges, `GET /jobs/{id}/metrics`, the SVG chart primitives, render-by-kind panels, the experiment tabs, live training charts |
| E14 | M4.5 | The token layer, light/dark themes, the primitive set, the schema→control mapping fixed so every pydantic shape reaches the right control |
| E15 | M4.6 | Query invalidation on a finished run, the Samples gallery with outcome filters, the sample page rebuilt around its canvas, the anomaly segmentation overlay, the artifact listing and *Reveal in Finder*, the diagnostics budget spread across the run |
| E16 | M4.7 | Overview rebuilt around results, the Jobs & files tab and the run bar, threshold curves and verdict thumbnails, tab-preserving sample navigation, the value readout, the layer-level architecture tree, resumable training, and on-demand diagnostics from a resident worker |
| E10 | M5 | `GET /api/compare` and the operating-point rules, the run picker guarded to one split, the two metric tables split by threshold dependence, overlaid curves, the preprocessing-first config diff, the disagreement table, and the N-way sample map view |

---

## Open

### E7 — Evaluation (M3)

- [ ] **Give `TrainContext` labelled validation data**, so a method can report val AUROC per epoch (M) — **ADR-0007**, **ADR-0011**. M4 wanted the chart and could not have it: `val` is filtered to normals and carries no labels, so there is one class and no AUROC. A plugin-interface decision, not a chart.

### E9 — Anomalib integration (M7)

- [ ] `patchcore_anomalib` wrapper + memory sizing (coreset ratio, memory-bank footprint, documented limits) (M)
- [ ] Revisit the training loop against anomalib's Lightning path if their module stops reaching into `trainer.datamodule`; ours exists only because that coupling would cost the preprocessing bridge (S)

### E11 — Custom EfficientAD (M6)

- [x] Reimplement EfficientAD from arXiv:2303.14535 as `efficientad_custom` behind the unchanged plugin interface (L) — **ADR-0029**
  - [x] PDN backbone, pinned against the reference at `atol=0` so the published teacher weights load into the network they describe
  - [x] Teacher statistics, student loss, autoencoder branch, quantile normalization, training loop with progress events
  - [x] Asset acquisition of our own, so the method needs torch and not anomalib
  - [x] Resume (**ADR-0025**), including the penalty order the wrapper restarts
  - [x] A detection test with a real ROC-AUC bar — the thing no EfficientAD in this repo previously had
  - [ ] The head-to-head against `efficientad_anomalib` on VisA, recorded in `measurements-efficientad.md`
- [ ] Measure the hypotheses in `measurements-efficientad.md`, in order: the step-budget curve first (nearly free — one run continues into three points), then `calibration_holdout`, then `score_reduction` (S each, mostly unattended compute)
- [ ] A walkthrough confirming every M4 view renders for `efficientad_custom` with no frontend change; any that needs a special case is a finding about **ADR-0018** and gets its own record (S)

### E17 — The teacher is ours too (M6)

The teacher turned out to be the largest single effect measured in this milestone — swapping which
published one is loaded moved AU-PRO 0.560 → 0.916 at a fixed budget. A weight file nobody in this
repository produced is therefore the most important input we do not control, and this epic is about
producing it: **distil a frozen source model into the compact PDN ourselves**, so the teacher becomes
something the workbench measures rather than something it is handed.

Inference cost is unchanged by any of this. The source model is training-only; what ships is the
same 2.7M-parameter PDN.

- [ ] **Validate the PDN against the reference architecture, not just against its shapes** (S). Build
      nelson1425's `get_pdn_small`/`get_pdn_medium` as a literal `nn.Sequential`, load the published
      weights into both it and ours, and assert identical output. Loading foreign weights is already
      guarded by an order-of-appearance remap and a shape check; neither would catch a network that
      differs in a way the shapes permit.
- [ ] **The distillation stage** (L, and it splits): a `distill` job kind and one module.
      `wide_resnet101_2` (`IMAGENET1K_V1`) frozen, `layer2` + `layer3`, patch-aggregated to 384
      channels at 64×64, MSE into the PDN, Adam 1e-4 / weight decay 1e-5. Writes the PDN weights,
      the feature-normalization statistics, and the full configuration into the model cache as one
      described artifact.
  - [ ] The **feature source behind a protocol** — `features(batch) -> (N, 384, 64, 64)` plus the
        preprocessing it requires. This is the whole cost of making DINOv2-S a second implementation
        later instead of a rewrite, and it is nearly free now.
  - [ ] **Imagenette as the smoke corpus** (already downloaded for the penalty set — 13 394 images),
        ImageNet-1K as an opt-in path that is never the default.
  - [ ] Resumable on the same terms as training (**ADR-0025**), because this is the one job here
        that is genuinely long.
- [ ] **Consume a distilled teacher from the student stage** (S): `teacher_source` names it, and the
      load validates architecture, output channels and recorded preprocessing before it is used.
- [ ] **Audit the baseline on the official protocol** (M): VisA `candle`'s official `1cls` split
      imported, training confirmed to see only normals, `efficientad_anomalib` run on it, and
      per-image labels, scores and maps exported. This is the run that may be compared to a published
      number; nothing measured so far may be.
- [ ] **Make the aggregation comparison repeatable** (S). Already measured once, and the answer is
      that it barely matters — `max`, top-k, p95/p99 and the plain mean all land inside 0.71–0.77 on
      the same maps. Worth having as an auditable artifact rather than a scratchpad script,
      **not** worth expecting a win from.

**Two things to be honest about before starting:**

- **An Imagenette-distilled teacher is a pipeline validation, not a competitive teacher.** Imagenette
  is 13 394 images across 10 classes; the reference recipe distils over ImageNet-1K, 1.28M images
  across 1000. Expect it to lose to `nelson1425` and report that as the expected result rather than a
  regression.
- **A full ImageNet distillation is not an overnight job on this hardware.** The reference recipe is
  60 000 steps at batch 16 with a WideResNet-101 forward at 512×512. The step cost will be measured
  and the extrapolation written down before anything long is started.

**Found while building it, not scheduled:**

- [ ] Make the autoencoder resolution-agnostic — replace the hard-coded `//64 - 1` upsample ladder and the 8×8 bottleneck, so the 256 px floor goes away (M). The guard refusing smaller inputs is honest but it refuses a configuration the architecture could support, and this crashes the wrapper outright.
- [ ] Consider the same input-size guard for `efficientad_anomalib`, which fails inside `conv2d` with a message about a padded input size (S).
- [ ] Batched inference for the deep methods — one image per forward pass today (M).

### E6 — `classical_circular` (M8, optional)

- [ ] Circle detection: Hough seed → radial-ray subpixel edges → robust circle fit (RANSAC + Taubin), with a median-prior fallback (M) — **ADR-0010**
- [ ] Polar transform + FFT angular-correlation orientation, with a bootstrapped reference (M) — **ADR-0010**
- [ ] Per-channel robust reference: median + MAD over aligned polar training images (S)
- [ ] Predict path: z-map → smoothing → inverse-polar warp → high-percentile score (M)
- [ ] Parameter defaults sweep; record the chosen values and the numbers that justified them (M)
- [ ] Unit tests on synthetic discs: centre, radius and rotation recovered within tolerance; an injected blob raises the score (S)

### E12 — README & polish (M9)

- [ ] "How to add a new anomaly-detection method", written against the real interface with a worked example (S) — **ADR-0007**
- [ ] Docs refresh: handbook pages and ADR amendments where the implementation diverged (S)
- [ ] Loading and empty states, error surfaces, keyboard navigation, cross-screen layout consistency (M)
- [ ] Backlog re-triage and a refreshed *Later / ideas* list (S)

*The user-facing README was rewritten in M4.6 and is no longer an M9 task.*

---

## Later / ideas

Not scheduled. Revisit at each re-triage; promote when there is a concrete reason to.

- **`mask.sha256`, as a numbered migration.** `verify` can check a mask file is still there, not that it is still the same file, so a mask re-exported in place silently changes a pixel metric. Worth doing before pixel numbers are relied on (ADR-0016, ADR-0017).
- **Warn when an imported split's train subset contains defects.** The `imported` strategy trusts the source completely, which is the point, but a bad benchmark file produces a bad experiment quietly. The training handler excludes them and logs it; the split screen says nothing.
- **Show the pixel-metric protocol on the results screen.** "Normal images count, with an empty mask" moves the number substantially and is documented nowhere the reader will look.
- **Type-check under the `dl` extra too.** The `backend-dl` CI job runs the gated tests but not mypy, so torch's real types are never followed. It is a different check from the torch-free one and could surface a pile of findings at once, which is why it was not bundled into the fix that created the job.
- **Warn when the diagnostics directory gets large.** The clear button reports what it freed, but nothing says when there is something to free; on-demand entries are deliberately uncapped (ADR-0027) and the artifact listing's byte count is the only signal.
- **Make the F1-optimal threshold a sorted sweep.** `suggest_threshold` evaluates a full confusion matrix per candidate score, so it is quadratic; the comparison screen pays it once per run. Fine on a few hundred samples and not on a few thousand. It must stay **one** implementation — the comparison deliberately shares the results screen's (ADR-0028), so this is a change to that function, never a second copy.
- **A cross-dataset leaderboard.** A comparison is refused across datasets and splits because the numbers are not commensurable, which is right for a table of confusion matrices and wrong for the question "how does this method do across the benchmarks". That is a different screen with a different guard, not a loosening of this one.
- Per-channel score quantile normalization before aggregation, so one illumination channel cannot dominate a sample score by scale alone.
- Per-image inference batching for the deep methods — one image per forward pass today, which is simple and leaves throughput on the table.
- Uninformed Students ([papers.md](papers.md) #4) as a fourth method.
- Export / report generation: results and comparison tables to PDF or HTML for sharing without the app.
- Orientation fallback for near-rotationally-symmetric parts, when FFT angular correlation is ambiguous.
- Per-set classical references, if lighting or fixturing drift between capture sets turns out to matter.
