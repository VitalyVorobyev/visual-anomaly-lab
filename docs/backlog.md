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

---

## Open

### E7 — Evaluation (M3)

- [ ] **Give `TrainContext` labelled validation data**, so a method can report val AUROC per epoch (M) — **ADR-0007**, **ADR-0011**. M4 wanted the chart and could not have it: `val` is filtered to normals and carries no labels, so there is one class and no AUROC. A plugin-interface decision, not a chart.

### E9 — Anomalib integration (M7)

- [ ] `patchcore_anomalib` wrapper + memory sizing (coreset ratio, memory-bank footprint, documented limits) (M)
- [ ] Revisit the training loop against anomalib's Lightning path if their module stops reaching into `trainer.datamodule`; ours exists only because that coupling would cost the preprocessing bridge (S)

### E16 — The workbench you can iterate in (M4.7)

Six sub-milestones, ordered so the layout churn lands first and later items arrive *into* the new
layout instead of being moved twice. **a–c are self-contained**; d–f are the research features.

- [x] **a — Overview redesign, `Jobs & files` tab, the run bar, threshold curves, thumbnails** (M).
  `pr_curve` returns the cut scores it already computes so `Curve` can carry `t`; precision, recall
  and F1 drawn against the threshold beside its slider. Run controls move to a persistent bar above
  the tabs; the run list, job logs and the artifact listing move off Overview entirely.
- [x] **b — Sample viewer navigation and canvas** (S). `tab` joins `ResultsState`, which fixes the
  gallery link, the back link and prev/next in one change. Cursor-anchored wheel zoom with a
  non-passive listener, double-click toggling fit ↔ 1:1, and the duplicated `ZoomPan` in
  `SampleRoute` deleted in favour of `ZoomPanCanvas`.
- [x] **c — The value under the cursor** (M) — **ADR-0023**. A fixed-header float32 blob fetched once
  per image and indexed in the browser; the colormap and the display range stay server-side. Reports
  the map value in map units and the preprocessed source values, never a colour read back out of a
  picture.
- [x] **d — Layer-level architecture** (M) — **ADR-0024**. Forward hooks over `named_modules()` in a
  shared helper, split so the tree-building half is tested without torch. Bounded by node count, not
  depth. `edges` stays branch-level because `named_modules()` sees modules, not wiring.
- [x] **e — Continue training** (L) — **ADR-0025**. `Capabilities.supports_resume` and a
  `SupportsResume` protocol; `TrainParams.additional_steps`; a format-2 checkpoint; absolute steps.
  Resolves the collision this backlog flagged between warm-starting and a frozen config.
- [ ] **f — Resident worker and on-demand diagnostics** (L) — **ADR-0026**, **ADR-0027**. Spike
  against `pixel_reference` before committing to the design. One resident globally, evicted before
  any job spawns, keyed by a checkpoint fingerprint so stale weights cannot be served.
  - [x] The index half: `DiagnosticEntry.origin`, merge and range rules scoped by it, run-level
    budget carried forward, atomic write. Nothing produces an on-demand entry yet.
  - [ ] `jobs/resident.py`, `jobs/inspector.py`, `experiments/diagnose.py`, the `before_spawn` hook
    on the queue, `POST /{id}/diagnose`, `DELETE /{id}/diagnostics`, the health block, and the
    frontend's diagnose button and clear action. **ADR-0026 and ADR-0027 are not yet written.**

*Items d, e and f are research features M5 does not need, and M6's `efficientad_custom` would
exercise all three anyway. Ordered last for that reason.*

### E10 — Comparison UI (M5)

Sized `M` when it was one line; it is more than that now, and should be split before it is started.

- [ ] N-way metric table: sample / image / pixel ROC-AUC, AU-PRO, AP, confusion at a shared threshold, timing (M)
- [ ] Overlaid ROC and PR curves across the selected experiments (S)
- [ ] Config diff, calling out **preprocessing** differences loudly — comparability under identical preprocessing is the milestone's first exit criterion (S)
- [ ] A/B anomaly-map view on one sample, each map on its own recorded run-wide range with that range printed (M) — needs an ADR: **ADR-0019** does not say how two runs' scales are shown together, and cross-method score units are not comparable
- [ ] Guard the selection to one dataset and one split; warn rather than block on differing preprocessing (S)

*Build it N-way and capability-driven, never two-way and method-named: M7 requires a third method to
appear here with no change outside the plugin. Do not pull training curves for several runs at once —
**ADR-0020** flags that as a known cliff.*

### E11 — Custom EfficientAD (M6)

- [ ] Reimplement EfficientAD from arXiv:2303.14535 as `efficientad_custom` behind the unchanged plugin interface (L)
  - split when scheduled: PDN backbone → teacher distillation → student loss → autoencoder branch → quantile normalization → training loop with progress events → comparison run against `efficientad_anomalib`

### E6 — `classical_circular` (M8, optional)

- [ ] Circle detection: Hough seed → radial-ray subpixel edges → robust circle fit (RANSAC + Taubin), with a median-prior fallback (M) — **ADR-0010**
- [ ] Polar transform + FFT angular-correlation orientation, with a bootstrapped reference (M) — **ADR-0010**
- [ ] Per-channel robust reference: median + MAD over aligned polar training images (S)
- [ ] Predict path: z-map → smoothing → inverse-polar warp → high-percentile score (M)
- [ ] Parameter defaults sweep; record the chosen values and the numbers that justified them (M)
- [ ] Unit tests on synthetic discs: centre, radius and rotation recovered within tolerance; an injected blob raises the score (S)

### E12 — README & polish (M9)

- [ ] "How to add a new anomaly-detection method", written against the real interface with a worked example (S) — **ADR-0007**
- [ ] Docs refresh: `system-design.md` and ADR amendments where the implementation diverged (S)
- [ ] Loading and empty states, error surfaces, keyboard navigation, cross-screen layout consistency (M)
- [ ] Backlog re-triage and a refreshed *Later / ideas* list (S)

*The user-facing README was rewritten in M4.6 and is no longer an M9 task.*

---

## Later / ideas

Not scheduled. Revisit at each re-triage; promote when there is a concrete reason to.

- **`mask.sha256`, as a numbered migration.** `verify` can check a mask file is still there, not that it is still the same file, so a mask re-exported in place silently changes a pixel metric. Worth doing before pixel numbers are relied on (ADR-0016, ADR-0017).
- **Warn when an imported split's train subset contains defects.** The `imported` strategy trusts the source completely, which is the point, but a bad benchmark file produces a bad experiment quietly. The training handler excludes them and logs it; the split screen says nothing.
- **Show the pixel-metric protocol on the results screen.** "Normal images count, with an empty mask" moves the number substantially and is documented nowhere the reader will look.
- Per-channel score quantile normalization before aggregation, so one illumination channel cannot dominate a sample score by scale alone.
- Per-image inference batching for the deep methods — one image per forward pass today, which is simple and leaves throughput on the table.
- Uninformed Students ([papers.md](papers.md) #4) as a fourth method.
- Export / report generation: results and comparison tables to PDF or HTML for sharing without the app.
- Orientation fallback for near-rotationally-symmetric parts, when FFT angular correlation is ambiguous.
- Per-set classical references, if lighting or fixturing drift between capture sets turns out to matter.
