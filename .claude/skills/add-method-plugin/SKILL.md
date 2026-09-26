---
name: add-method-plugin
description: Add, port or substantially change an anomaly-detection method in visual-anomaly-lab. Use when the user says "add method X", "port Y", "wrap Z from anomalib", "new plugin", or changes a method's training, scoring, seeding, memory plan or export. Walks the ADR-0007 plugin boundary as a checklist, from the registry entry to the public promotion gate.
---

# Add a method plugin

The contract (ADR-0007): **a new method costs one module and one entry in
`backend/src/anomaly_lab/models/registry.py`** — no route, no schema, no line of TypeScript. If the
work below makes you touch jobs, evaluation, results or the frontend, stop: the boundary is wrong,
and that is the finding to report, not a caller to patch.

Read first: `docs/architecture/methods.md` (the whole page — it records every trap a previous
method fell into), `backend/src/anomaly_lab/models/base.py`, and the closest existing plugin:

| If the method…                                   | model it on           |
| ------------------------------------------------ | --------------------- |
| trains nothing, holds statistics                 | `pixel_reference`, `subspace_ad` |
| holds a memory bank of features                  | `patchcore_anomalib`, `dino_memory` |
| trains for steps and can continue                | `efficientad_custom`, `dinomaly_custom` |
| synthesises anomalies                            | `glass_anomalib`      |

## Checklist

**Before code**

- [ ] If it depends on a new library or the accelerator, run `scripts/mps-smoke-test.py` (and write
      a `scripts/<method>-smoke-test.py` like the existing ones) before any wrapper code (ADR-0029).
- [ ] Decide ours versus wrapped. A wrapper that reaches into a library's trainer or datamodule is
      a cost; say why it is worth it.

**The module** — `backend/src/anomaly_lab/models/<key>.py`

- [ ] A subclass of `AnomalyModel` with `title`, `summary`, `config_model()`, `capabilities()`,
      `availability()`, `fit`, `predict`, `save`, `load`.
- [ ] **Heavy imports inside functions.** Module scope imports pydantic and numpy at most.
      `describe_all` imports every plugin to render the picker; one module-scope `import torch`
      costs everyone three seconds. `availability()` uses `module_available(...)` so a torch-free
      checkout shows the method greyed out with a reason.
- [ ] The config is a pydantic model whose every field has a `description`, bounds where they
      exist, and a `Literal`/`StrEnum` for choices. The form is generated from it; if it renders
      wrong, fix the schema-to-control mapping, never special-case the method.
- [ ] Mark the one to four fields a person actually decides — encoder or backbone, scoring rule,
      training length, the axes a gate or sweep varied — with
      `json_schema_extra={"x-primary": True}`; the rest fold. `tests/test_method_decisions.py`
      fails a method with none or more than four.
- [ ] Add the key to `STATUS` in `models/registry.py` (`experimental` until a gate says otherwise,
      `floor` for a task's numpy baseline). Move it to `supported`, or into `RECOMMENDED` for its task,
      only with the `docs/measurements.md` verdict that decides it, and rerun `scripts/build-book.py`.
- [ ] **Pixels come only through `models/preprocessing.load_array`.** A method that decodes images
      any other way makes every comparison against it partly a measurement of its resize.
- [ ] Standardisation for a backbone is the model's business (methods.md, "Standardizing for a
      backbone"), read from the backbone's own config.
- [ ] Anything the method cannot read — a patch size the frame does not divide, a channel count —
      is refused in `check_input`, so creation says so instead of a job failing at fit.
- [ ] **Declare `native_size(config)`** — the frame the method's measurement ran at, the size a run
      that names none reads (methods.md, "Native size"). The base default is 256 × 256; a method gated
      at another frame returns it, so a run at its defaults reproduces the measured protocol. If the
      config decides a patch, return a size it divides (`dino_backbone.native_frame`) and declare
      `size_multiple(config)` so the create form snaps to it. `test_check_input.py` asserts every
      registered method's native size passes its own `check_input`; a new method is covered by being
      registered. Add a row to the native-size table in methods.md.
- [ ] `Capabilities` declares what is true — `requires_training`, `supports_resume` (then also
      satisfy `SupportsResume`), `produces_diagnostics`, `channel_aware`, `preferred_device`.
      `portable_formats` stays **empty** until `scripts/export-parity-gate.py` has passed on a
      real fit and its verdict is in `docs/measurements.md`.
- [ ] Scores are per image (methods.md, "Contract: scores are per-image"); maps go through
      `ctx.write_map`.
- [ ] **Bound anything linear in the dataset before it runs**, print the plan, and use
      `models.base.evenly_spaced` — never the first N — when capping. A memory bank gets a
      `plan_bank`-style footprint check (methods.md, "Bounding a memory-bank method").
- [ ] Cancellation: call `ctx.raise_if_cancelled()` inside every long loop.

**The registry** — one lazy loader and one `LOADERS` entry, plus a comment line in the running
note above `LOADERS` saying what the method proved about the boundary.

**Tests**

- [ ] Anything that imports torch lives in `backend/tests/test_dl_<key>.py`. The prefix is what CI's
      `Backend (dl extra)` job globs; a file without it runs in no job.
- [ ] Torch-free logic (plans, math, config validation) gets an ordinary test file so the default
      job measures it.
- [ ] **Seed reproducibility in both directions**: same seed → identical artefact and scores;
      different seed → different. Libraries hide RNG streams (anomalib's coreset uses
      scikit-learn's numpy RNG; torch's global stream feeds weight init) — seed all of them.
- [ ] Save → load round-trip scores identically. If resumable: N steps then M more equals N+M.
- [ ] A cancelled fit leaves no half-written artefact.

**Verification in the app** — nothing in the frontend should need to change. Create an experiment
with the new method on a VisA class, train, score, open the results and a sample; the picker, form
and results must all work untouched. The `lab-visual-pass` skill drives this.

**Promotion** — a method ships `experimental` until it passes a paired public gate: shared immutable
pixels, VisA `candle` and `pcb1` at 448 × 448 against PatchCore. `scripts/dino-memory-public-gate.py`
is the template. The verdict — protocol, numbers, decision — goes into `docs/measurements.md`; the
roadmap and methods handbook cite it rather than restating numbers. Research sweeps run outside the
app and only their verdict ships (ADR-0038).

**Docs in the same change**

- [ ] A section in `docs/architecture/methods.md`: what the method is, what it bounds, what it
      refuses, and anything that surprised you.
- [ ] The method count and list in `CLAUDE.md` **and** `AGENTS.md`, and in `docs/roadmap.md`.
- [ ] `docs/backlog.md`: the gate and the export as open items if they are not done.
- [ ] `scripts/gen-api-types.sh` — normally a no-op, since methods are data; a diff means the
      boundary leaked.

Then commit with the `safe-commit` skill.
