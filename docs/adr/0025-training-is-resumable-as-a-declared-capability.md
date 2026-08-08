# ADR-0025: Training is resumable as a declared capability, and steps are absolute

**Status:** Folded into the handbook (2026-08-08). Accepted 2026-08-08.

> **Read [`architecture/jobs.md`](../architecture/jobs.md) instead** for how this works
> today. This record is kept for its number — cited in the code — and for its reasoning,
> which the handbook does not repeat. It is not where to look up current behaviour (ADR-0030).

Extends **ADR-0007** (the plugin interface) and **ADR-0020** (metric series from the job log). It
resolves the collision the backlog recorded: warm-starting a run wants `max_steps` to be a total,
and an experiment's configuration is frozen at creation.

## Context

Pressing **Train** on a trained experiment threw the model away and started over. Four thousand
steps is eleven minutes on this machine and the paper uses seventy thousand, so "see whether more
steps help" meant paying for the first four thousand again every time. The workbench could measure a
run but not extend one, which makes it a thing you take readings with rather than a thing you tune
in.

Three constraints shaped this.

**Most methods have no steps.** `pixel_reference` computes a median over the training set and is
either fitted or not. Whatever this is, it cannot be two more abstract methods on `AnomalyModel`
without putting `raise NotImplementedError` into the one interface that is supposed to say what
every method does.

**The configuration is frozen at creation.** `db/repositories/experiments.py` offers no way to
update `model_config`, deliberately: it is the record that makes a run reproducible. Making
`max_steps` editable would trade that away for a UI convenience.

**Weights are not enough.** The old checkpoint held a `state_dict` and `model_size`. Continuing from
it means a fresh Adam, so the moments — most of what a long run has learned about its own gradients
— are gone; a fresh `StepLR`, so the schedule restarts; and a fresh image order. The result is a
visible loss spike and a run that is not the run it claims to continue.

## Decision

**Resume is a declared capability, the step budget stays per-run, and step numbers become absolute.**

- **`Capabilities.supports_resume`, plus a `runtime_checkable` `SupportsResume` Protocol** carrying
  `completed_steps()` and `fit_more(train, ctx, *, additional_steps)`. A Protocol, not abstract
  methods: the ABC stays honest and `pixel_reference` grows no stub. The train handler checks the
  flag **and** the protocol against each other and names a disagreement as a plugin bug, rather than
  trusting a flag and failing later on an attribute.
- **`TrainParams.additional_steps`, so the experiment's config is untouched.** How long to continue
  for is a property of *this run*, not of the experiment, and the frozen record stays frozen.
  Declared with `default_factory` — a literal `= None` emits `"default": null` and
  `openapi-typescript` makes the property required, the trap that silently pinned a value for two
  milestones. A test asserts the schema carries no default.
- **Checkpoint format 2** adds optimizer moments, LR-scheduler state, the absolute step counter and
  both RNG streams (torch, and the numpy generator that picks images), with MPS's state guarded by
  `hasattr` so a checkpoint written on Apple silicon still loads elsewhere. Measured on the reference
  configuration: 32 MB becomes **75 MB**. **No option to skip it** — an option would make "can I
  continue this run?" depend on a flag chosen before anyone knew the answer.
- **A format-1 checkpoint is refused for continuation, by name.** It still loads and still infers;
  only `fit_more` refuses it. A continuation that silently restarts Adam's moments is not a
  continuation, and the house rule is that a visible gap beats a fabricated number.
- **Steps reported to `ctx.metric` are absolute across an experiment's training.** A continued run's
  curve is therefore a continuation of the first, with no stitching in the chart and no second log
  to read — which is the bounded form of the cliff ADR-0020 named.
- **A JSON sidecar, `model/train_state.json`,** carries `completed_steps`, `runs` and
  `last_run_steps`, written by the *handler* from `completed_steps()`. The API process has no torch
  by design and cannot open a `.pt`; without this, the configuration panel shows `max_steps: 4000`
  beside an 8000-step model, which is a lie in the reproducibility record.
- **The schedule is recomputed against the new total**, with `last_epoch` set so the position
  carries over. `StepLR` reads `initial_lr` off each param group when `last_epoch >= 0` and `Adam`
  does not set it, so it is seeded first — otherwise resuming raises a `KeyError` a long way from
  its cause.
- **A continuation is refused as a form, with 422**, when the method cannot resume, nothing is
  trained, or the checkpoint is format 1. The same reasoning `create_experiment` applies to a config
  that fails its method's schema: a request that cannot succeed should fail as a request, not as a
  job ten minutes later.

**What "exact" means, precisely, because the loose version is false.** Measured on a real model:
continuing through a save and a load is **bit-identical** to continuing without ever leaving the
process. That is what the checkpoint is responsible for and it is pinned with `rtol=0, atol=0`.

It is **not** true that 10 + 10 equals 20, and the reason is not a defect. `max_steps` is a per-run
budget, so a run of 10 sizes its own `StepLR` for 10 and completes its tenfold decay inside those
steps; the continuation then resizes the schedule to the new total, which puts the learning rate
back up at the resume point. Both facts are deliberate, both are printed on screen, and a second
test asserts that the two paths *differ* — so that nobody later mistakes the difference for a bug
and quietly removes it.

Also outside the checkpoint, and said on screen rather than only here: **the ImageNette penalty-set
iterator restarts.** The optimizer, the schedule, the step counter and the training image order
resume exactly; the penalty batch order does not.

**Ruled out:** making `max_steps` editable (trades away the frozen record); two more abstract methods
on `AnomalyModel` (a lie in the interface for every method without steps); a weights-only warm start
(Adam's moments and the schedule restart — honest, but measurably not a continuation); and freezing
the LR drop at the original total (the entire extension would then train at a tenth of the base
rate, which is close to a no-op).

## Consequences

Continuing a run is one number and one button, and the loss chart reads as one curve.

Negative consequences, accepted honestly:

- **Checkpoints more than double in size.** 32 MB to 75 MB for `small`, measured, and every train job
  now writes that. The artifact listing shows it; nothing else changes.
- **`max_steps` in the frozen config no longer describes the model.** It is the per-run budget, and
  the total lives in the sidecar. Rendering `training_state` beside the configuration is the whole
  mitigation, and without it the panel misinforms.
- **The learning rate visibly goes back up at a resume point**, which will read as a bug to anyone
  who has not been told. The control prints the recomputed schedule before the run starts; that is
  the only thing that makes either choice defensible rather than hidden.
- **The `dl`-gated tests are not run by CI.** They are the only check on the exactness claim, and a
  checkout that never installs the extra will not notice them breaking.
- **A third thing now has a format version** — the checkpoint, alongside the diagnostics index and
  the schema. Each is versioned separately and none of them is checked against the others.
