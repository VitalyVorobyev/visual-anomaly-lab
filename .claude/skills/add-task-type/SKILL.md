---
name: add-task-type
description: Add or extend a task in visual-anomaly-lab — semantic segmentation, object detection, instance segmentation, or anything else an experiment can be asked to do beside anomaly ranking. Use when the user asks to "add segmentation", "support detection", "add a task", "a supervised method", "a new evaluator", or to make the first plugin for a task that exists but has none. Walks ADR-0039's seams in the order they depend on each other.
---

# Add a task type

Read first: `docs/adr/0039-a-task-is-frozen-on-the-experiment-and-chooses-its-evaluator.md` (the
decision and its open questions), `docs/architecture/evaluation.md`, `docs/architecture/methods.md`,
and the `## Tasks` section of `docs/backlog.md`, which lists what is built and what is not. **Check
the backlog before assuming a seam below exists** — the ones marked *(planned)* were decided, not
built, when this skill was written.

The one rule behind all of it: **a task is a property of the experiment, frozen at creation.** It
decides the training set, the evaluator and the result screens. The dataset does not have a task; it
has annotation, and whether it is ready for a task is a readiness check.

## The seams, in dependency order

1. **The enum.** `Task` in `backend/src/anomaly_lab/domain/entities.py`. Adding a value needs no
   migration — `experiment.task` is validated in Python (migration 021), like `job.kind`. Add the
   label to `TASK_ORDER`/`TASK_LABEL` in `frontend/src/routes/ExperimentsRoute.tsx`, then
   `scripts/gen-api-types.sh`.
2. **The evaluator.** One class and one entry in `EVALUATORS`, `backend/src/anomaly_lab/eval/evaluators.py`.
   Until it exists, `create_experiment` refuses the task — that is the guard, keep it. An evaluator
   reads stored predictions and ground truth and writes `MetricSet`s; it never imports a model.
   - **Bound memory.** Segmentation accumulates a per-class confusion matrix, never per-pixel arrays;
     detection keeps per-class lists of (confidence, matched) pairs. Linear in images is fine;
     linear in pixels is not.
   - **A metric that cannot be computed is `None`**, not 0.0 — a class absent from a subset has no IoU.
   - **ADR-0028 still holds.** A confidence is not comparable across runs. Anything thresholded is
     resolved per run by one shared rule whose name and value are printed, and Compare puts only
     threshold-free metrics side by side.
3. **Ground truth.** *(planned: annotation schema v2)* `BoxShape` and `instance_id` in
   `domain/annotations.py`; completion writes a class-index PNG and an instances file next to the
   binary mask, and the revision pins its class-to-index table. v1 documents must read as v2
   unchanged. The taxonomy is `AnnotationLabel`, managed on the Annotate tab
   (`routes/dataset/ClassManager.tsx`). Class hotkeys must avoid `0`/`1`.
4. **Predictions.** *(planned)* `Prediction` in `models/base.py` gains optional `label_map` and
   `instances`. Every prediction keeps an image-level `score` (for detection, the top confidence) so
   ranking, the gallery and disagreement keep working.
5. **Targets.** *(planned)* `TrainContext.targets: TargetProvider | None`. It is `None` for
   `anomaly`, which is what makes an anomaly method unable to see a defect mask by construction —
   **never pass ground truth to a plugin any other way.** The training-set policy is the task's:
   `anomaly` trains on the train subset's normals; a supervised task on its annotated samples.
6. **The first plugin.** Follow the `add-method-plugin` skill; declare the task in
   `Capabilities.tasks`. It must still cost one module and one registry entry. If it needs a route,
   a schema or TypeScript, the boundary is wrong — fix the boundary.
7. **Result screens.** The shared shell stays — run bar, subset, `ResultsState` in the URL, the one
   `SampleStage`. Predictions and truth are drawn with `VectorLayer`
   (`frontend/src/components/viewer/`): truth dashed, predictions solid, toned per shape (`normal`
   for a match, `defect` for a false positive, `warn` for a miss). What branches on the task is the
   body of Overview and Benchmark, not the screens around it.

## Tests that prove the seam, not just the feature

- The anomaly evaluator's numbers are unchanged — the existing `test_eval_*` files are the gate.
- `tests/test_task.py`: every registered method still lists `anomaly`; a task without an evaluator
  is refused by name.
- A method is refused for a task it does not declare (`test_experiments_api.py`).
- The evaluator on a tiny synthetic case with a hand-computed answer (two classes, one image, known
  IoU; two boxes, one match, known AP).
- An anomaly plugin's `fit` receives `targets is None`.

## Open questions to raise, not to settle silently

- What a sample-level result means for a multi-channel part outside `anomaly` (ADR-0011 aggregates
  scores; there is no rule yet for classes or boxes). Evaluate per image until one is decided, and
  say so on screen.
- Whether a split for a supervised task should stratify by class.

Record any decision with a live alternative by editing ADR-0039 in place (ADR-0030: no changelog), and
commit with the `safe-commit` skill.
