---
name: add-task-type
description: Add or extend a task in visual-anomaly-lab — semantic segmentation, object detection, instance segmentation, few-shot segmentation, or anything else an experiment can be asked to do beside anomaly ranking. Use when the user asks to "add segmentation", "support detection", "add a task", "a supervised method", "a new evaluator", or to make the first plugin for a task that exists but has none. Walks ADR-0039's seams in the order they depend on each other.
---

# Add a task type

Read first: `docs/adr/0039-a-task-is-frozen-on-the-experiment-and-chooses-its-evaluator.md` and,
for few-shot segmentation, `docs/adr/0040-few-shot-segmentation-is-a-task-and-its-references-are-a-split.md` (the
decision and its open questions), `docs/architecture/evaluation.md`, `docs/architecture/methods.md`,
and the `## Few-shot segmentation` and `## Supervised tasks` sections of `docs/backlog.md`, which lists what is built and what is not. **Check
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
   reads stored predictions and ground truth and writes `MetricSet`s; it never imports a model. It
   names a `headline` metric and answers `current_digest`, so staleness is judged against its own
   truth (`FewShotSegmentationEvaluator` with `eval/segmentation.py`, and
   `SemanticSegmentationEvaluator` with `eval/semantic.py`, are the other two examples).
   - **Bound memory.** Segmentation accumulates a per-class confusion matrix, never per-pixel arrays;
     detection keeps per-class lists of (confidence, matched) pairs. Linear in images is fine;
     linear in pixels is not.
   - **A metric that cannot be computed is `None`**, not 0.0 — a class absent from a subset has no IoU.
   - **ADR-0028 still holds.** A confidence is not comparable across runs. Anything thresholded is
     resolved per run by one shared rule whose name and value are printed, and Compare puts only
     threshold-free metrics side by side.
3. **Ground truth.** Completion writes a class-index PNG and an instances JSON next to the binary
   mask, and the revision pins its class table and the instances file's path and digest
   (`annotation_render.py`); `annotations/class_truth.py` resolves and loads a class's region per
   image (`resolve_class_truth`), or every pinned class at once as a label map
   (`resolve_label_truth`) — an image is labelled for that only when it answers every class. A document holds polygons, boxes and bitmaps, each with an optional `instance_id`; an
   instance is `{instance_id, label_key, box, pixels}` over its final pixels. **The shape union grows
   without a schema version**: a new shape kind is additive, an unset optional field is left out of
   the canonical JSON so stored digests never move, and every kind branch names its kind
   (`assert_never` in Python, a `never` check in TypeScript) rather than falling through an `else`.
   The taxonomy is `AnnotationLabel`, managed on the Annotate tab (`routes/dataset/ClassManager.tsx`);
   the editor's class keys are `2`–`9`, because `0`/`1` are the view's.
4. **Predictions.** A binary task writes its mask with `InferContext.write_mask`; a multi-class
   one writes an 8-bit label map with `InferContext.write_label_map` (nearest, never interpolated)
   and sets `Prediction.label_map`. *(planned)* `Prediction` gains optional `instances`. Every prediction keeps an image-level `score` (for detection, the top confidence) so
   ranking, the gallery and disagreement keep working.
5. **Targets.** `TrainContext.targets: TargetProvider | None` for one class, and its sibling
   `TrainContext.label_targets: LabelTargetProvider | None` for a pinned class list. Both are `None`
   for `anomaly`, which is what makes an anomaly method unable to see a defect mask by
   construction — **never pass ground truth to a plugin any other way.** The training-set policy is the task's, in
   `experiments/policy.py`: `anomaly` trains on the train subset's normals, `few_shot_segmentation`
   on its references, `semantic_segmentation` on the train subset's images labelled for every
   pinned class; a new task adds its branch there. The `infer` log names the evaluator's
   `headline` metric.
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
