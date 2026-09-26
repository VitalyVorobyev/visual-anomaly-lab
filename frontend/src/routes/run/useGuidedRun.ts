/**
 * Every choice of a guided run, resolved: what the reader picked where they picked, and the
 * default everywhere else.
 *
 * The defaults are the ones the rest of the workbench already states — the task the
 * dataset's truth answers (`suggestedTask`), the dataset's "Full frame" profile, the task's
 * first split preset, the task's recommended method from the registry (`orderMethods`) and
 * the method's own input size (`POST /api/experiments/input-size`) — so a run nobody touched
 * is the run the classic form would have started from. Nothing here names a method or a
 * dataset.
 */

import { describeFields, initialValues, jsonErrors, missingRequired, outOfRange, toOptions, type RawValues } from "@vitavision/lab-ui";
import { useCallback, useEffect, useMemo, useState } from "react";

import type {
  ClassCount,
  ModelDescription,
  RegionProfileRevision,
  SplitDetail,
  SplitParamsInput,
  SplitPreset,
  SubsetComposition,
  Task,
} from "../../api/client";
import {
  guidedRunKey,
  readGuidedRun,
  suggestedTask,
  writeGuidedRun,
  type GuidedRunState,
  type SplitChoice,
} from "../../api/guidedRun";
import { fullFrameProfile, inputSizeState } from "../../api/inputSize";
import { isRecommended, orderMethods } from "../../api/methodChoice";
import { truthServesTask } from "../../api/truth";
import { compositionMode, type ClassInfo, type CompositionMode } from "../../components/SplitComposition";
import { useAnnotationLabels, useClassCoverage } from "../../hooks/useAnnotations";
import { useDataset, useSplits } from "../../hooks/useCatalog";
import { useDatasetReadiness, splitServesTask, type TaskReadiness } from "../../hooks/useDatasetReadiness";
import { useInputSize, useModelTypes } from "../../hooks/useExperiments";
import { useRegionProfiles } from "../../hooks/useRegionProfiles";
import { useSplitPresets, useSplitPreview } from "../../hooks/useSplitPresets";

export const TASK_ORDER: Task[] = [
  "anomaly",
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];

export const TASK_TITLE: Record<Task, string> = {
  anomaly: "Anomaly detection",
  few_shot_segmentation: "Few-shot segmentation",
  semantic_segmentation: "Segmentation",
  object_detection: "Object detection",
};

/** What the run is asked to do, in one line a person reads before choosing it. */
export const TASK_MEANING: Record<Task, string> = {
  anomaly: "Learn what normal looks like, then rank every image by how far it strays from it.",
  few_shot_segmentation: "Show a handful of examples of one class, then find it in every other image.",
  semantic_segmentation: "Learn every annotated class from labelled pixels, then label each pixel.",
  object_detection: "Learn every annotated class from its boxes, then box each one it finds.",
};

/** One way to split, resolved to what it would contain. */
export interface SplitOption {
  choice: SplitChoice;
  label: string;
  meaning: string;
  /** The name the split has, or will be given. */
  name: string;
  composition: readonly SubsetComposition[] | undefined;
  mode: CompositionMode;
  /** Why this option cannot be drawn on the dataset, from its dry run. */
  error: string | null;
  /** A preset's params, targeted at the chosen class; absent for an existing split. */
  params?: SplitParamsInput;
  pending: boolean;
}

export function useGuidedRun(datasetId: number) {
  const key = guidedRunKey(datasetId);
  const [state, setState] = useState<GuidedRunState>(() => readGuidedRun(key));
  useEffect(() => writeGuidedRun(key, state), [key, state]);
  const update = useCallback(
    (change: Partial<GuidedRunState>) => setState((current) => ({ ...current, ...change })),
    [],
  );

  const dataset = useDataset(datasetId);
  const catalog = useModelTypes();
  const presets = useSplitPresets(datasetId);
  const splits = useSplits(datasetId);
  const profiles = useRegionProfiles(datasetId);
  const readiness = useDatasetReadiness(datasetId);

  // --- goal --------------------------------------------------------------------------------
  const methods = catalog.data?.methods;
  const offered = useMemo(
    () =>
      TASK_ORDER.filter(
        (task) =>
          (methods ?? []).some((method) => method.capabilities.tasks.includes(task)) &&
          truthServesTask(dataset.data?.truth, task),
      ),
    [methods, dataset.data?.truth],
  );
  const needsAnnotation = (task: Task) =>
    readiness.known &&
    (readiness.tasks.find((entry: TaskReadiness) => entry.task === task)?.missing.includes("annotate") ??
      false);
  const runnable = offered.filter((task) => !needsAnnotation(task));
  const suggested = suggestedTask(dataset.data?.truth, dataset.data?.class_geometry, runnable);
  const task: Task | undefined =
    state.task !== undefined && runnable.includes(state.task) ? state.task : suggested;
  const targeted = task === "few_shot_segmentation";

  // The classes a run can learn: every class some sample shows, from the coverage the
  // readiness check reads — which counts an imported defect mask as its class, as a few-shot
  // run does — and from the dataset's class counts until that has answered. Most frequent
  // first; class order breaks ties, as the presets order them.
  const labels = useAnnotationLabels(datasetId);
  const coverage = useClassCoverage(datasetId);
  const classes: ClassCount[] = useMemo(() => {
    const named = new Map((labels.data ?? []).map((label) => [label.key, label]));
    const shown: ClassCount[] = coverage.data
      ? coverage.data
          .filter((entry) => entry.present > 0)
          .map((entry) => ({
            key: entry.label_key,
            name: named.get(entry.label_key)?.name ?? entry.label_key,
            color: named.get(entry.label_key)?.color ?? "",
            samples: entry.present,
          }))
      : [...(dataset.data?.class_counts ?? [])];
    return shown.sort((a, b) => b.samples - a.samples);
  }, [coverage.data, labels.data, dataset.data?.class_counts]);
  const classInfo: ClassInfo[] = useMemo(
    () =>
      labels.data
        ? labels.data.map(({ key: k, name, color }) => ({ key: k, name, color }))
        : (dataset.data?.class_counts ?? []).map(({ key: k, name, color }) => ({ key: k, name, color })),
    [labels.data, dataset.data?.class_counts],
  );
  const targetClass = targeted
    ? (classes.find((entry) => entry.key === state.targetClass) ?? classes[0])?.key
    : undefined;

  // --- method ------------------------------------------------------------------------------
  // In the registry's order of standing, with what this installation can run before what it
  // cannot: the choice in front of the reader should be one that can start.
  const methodsForTask: ModelDescription[] = useMemo(() => {
    if (task === undefined) return [];
    const ordered = orderMethods(
      (methods ?? []).filter((method) => method.capabilities.tasks.includes(task)),
      task,
    );
    return [
      ...ordered.filter((method) => method.availability.available),
      ...ordered.filter((method) => !method.availability.available),
    ];
  }, [methods, task]);
  // The registry's recommendation when this installation cannot run it, so the step can say
  // why the method it opens on is not the one the registry names.
  const unavailableRecommendation = methodsForTask.find(
    (entry) => task !== undefined && isRecommended(entry, task) && !entry.availability.available,
  );
  // The recommended method, unless this installation cannot run it: a front door that
  // opens on a method missing its dependencies has chosen a run that cannot start.
  const method =
    methodsForTask.find((entry) => entry.key === state.methodKey) ??
    methodsForTask.find((entry) => entry.availability.available) ??
    methodsForTask[0];
  const configFields = useMemo(
    () => (method ? describeFields(method.config_schema) : []),
    [method],
  );
  const configValues: RawValues = useMemo(
    () => ({
      ...initialValues(configFields),
      ...(state.methodKey === method?.key ? state.configValues : {}),
    }),
    [configFields, state.methodKey, state.configValues, method?.key],
  );
  const configProblems = [
    ...jsonErrors(configFields, configValues),
    ...missingRequired(configFields, configValues),
    ...outOfRange(configFields, configValues),
  ];
  const methodOptions = useMemo(
    () => (jsonErrors(configFields, configValues).length === 0 ? toOptions(configFields, configValues) : {}),
    [configFields, configValues],
  );
  const inputSize = useInputSize(method?.key, methodOptions);
  const size = inputSizeState("", "", inputSize.data);

  // --- look --------------------------------------------------------------------------------
  const fullFrame: RegionProfileRevision | undefined = profiles.data
    ? fullFrameProfile(profiles.data)
    : undefined;
  const profile =
    profiles.data?.find((entry) => entry.id === state.profileId) ?? fullFrame ?? profiles.data?.[0];
  const isFullFrame = profile === undefined || profile.id === fullFrame?.id;

  // --- split -------------------------------------------------------------------------------
  const presetsForTask: SplitPreset[] = useMemo(
    () => (task === undefined ? [] : (presets.data ?? []).filter((preset) => preset.tasks.includes(task))),
    [presets.data, task],
  );
  const existingForTask: SplitDetail[] = useMemo(
    () =>
      task === undefined
        ? []
        : (splits.data ?? []).filter(
            (split) =>
              splitServesTask(split, task) &&
              // A split of references drawn for another class is not this run's.
              (!targeted || split.params.label_key == null || split.params.label_key === targetClass),
          ),
    [splits.data, task, targeted, targetClass],
  );
  const stored = state.split;
  const storedStands =
    stored === undefined
      ? false
      : stored.kind === "preset"
        ? presetsForTask.some((preset) => preset.key === stored.key)
        : existingForTask.some((split) => split.id === stored.id);
  const [firstPreset] = presetsForTask;
  const [firstExisting] = existingForTask;
  const choice: SplitChoice | undefined = storedStands
    ? stored
    : firstPreset
      ? { kind: "preset", key: firstPreset.key }
      : firstExisting
        ? { kind: "existing", id: firstExisting.id }
        : undefined;

  const chosenPreset =
    choice?.kind === "preset" ? presetsForTask.find((preset) => preset.key === choice.key) : undefined;
  // A few-shot preset is drawn for the chosen class; another class than its own is a fresh
  // dry run, as on the Splits tab's card.
  const retargeted =
    chosenPreset !== undefined &&
    targeted &&
    targetClass !== undefined &&
    chosenPreset.params.label_key !== targetClass &&
    chosenPreset.classes.some((entry) => entry.key === targetClass);
  const presetParams = (preset: SplitPreset): SplitParamsInput =>
    retargeted && preset === chosenPreset ? { ...preset.params, label_key: targetClass } : preset.params;
  const dryRun = useSplitPreview(
    datasetId,
    retargeted && chosenPreset ? { params: presetParams(chosenPreset) } : undefined,
  );

  const splitOptions: SplitOption[] = [
    ...presetsForTask.map((preset): SplitOption => {
      const params = presetParams(preset);
      const redrawn = retargeted && preset === chosenPreset;
      const composition = redrawn ? dryRun.data?.composition : preset.composition;
      return {
        choice: { kind: "preset", key: preset.key },
        label: preset.label,
        meaning: preset.meaning,
        name: redrawn ? (dryRun.data?.name ?? preset.name) : preset.name,
        composition,
        mode: compositionMode(preset.params.strategy, params, composition ?? []),
        error: redrawn ? (dryRun.data?.error ?? dryRun.error?.message ?? null) : null,
        params,
        pending: redrawn && dryRun.data === undefined && dryRun.error === null,
      };
    }),
    ...existingForTask.map(
      (split): SplitOption => ({
        choice: { kind: "existing", id: split.id },
        label: split.name,
        meaning: `An existing split${split.experiments.length > 0 ? `, used by ${split.experiments.length} run${split.experiments.length === 1 ? "" : "s"}` : ""}.`,
        name: split.name,
        composition: split.composition,
        mode: compositionMode(split.strategy, split.params, split.composition),
        error: null,
        pending: false,
      }),
    ),
  ];
  const split = splitOptions.find((option) => sameChoice(option.choice, choice));

  // --- run ---------------------------------------------------------------------------------
  const datasetName = dataset.data?.name;
  const targetName = classes.find((entry) => entry.key === targetClass)?.name ?? targetClass;
  const suggestedName =
    method && datasetName
      ? `${method.title}${targetName ? ` · ${targetName}` : ""} on ${datasetName}`
      : "";

  // The reads every step depends on. Until they answer nothing is missing yet, and a failed
  // one is said once, above the step, rather than read as "nothing fits".
  const reads = [dataset, catalog, presets, splits, profiles];
  const loading = reads.some((read) => read.isPending && read.error === null);
  const readError = reads.find((read) => read.error !== null)?.error ?? null;

  const missing: string[] = [];
  if (task === undefined) missing.push("a task this dataset's truth can serve");
  if (targeted && targetClass === undefined) missing.push("a class to segment");
  if (split === undefined) missing.push("a split");
  else if (split.error) missing.push("a split that can be drawn");
  if (method === undefined) missing.push("a method");
  else if (!method.availability.available) missing.push("a method this installation can run");
  if (configProblems.length > 0) missing.push("method options within their range");

  return {
    state,
    update,

    dataset,
    catalog,
    presets,
    splits,
    profiles,
    readiness,
    offered,
    needsAnnotation,
    suggested,
    task,
    targeted,
    classes,
    classInfo,
    targetClass,
    methodsForTask,
    unavailableRecommendation,
    method,
    configFields,
    configValues,
    configProblems,
    inputSize,
    methodOptions,
    size,
    profile,
    isFullFrame,
    splitOptions,
    split,
    choice,
    suggestedName,
    name: state.name.trim() || suggestedName,
    missing,
    loading,
    readError,
  };
}

export type GuidedRun = ReturnType<typeof useGuidedRun>;

export function sameChoice(a: SplitChoice | undefined, b: SplitChoice | undefined): boolean {
  if (a === undefined || b === undefined) return false;
  if (a.kind === "preset" && b.kind === "preset") return a.key === b.key;
  if (a.kind === "existing" && b.kind === "existing") return a.id === b.id;
  return false;
}

/** What a split choice is drawn from, so a split created for it is reused only for it. */
export function choiceKey(option: SplitOption): string {
  return option.choice.kind === "existing"
    ? `split:${option.choice.id}`
    : `preset:${JSON.stringify(option.params)}`;
}
