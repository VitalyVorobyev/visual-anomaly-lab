/**
 * The experiment list, and the form that creates one.
 *
 * The method picker, the configuration form and the preprocessing form are all generated
 * from schemas the backend serves (`useModelTypes`). Nothing in this file names a method,
 * knows a hyperparameter, or has an opinion about what EfficientAD needs — which is the
 * claim ADR-0007 makes, tested by the fact that `pixel_reference` and
 * `patchcore_anomalib` have nothing in common and both render here.
 *
 * The form is three numbered steps, and the numbering is not decoration: a split belongs to
 * a dataset, and a configuration belongs to a method, so each choice is only answerable
 * once the one before it is made.
 *
 * Two things this screen used to do that are worth not doing again. The method arrived as a
 * `<select>` whose options could be picked and then silently refused — the Create button
 * disabled itself with the reason half a screen away — so the answer to "why can't I run
 * EfficientAD" was nowhere near the control that raised the question; it is now on the card
 * you press. And the three option groups were three stacked `<details>`, the first of which
 * wrapped a *second* `<details>` from `SchemaForm`, giving two nested boxes with two
 * uppercase summaries over eight fields that nobody has to touch.
 */

import { ArrowDown, ArrowUp, ArrowUpDown, Plus, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import type { ExperimentSummary, ModelDescription, Task } from "../api/client";
import { useRegionProfiles } from "../hooks/useRegionProfiles";
import {
  activeFilterCount,
  EMPTY_EXPERIMENT_CATALOG,
  type ExperimentSort,
  readExperimentCatalogState,
  toExperimentListQuery,
  writeExperimentCatalogState,
} from "../api/experimentState";
import { Badge, Button, Callout, Checkbox, cn, SegmentedControl, Tooltip, ConfirmDialog, describeFields, ErrorBox, Field, initialValues, Input, jsonErrors, missingRequired, NumberInput, outOfRange, overrideCount, PageHeader, Panel, SchemaForm, Section, Select, SkeletonRows, Table, Tabs, ToggleChip, toOptions, type Column, type RawValues } from "@vitavision/lab-ui";
import { useDataset, useDatasets, useSplits } from "../hooks/useCatalog";
import { useAnnotationLabels } from "../hooks/useAnnotations";
import { TabScroll } from "./dataset/TabScroll";
import {
  CATALOGUE_PAGE,
  useCreateExperiment,
  useDeleteExperiment,
  useExperimentDeletionPreview,
  useExperimentPages,
  useInputSize,
  useModelTypes,
} from "../hooks/useExperiments";
import { refusalReason, toggleRun } from "../api/compareState";
import { clearDraft, draftKey, readDraft, writeDraft } from "../api/experimentDraft";
import { formatBytes } from "../api/format";
import { formatHeadline } from "../api/headline";
import { truthServesTask } from "../api/truth";
import { splitServesTask, SUPERVISED_TASKS } from "../hooks/useDatasetReadiness";
import { fullFrameProfile, inputSizeState, snapToMultiple } from "../api/inputSize";
import { experimentStatusTone } from "../api/statusTone";
import { defaultMethod, isRecommended, orderMethods, schemaForTask } from "../api/methodChoice";

type ExperimentRow = ExperimentSummary;

/**
 * A column header that orders the catalogue. The server sorts — a page is a slice of the
 * whole ordered list, so sorting the loaded rows here would order only what happened to
 * arrive. Created flips between newest and oldest; the others order one way.
 */
function SortHeader({
  label,
  sorts,
  current,
  onSort,
}: {
  label: string;
  /** The orders this column cycles through, the first one on the first press. */
  sorts: readonly ExperimentSort[];
  current: ExperimentSort;
  onSort: (sort: ExperimentSort) => void;
}) {
  const index = sorts.indexOf(current);
  const active = index >= 0;
  const next = sorts[(index + 1) % sorts.length] ?? sorts[0];
  const Icon = !active ? ArrowUpDown : current === "newest" ? ArrowDown : ArrowUp;
  return (
    <button
      type="button"
      onClick={() => next !== undefined && onSort(next)}
      className={cn(
        "inline-flex items-center gap-1 rounded-control font-inherit transition-colors hover:text-fg",
        active ? "text-fg" : "text-inherit",
      )}
      aria-label={`${label}, ${active ? `ordered ${SORT_LABEL[current]}` : "not ordered"}; order by ${SORT_LABEL[next ?? current]}`}
    >
      {label}
      <Icon className={cn("size-3", !active && "opacity-50")} aria-hidden />
    </button>
  );
}

const SORT_LABEL: Record<ExperimentSort, string> = {
  newest: "newest first",
  oldest: "oldest first",
  name: "by name",
  method: "by method",
  status: "by status",
};

/**
 * The catalogue itself: filters, table, deletion dialog, and no page chrome.
 *
 * It is mounted twice — as a screen of its own at `/experiments`, and as a tab under the
 * dataset band — so `datasetId` arrives as a prop rather than out of `useParams`. That is
 * what makes the two mounts honest: the Dataset column is genuinely absent inside a dataset
 * rather than incidentally so.
 */
export function ExperimentCatalog({ datasetId }: { datasetId?: number }) {
  const datasets = useDatasets();
  const methods = useModelTypes();
  const remove = useDeleteExperiment();
  const navigate = useNavigate();
  const [pendingDelete, setPendingDelete] = useState<ExperimentRow | null>(null);
  // Runs picked for a comparison. The first one anchors it, and the same rule the Compare
  // screen's picker uses decides what may join — so a selection made here always opens.
  const [picked, setPicked] = useState<number[]>([]);
  const deletionPreview = useExperimentDeletionPreview(pendingDelete?.id);
  const [searchParams, setSearchParams] = useSearchParams();
  const state = readExperimentCatalogState(searchParams);
  const experiments = useExperimentPages(toExperimentListQuery(state, datasetId));

  const datasetNames = new Map((datasets.data ?? []).map((entry) => [entry.id, entry.name]));
  const methodNames = new Map((methods.data?.methods ?? []).map((entry) => [entry.key, entry.title]));
  const activeFilters = activeFilterCount(state);

  const update = (patch: Partial<typeof state>) =>
    setSearchParams(writeExperimentCatalogState({ ...state, ...patch }), { replace: true });

  const rows = useMemo(
    () => (experiments.data?.pages ?? []).flatMap((page) => page.items),
    [experiments.data],
  );
  const total = experiments.data?.pages[0]?.total;
  const sortHeader = (label: string, sorts: readonly ExperimentSort[]) => (
    <SortHeader label={label} sorts={sorts} current={state.sort} onSort={(sort) => update({ sort })} />
  );
  const anchor = rows.find((row) => row.id === picked[0]);
  const togglePicked = (id: number) =>
    setPicked((current) => toggleRun(current, id));

  const columns: Column<ExperimentRow>[] = [
    {
      key: "pick",
      header: <span className="sr-only">Pick for comparison</span>,
      width: "2rem",
      cell: (row) => {
        const refusal = refusalReason(row, anchor, picked);
        return (
          <Tooltip content={refusal ?? "Pick for comparison"}>
            <span className="inline-flex">
              <Checkbox
                checked={picked.includes(row.id)}
                disabled={refusal !== null}
                onCheckedChange={() => togglePicked(row.id)}
                aria-label={`Pick ${row.name} for comparison`}
              />
            </span>
          </Tooltip>
        );
      },
    },
    {
      key: "name",
      header: sortHeader("Name", ["name"]),
      cell: (row) => (
        <Link
          to={`/experiments/${row.id}`}
          className="font-medium text-fg transition-colors hover:text-signal"
        >
          {row.name}
        </Link>
      ),
    },
    ...(datasetId === undefined
      ? [
          {
            key: "dataset",
            header: "Dataset",
            cell: (row: ExperimentRow) => (
              <Link
                to={`/datasets/${row.dataset_id}/experiments`}
                className="text-fg-muted transition-colors hover:text-signal"
              >
                {datasetNames.get(row.dataset_id) ?? `Dataset ${row.dataset_id}`}
              </Link>
            ),
          },
        ]
      : []),
    {
      key: "method",
      header: sortHeader("Method", ["method"]),
      cell: (row) => (
        <span className="flex flex-col">
          <span>{methodNames.get(row.model_type) ?? row.model_type}</span>
          <span className="font-mono text-[11px] text-fg-subtle">{row.model_type}</span>
        </span>
      ),
    },
    {
      key: "created",
      header: sortHeader("Created", ["newest", "oldest"]),
      cell: (row) => (
        <time dateTime={row.created_at} className="whitespace-nowrap text-xs text-fg-muted">
          {formatDate(row.created_at)}
        </time>
      ),
    },
    {
      key: "status",
      header: sortHeader("Status", ["status"]),
      cell: (row) => <Badge tone={experimentStatusTone(row.status)}>{row.status}</Badge>,
    },
    {
      key: "headline",
      header: "Headline",
      numeric: true,
      width: "7.5rem",
      // The task's own metric, labelled, so an IoU never sits unmarked beside an AUROC. A
      // metric that could not be computed is a dash, never a zero.
      cell: (row) => formatHeadline(row) ?? <span className="text-fg-subtle">—</span>,
    },
    {
      key: "actions",
      header: <span className="sr-only">Actions</span>,
      width: "2.25rem",
      cell: (row) => (
        <Button
          variant="ghost"
          size="sm"
          aria-label={`Delete ${row.name}`}
          icon={<Trash2 />}
          onClick={() => setPendingDelete(row)}
        />
      ),
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      <Panel
        title={
          total === undefined
            ? "Experiment history"
            : total === 1
              ? "1 experiment"
              : `${total} experiments`
        }
        actions={
          <div className="flex items-center gap-2">
            {picked.length > 0 && (
              <>
                <Button variant="ghost" size="sm" onClick={() => setPicked([])}>
                  Clear selection
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  disabled={picked.length < 2}
                  onClick={() => void navigate(`/compare?ids=${picked.join(",")}`)}
                >
                  {picked.length < 2 ? "Pick one more to compare" : `Compare ${picked.length}`}
                </Button>
              </>
            )}
            {activeFilters > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSearchParams(writeExperimentCatalogState(EMPTY_EXPERIMENT_CATALOG))}
              >
                Clear {activeFilters}
              </Button>
            )}
          </div>
        }
      >
        <div className="mb-4 flex flex-col gap-3 border-b border-line pb-4">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Field label="Search">
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-subtle"
                  aria-hidden
                />
                <Input
                  value={state.query}
                  onChange={(event) => update({ query: event.target.value })}
                  placeholder="Name, notes or #number"
                  className="pl-8"
                />
              </div>
            </Field>
            <Field as="group" label="Status">
              <Select
                aria-label="Status"
                value={state.status ?? ""}
                placeholder="Any status"
                options={[
                  { value: "draft", label: "Draft" },
                  { value: "training", label: "Training" },
                  { value: "trained", label: "Trained" },
                  { value: "failed", label: "Failed" },
                ]}
                onValueChange={(value) =>
                  update({ status: value ? (value as typeof state.status) : undefined })
                }
              />
            </Field>
            <Field label="Created from">
              <Input
                type="date"
                value={state.createdFrom ?? ""}
                max={state.createdTo}
                onChange={(event) => update({ createdFrom: event.target.value || undefined })}
              />
            </Field>
            <Field label="Created to">
              <Input
                type="date"
                value={state.createdTo ?? ""}
                min={state.createdFrom}
                onChange={(event) => update({ createdTo: event.target.value || undefined })}
              />
            </Field>
          </div>
          <div role="group" aria-label="Methods" className="flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-xs text-fg-muted">
              {state.modelTypes.length === 0 ? "Every method" : "Only"}
            </span>
            {(methods.data?.methods ?? []).map((method) => (
              <ToggleChip
                key={method.key}
                checked={state.modelTypes.includes(method.key)}
                onCheckedChange={(on) =>
                  update({
                    modelTypes: on
                      ? [...state.modelTypes, method.key]
                      : state.modelTypes.filter((key) => key !== method.key),
                  })
                }
                title={method.key}
              >
                {method.title}
              </ToggleChip>
            ))}
          </div>
        </div>

        {remove.error && <ErrorBox>{remove.error.message}</ErrorBox>}
        {experiments.isPending && <SkeletonRows rows={3} />}
        {experiments.error && !experiments.data && <ErrorBox>{experiments.error.message}</ErrorBox>}
        {experiments.data && (
          <Table
            columns={columns}
            rows={rows}
            rowKey={(row) => row.id}
            caption="Experiments"
            empty={
              activeFilters > 0
                ? "No experiments match these filters."
                : "No experiments yet. Create one to train a method on a split."
            }
          />
        )}
        {experiments.error && experiments.data && (
          <ErrorBox>{experiments.error.message}</ErrorBox>
        )}
        {experiments.hasNextPage && (
          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-xs text-fg-muted">
              {rows.length} of {total ?? rows.length} shown
            </span>
            <Button
              variant="secondary"
              size="sm"
              loading={experiments.isFetchingNextPage}
              onClick={() => void experiments.fetchNextPage()}
            >
              Load {Math.min(CATALOGUE_PAGE, (total ?? 0) - rows.length)} more
            </Button>
          </div>
        )}
      </Panel>

      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        title="Delete this experiment?"
        description={
          pendingDelete && (
            <>
              <span className="font-medium text-fg">{pendingDelete.name}</span>, its metrics,
              diagnostics and generated artifacts will be removed. Source dataset files are
              never touched.
              {deletionPreview.isPending && (
                <span className="mt-3 block text-fg-subtle">Inspecting generated files…</span>
              )}
              {deletionPreview.error && (
                <span className="mt-3 block text-defect">{deletionPreview.error.message}</span>
              )}
              {deletionPreview.data && (
                <span className="mt-3 block rounded-control border border-line bg-raised px-3 py-2">
                  <span className="block font-mono text-xs text-fg">
                    {deletionPreview.data.generated_files} generated files ·{" "}
                    {formatBytes(deletionPreview.data.generated_bytes)}
                  </span>
                  {deletionPreview.data.resident_loaded && (
                    <span className="mt-1 block text-xs">
                      The loaded inference worker will be evicted first.
                    </span>
                  )}
                  {deletionPreview.data.blocker && (
                    <span className="mt-1 block text-xs text-warn">
                      {deletionPreview.data.blocker}
                    </span>
                  )}
                </span>
              )}
            </>
          )
        }
        confirmLabel="Delete experiment"
        destructive
        loading={remove.isPending}
        disabled={!deletionPreview.data?.can_delete}
        onConfirm={() => {
          if (pendingDelete === null || !deletionPreview.data?.can_delete) return;
          remove.mutate(pendingDelete.id, { onSettled: () => setPendingDelete(null) });
        }}
      />
    </div>
  );
}

/** `/experiments` — the cross-dataset view, under `ReadingLayout`. */
export function ExperimentsRoute() {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader
        title="Experiments"
        actions={
          <Link to="/experiments/new">
            <Button variant="primary" icon={<Plus />}>
              New experiment
            </Button>
          </Link>
        }
      />
      <ExperimentCatalog />
    </div>
  );
}

/**
 * `/datasets/:id/experiments` — the tab.
 *
 * No header of its own: the band above already carries the dataset's name and the
 * New-experiment button, and a title here would only repeat the tab you just clicked. All
 * this contributes is its one scroll region.
 */
export function DatasetExperimentsRoute() {
  const datasetId = Number(useParams()["datasetId"]);

  return (
    <TabScroll>
      <ExperimentCatalog datasetId={datasetId} />
    </TabScroll>
  );
}

/** `/experiments/new` — the cross-dataset form, under `ReadingLayout`. */
export function CreateExperimentRoute() {
  return (
    <div className="flex flex-col gap-4">
      <PageHeader
        back={{ to: "/experiments", label: "Back to experiments" }}
        title="New experiment"
      />
      <CreateExperiment />
    </div>
  );
}

/**
 * `/datasets/:id/experiments/new` — a sub-page *of* the Experiments tab, not a tab.
 *
 * This back link survives where the tabs' did not: it points at a sibling inside the tab,
 * not up to a destination the strip already offers one click away.
 */
export function DatasetCreateExperimentRoute() {
  const datasetId = Number(useParams()["datasetId"]);

  return (
    <TabScroll>
      <div className="flex flex-col gap-3">
        <Link
          to={`/datasets/${datasetId}/experiments`}
          className="w-fit text-xs text-fg-muted transition-colors hover:text-signal"
        >
          ← Back to experiments
        </Link>
        <CreateExperiment initialDatasetId={datasetId} title="New experiment" />
      </div>
    </TabScroll>
  );
}


const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : DATE_FORMAT.format(date);
}


type ConfigTab = "method" | "preprocessing" | "evaluation";

const TASK_ORDER: Task[] = [
  "anomaly",
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];
const TASK_LABEL: Record<Task, string> = {
  anomaly: "Anomaly detection",
  few_shot_segmentation: "Few-shot segmentation",
  semantic_segmentation: "Segmentation",
  object_detection: "Object detection",
};

/**
 * `title` is how the dataset mount names the form without a second `<h1>` — the band's
 * dataset name is the only one on the screen, so this heading is the panel's, not the page's.
 */
function CreateExperiment({
  initialDatasetId,
  title,
}: {
  initialDatasetId?: number;
  title?: string;
}) {
  const navigate = useNavigate();
  const catalog = useModelTypes();
  const datasets = useDatasets();
  const create = useCreateExperiment();

  // Read once, on mount: the draft this form left behind when the reader followed one of
  // its own prerequisite links out of it.
  const storageKey = draftKey(initialDatasetId);
  const [draft] = useState(() => readDraft(storageKey));
  const pendingValues = useRef(draft);

  const [name, setName] = useState(draft?.name ?? "");
  const [datasetId, setDatasetId] = useState<number | undefined>(
    initialDatasetId ?? draft?.datasetId,
  );
  const [splitId, setSplitId] = useState<number | undefined>(draft?.splitId);
  const [regionProfileId, setRegionProfileId] = useState<number | undefined>(
    draft?.regionProfileId,
  );
  const [methodKey, setMethodKey] = useState<string | undefined>(draft?.methodKey);
  const [task, setTask] = useState<Task>("anomaly");
  const [targetLabel, setTargetLabel] = useState<string>("");
  const [configValues, setConfigValues] = useState<RawValues>({});
  const [preprocessingValues, setPreprocessingValues] = useState<RawValues>({});
  const [evaluationValues, setEvaluationValues] = useState<RawValues>({});
  const [channels, setChannels] = useState<string[]>(draft?.channels ?? []);
  // Empty means "the method's own size" — the empty-means-unset contract every option keeps.
  const [inputWidth, setInputWidth] = useState(draft?.width ?? "");
  const [inputHeight, setInputHeight] = useState(draft?.height ?? "");
  const [tab, setTab] = useState<ConfigTab>("method");
  // Field-level errors wait for the first press of Create: a form that opens covered in
  // red for fields nobody has reached yet is shouting, not helping.
  const [attempted, setAttempted] = useState(false);

  const splits = useSplits(datasetId);
  const dataset = useDataset(datasetId);
  const datasetChannels = dataset.data?.channels ?? [];
  const regionProfiles = useRegionProfiles(datasetId);
  const labels = useAnnotationLabels(datasetId);
  // A targeted task segments one class (ADR-0040); a `few_shot` split was drawn for one, so
  // it names the class unless the reader has chosen.
  const targeted = task === "few_shot_segmentation";
  // A supervised run — segmentation or detection — fits on annotated samples, which a drawn
  // split of normals never trains on.
  const supervised = SUPERVISED_TASKS.includes(task);
  const splitClass = splits.data?.find((entry) => entry.id === splitId)?.params.label_key ?? "";
  const effectiveTarget = targetLabel || splitClass;
  // The tasks any method can be run as (ADR-0039). One task is not a choice, so the picker
  // appears only when there are two — the same rule the channel chips follow.
  // A dataset of classes alone is not offered anomaly detection (ADR-0041).
  const tasks = TASK_ORDER.filter(
    (entry) =>
      (catalog.data?.methods ?? []).some((method) => method.capabilities.tasks.includes(entry)) &&
      truthServesTask(dataset.data?.truth, entry),
  );
  // The task starts at the first one the dataset serves: `anomaly` for an anomaly dataset,
  // few-shot segmentation for a dataset of classes.
  const [firstTask] = tasks;
  useEffect(() => {
    if (firstTask !== undefined && !tasks.includes(task)) setTask(firstTask);
  }, [firstTask, tasks, task]);

  // In the order a reader should weigh them: the task's default first, the floor last.
  const methodsForTask = useMemo(
    () =>
      orderMethods(
        (catalog.data?.methods ?? []).filter((entry) => entry.capabilities.tasks.includes(task)),
        task,
      ),
    [catalog.data, task],
  );
  // A task trains on its own kind of split: an anomaly run on a drawn or adopted partition,
  // a few-shot run on a split of references (ADR-0040), a supervised run on annotated
  // samples (ADR-0039). The others are not offered.
  const taskSplits = useMemo(
    () => (splits.data ?? []).filter((split) => splitServesTask(split, task)),
    [splits.data, task],
  );
  const method: ModelDescription | undefined = methodsForTask.find(
    (entry) => entry.key === methodKey,
  );

  const configFields = useMemo(
    () => (method ? describeFields(method.config_schema) : []),
    [method],
  );
  const preprocessingFields = useMemo(
    () => (catalog.data ? describeFields(catalog.data.preprocessing_schema) : []),
    [catalog.data],
  );
  // Only the options this task's evaluator reads: a field that would change nothing is not
  // a question to put to the reader.
  const evaluationFields = useMemo(
    () =>
      catalog.data ? describeFields(schemaForTask(catalog.data.evaluation_schema, task)) : [],
    [catalog.data, task],
  );
  // The size a run of this method with this configuration reads when none is typed: a DINO
  // backbone's patch decides it, so it follows the config rather than the method alone.
  const methodOptions = useMemo(
    () =>
      jsonErrors(configFields, configValues).length === 0
        ? toOptions(configFields, configValues)
        : {},
    [configFields, configValues],
  );
  const inputSize = useInputSize(method?.key, methodOptions);
  const size = inputSizeState(inputWidth, inputHeight, inputSize.data);

  // Default to the task's recommended method the moment the catalog lands, so the form is
  // never a blank screen waiting for a choice nobody knew they had to make.
  useEffect(() => {
    if (methodsForTask.length > 0 && !methodsForTask.some((entry) => entry.key === methodKey)) {
      setMethodKey(defaultMethod(methodsForTask, task)?.key);
    }
  }, [methodsForTask, methodKey, task]);

  // Each group starts from its schema's defaults, with the draft laid over it once — and the
  // method's values only for the method they were typed for.
  useEffect(() => {
    const restored = pendingValues.current;
    setConfigValues(
      restoreValues(
        configFields,
        restored?.methodKey === methodKey ? restored?.configValues : undefined,
      ),
    );
    if (restored && configFields.length > 0) restored.configValues = {};
    // `methodKey` changes together with `configFields`; the fields are the trigger.
  }, [configFields]);
  useEffect(
    () =>
      setPreprocessingValues(
        restoreValues(preprocessingFields, pendingValues.current?.preprocessingValues),
      ),
    [preprocessingFields],
  );
  useEffect(
    () =>
      setEvaluationValues(restoreValues(evaluationFields, pendingValues.current?.evaluationValues)),
    [evaluationFields],
  );

  // Where to look defaults to the dataset's own "Full frame" — or the only profile there is.
  useEffect(() => {
    if (regionProfileId !== undefined || regionProfiles.data === undefined) return;
    const only = regionProfiles.data.length === 1 ? regionProfiles.data[0] : undefined;
    const fallback = fullFrameProfile(regionProfiles.data) ?? only;
    if (fallback) setRegionProfileId(fallback.id);
  }, [regionProfiles.data, regionProfileId]);
  useEffect(() => {
    if (splits.data === undefined) return;
    // A split chosen for another task is not a choice for this one.
    if (splitId !== undefined && !taskSplits.some((split) => split.id === splitId)) {
      setSplitId(undefined);
      return;
    }
    const only = taskSplits.length === 1 ? taskSplits[0] : undefined;
    if (splitId === undefined && only) setSplitId(only.id);
  }, [splits.data, taskSplits, splitId]);

  useEffect(() => {
    writeDraft(storageKey, {
      name,
      datasetId,
      splitId,
      regionProfileId,
      methodKey,
      configValues,
      preprocessingValues,
      evaluationValues,
      channels,
      width: inputWidth,
      height: inputHeight,
    });
  }, [
    storageKey,
    name,
    datasetId,
    splitId,
    regionProfileId,
    methodKey,
    configValues,
    preprocessingValues,
    evaluationValues,
    channels,
    inputWidth,
    inputHeight,
  ]);

  const datasetName = datasets.data?.find((entry) => entry.id === datasetId)?.name;
  // The name a reader would have typed anyway. Used when the field is left empty, so a name
  // is never the thing standing between a reader and a run.
  const suggestedName =
    method && datasetName ? `${method.title} on ${datasetName}` : "";
  const effectiveName = name.trim() || suggestedName;

  const blocking = [
    ...jsonErrors(configFields, configValues),
    ...missingRequired(configFields, configValues),
    ...outOfRange(configFields, configValues),
    ...outOfRange(preprocessingFields, preprocessingValues),
    ...outOfRange(evaluationFields, evaluationValues),
  ];

  // Said out loud rather than left to a greyed-out button. "Why can't I press this" is a
  // question the screen has to answer without anyone reading the source.
  const missing: string[] = [];
  const fieldError: { name?: string; dataset?: string; size?: string; split?: string } = {};
  if (effectiveName === "") {
    missing.push("a name");
    fieldError.name = "Name the run.";
  }
  if (datasetId === undefined) {
    missing.push("a dataset");
    fieldError.dataset = "Choose the dataset to train on.";
  }
  if (splitId === undefined) {
    missing.push("a split");
    fieldError.split = "Choose which samples train and which are scored.";
  }
  if (size.error !== undefined) {
    missing.push("an input size the method reads");
    fieldError.size = size.error;
  }
  if (targeted && effectiveTarget === "") missing.push("a target class");
  if (method && !method.availability.available) missing.push("a method you can run");

  const ready =
    missing.length === 0 && methodKey !== undefined && blocking.length === 0;

  const submit = () => {
    setAttempted(true);
    if (
      !ready ||
      methodKey === undefined ||
      datasetId === undefined ||
      splitId === undefined
    ) {
      return;
    }
    create.mutate(
      {
        name: effectiveName,
        dataset_id: datasetId,
        split_id: splitId,
        // Omitted, the dataset's "Full frame" — the same default the field shows.
        ...(regionProfileId === undefined ? {} : { region_profile_id: regionProfileId }),
        // Omitted, the method's own size for this configuration.
        ...(size.named === undefined ? {} : size.named),
        model_type: methodKey,
        task,
        target_label: targeted ? effectiveTarget : null,
        config: toOptions(configFields, configValues),
        preprocessing: toOptions(preprocessingFields, preprocessingValues),
        evaluation: toOptions(evaluationFields, evaluationValues),
        channels,
      },
      {
        onSuccess: (created) => {
          clearDraft(storageKey);
          void navigate(`/experiments/${created.id}`);
        },
      },
    );
  };

  const noSplits = datasetId !== undefined && splits.data !== undefined && taskSplits.length === 0;
  // With one task there is nothing to choose, and the form starts at its inputs.
  const offset = tasks.length > 1 ? 1 : 0;

  return (
    <Panel title={title}>
      <div className="flex flex-col gap-7">
        {catalog.error && <ErrorBox>{catalog.error.message}</ErrorBox>}

        {tasks.length > 1 && (
          <Section
            step={1}
            title="Task"
            hint="What the run is asked to do. It decides the split, the methods and the results."
          >
            <SegmentedControl
              aria-label="Task"
              value={task}
              options={tasks.map((entry) => ({ value: entry, label: TASK_LABEL[entry] }))}
              onValueChange={(value) => setTask(value as Task)}
            />
          </Section>
        )}

        <Section step={1 + offset} title="What to train on">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Field label="Name" error={attempted ? fieldError.name : undefined}>
              <Input
                aria-label="Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder={suggestedName || "Name this run"}
              />
            </Field>

            <Field
              as="group"
              label="Dataset"
              error={attempted ? fieldError.dataset : undefined}
            >
              <Select
                aria-label="Dataset"
                value={datasetId === undefined ? "" : String(datasetId)}
                placeholder="Choose a dataset…"
                disabled={initialDatasetId !== undefined}
                options={(datasets.data ?? []).map((dataset) => ({
                  value: String(dataset.id),
                  label: dataset.name,
                }))}
                onValueChange={(value) => {
                  setDatasetId(value === "" ? undefined : Number(value));
                  // A split belongs to one dataset, so the old choice is now meaningless
                  // rather than merely stale.
                  setSplitId(undefined);
                  setRegionProfileId(undefined);
                }}
              />
            </Field>

            <Field
              as="group"
              label="Region profile"
              description={
                datasetId === undefined ? undefined : (
                  <>
                    Where to look. The run prepares it at its own size when it trains.{" "}
                    <Link
                      className="text-signal underline underline-offset-2"
                      to={`/datasets/${datasetId}/prepare`}
                    >
                      Define another
                    </Link>
                    .
                  </>
                )
              }
            >
              <Select
                aria-label="Region profile"
                value={regionProfileId === undefined ? "" : String(regionProfileId)}
                placeholder={datasetId === undefined ? "Pick a dataset first" : "Full frame"}
                disabled={datasetId === undefined}
                options={(regionProfiles.data ?? []).map((profile) => ({
                  value: String(profile.id),
                  label: `${profile.name} · r${profile.revision_no}`,
                  note: profile.extractor_type,
                }))}
                onValueChange={(value) =>
                  setRegionProfileId(value === "" ? undefined : Number(value))
                }
              />
            </Field>

            <Field
              as="group"
              label="Input size"
              error={attempted ? fieldError.size : undefined}
              description={size.caption}
            >
              <div className="grid grid-cols-2 gap-2">
                <NumberInput
                  aria-label="Input width"
                  min={8}
                  max={2048}
                  step={size.multiple}
                  value={inputWidth}
                  placeholder={inputSize.data ? String(inputSize.data.width) : "auto"}
                  onChange={(event) => setInputWidth(event.target.value)}
                  onBlur={() => setInputWidth(snapToMultiple(inputWidth, size.multiple))}
                />
                <NumberInput
                  aria-label="Input height"
                  min={8}
                  max={2048}
                  step={size.multiple}
                  value={inputHeight}
                  placeholder={inputSize.data ? String(inputSize.data.height) : "auto"}
                  onChange={(event) => setInputHeight(event.target.value)}
                  onBlur={() => setInputHeight(snapToMultiple(inputHeight, size.multiple))}
                />
              </div>
            </Field>

            <Field
              as="group"
              label="Split"
              error={attempted ? fieldError.split : undefined}
              description={
                noSplits ? (
                  <>
                    {targeted
                      ? "No split of references yet."
                      : supervised
                        ? "No split of annotated samples yet."
                        : "This dataset has no splits."}{" "}
                    <Link
                      className="text-signal underline underline-offset-2"
                      to={`/datasets/${datasetId}/splits${
                        targeted
                          ? "?strategy=few_shot"
                          : supervised
                            ? "?strategy=class_stratified"
                            : ""
                      }`}
                    >
                      {targeted ? "Choose references" : supervised ? "Draw one by class" : "Create one"}
                    </Link>
                    .
                  </>
                ) : undefined
              }
            >
              <Select
                aria-label="Split"
                value={splitId === undefined ? "" : String(splitId)}
                placeholder={datasetId === undefined ? "Pick a dataset first" : "Choose a split…"}
                disabled={datasetId === undefined}
                options={taskSplits.map((split) => ({
                  value: String(split.id),
                  label: split.name,
                  note: split.strategy,
                }))}
                onValueChange={(value) => setSplitId(value === "" ? undefined : Number(value))}
              />
            </Field>

            {targeted && (
              <Field as="group" label="Target class">
                <Select
                  aria-label="Target class"
                  value={effectiveTarget}
                  placeholder="Choose a class"
                  options={(labels.data ?? []).map((label) => ({
                    value: label.key,
                    label: label.name,
                    note: label.key,
                  }))}
                  onValueChange={setTargetLabel}
                />
              </Field>
            )}

            {/* Absent entirely for a single-view dataset: there is nothing to select, and
                a control offering one option is a question with no answer. Two channels
                render two chips — nothing here knows how many there should be. */}
            {datasetChannels.length > 0 && (
              <Field
                as="group"
                label="Channels"
                description={
                  channels.length === 0
                    ? "All channels. Pick some to train and score on a subset."
                    : `${channels.length} of ${datasetChannels.length} channels.`
                }
              >
                <div className="flex flex-wrap gap-2">
                  {datasetChannels.map((channel) => (
                    <ToggleChip
                      key={channel.id}
                      checked={channels.includes(channel.name)}
                      onCheckedChange={(checked) =>
                        setChannels((current) =>
                          checked
                            ? [...current, channel.name]
                            : current.filter((name) => name !== channel.name),
                        )
                      }
                    >
                      {channel.name}
                    </ToggleChip>
                  ))}
                </div>
              </Field>
            )}
          </div>
        </Section>

        <Section step={2 + offset} title="Method">
          {catalog.isPending && <SkeletonRows rows={2} />}
          <div className="grid gap-3 sm:grid-cols-2">
            {methodsForTask.map((entry) => (
              <MethodCard
                key={entry.key}
                method={entry}
                recommended={isRecommended(entry, task)}
                selected={entry.key === methodKey}
                onSelect={() => setMethodKey(entry.key)}
              />
            ))}
          </div>
        </Section>

        <Section
          step={3 + offset}
          title="Configuration"
          hint="Anything left alone uses the backend's own default."
        >
          <div className="flex flex-col gap-4">
            <Tabs
              label="Configuration group"
              active={tab}
              onSelect={setTab}
              items={[
                {
                  id: "method",
                  label: method ? method.title : "Method",
                  count: overrideCount(configFields, configValues),
                },
                {
                  id: "preprocessing",
                  label: "Colour",
                  count: overrideCount(preprocessingFields, preprocessingValues),
                },
                // Absent when this task's evaluator reads none of the options.
                ...(evaluationFields.length > 0
                  ? [
                      {
                        id: "evaluation" as const,
                        label: "Evaluation",
                        count: overrideCount(evaluationFields, evaluationValues),
                      },
                    ]
                  : []),
              ]}
            />

            {tab === "method" &&
              (method ? (
                <SchemaForm
                  fields={configFields}
                  values={configValues}
                  onChange={setConfigValues}
                />
              ) : (
                <p className="text-sm text-fg-muted">Choose a method to configure it.</p>
              ))}
            {tab === "preprocessing" && (
              <SchemaForm
                fields={preprocessingFields}
                values={preprocessingValues}
                onChange={setPreprocessingValues}
              />
            )}
            {tab === "evaluation" && evaluationFields.length > 0 && (
              <SchemaForm
                fields={evaluationFields}
                values={evaluationValues}
                onChange={setEvaluationValues}
              />
            )}
          </div>
        </Section>

        {blocking.length > 0 && (
          <Callout tone="warning" title="Some options need a second look">
            {blocking.join(", ")}
          </Callout>
        )}
        {create.error && <ErrorBox>{create.error.message}</ErrorBox>}

        <div className="flex items-center gap-3 border-t border-line pt-4">
          {/* Pressable while incomplete: pressing it is how the reader finds out what,
              beside the field that needs it rather than in a sentence down here only. */}
          <Button variant="primary" loading={create.isPending} onClick={submit}>
            Create experiment
          </Button>
          {missing.length > 0 && (
            <p className="text-xs text-fg-muted">Still needs {joinWords(missing)}.</p>
          )}
        </div>
      </div>
    </Panel>
  );
}

/** A group's schema defaults with whatever the draft kept for the same fields laid over. */
function restoreValues(fields: ReturnType<typeof describeFields>, saved?: RawValues): RawValues {
  const values = initialValues(fields);
  if (!saved) return values;
  for (const field of fields) {
    const value = saved[field.name];
    if (value !== undefined) values[field.name] = value;
  }
  return values;
}

/** `a, b and c` — a sentence, since this one is read as one. */
function joinWords(words: string[]): string {
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

/**
 * One method, as a thing you press.
 *
 * A native radio underneath, so a keyboard reaches the group with Tab and moves inside it
 * with the arrow keys, and the whole card is the hit target.
 */
function MethodCard({
  method,
  recommended,
  selected,
  onSelect,
}: {
  method: ModelDescription;
  /** The registry's default for the task being configured. */
  recommended: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const capabilities = method.capabilities;
  const unavailable = !method.availability.available;

  return (
    <label
      className={cn(
        "relative flex cursor-pointer flex-col gap-2 rounded-panel border p-3 transition-colors",
        "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-signal",
        selected
          ? "border-signal bg-signal/5 ring-1 ring-signal"
          : "border-line bg-raised/40 hover:border-line-strong",
      )}
    >
      <input
        type="radio"
        name="method"
        value={method.key}
        checked={selected}
        onChange={onSelect}
        className="absolute inset-0 cursor-pointer opacity-0"
      />

      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-semibold tracking-tight text-fg">{method.title}</span>
        <span className="font-mono text-[11px] text-fg-subtle">{method.key}</span>
      </div>

      <p className="text-xs leading-snug text-fg-muted">{method.summary}</p>

      <div className="flex flex-wrap gap-1.5">
        {/* The registry's verdict, first on the card: a gate decided it, and it is what a
            reader choosing between methods needs before any capability. */}
        {recommended && <Badge tone="normal">recommended</Badge>}
        {method.status === "experimental" && <Badge tone="warning">experimental</Badge>}
        {method.status === "floor" && <Badge tone="neutral">floor</Badge>}
        {capabilities.dataset_specific && <Badge tone="warning">dataset-specific</Badge>}
        {/* A segmenter's map is a foreground probability, not an anomaly map. */}
        {capabilities.produces_anomaly_map && (
          <Badge tone="info">
            {capabilities.tasks.includes("anomaly") ? "anomaly maps" : "probability maps"}
          </Badge>
        )}
        {capabilities.produces_diagnostics && <Badge tone="info">diagnostics</Badge>}
        {capabilities.channel_aware && <Badge tone="info">channel-aware</Badge>}
        {capabilities.portable_formats.map((format) => (
          <Badge key={format} tone="neutral">
            {format.toUpperCase()} export
          </Badge>
        ))}
        {!capabilities.requires_training && <Badge tone="neutral">no training</Badge>}
        <Badge tone="neutral">
          <span className="font-mono">{capabilities.preferred_device}</span>
        </Badge>
      </div>

      {/* Stated on the card rather than in a banner elsewhere: this is where the reader
          asks the question, so this is where it has to be answered. Nothing went wrong —
          a dependency is simply not installed — so it is a caveat, not an error. */}
      {unavailable && method.availability.reason && (
        <p className="rounded-control border border-warn/30 bg-warn/8 px-2 py-1.5 text-xs leading-snug text-fg-muted">
          {method.availability.reason}
        </p>
      )}
    </label>
  );
}
