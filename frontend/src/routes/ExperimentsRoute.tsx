/**
 * The experiment list, and the form that creates one.
 *
 * The method picker, the configuration form and the preprocessing form are all generated
 * from schemas the backend serves (`useModelTypes`). Nothing in this file names a method,
 * knows a hyperparameter, or has an opinion about what EfficientAD needs — which is the
 * claim ADR-0007 makes, tested by the fact that `pixel_reference` and
 * `efficientad_anomalib` have nothing in common and both render here.
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

import { Plus, Search, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import type { ModelDescription } from "../api/client";
import {
  useRegionBuild,
  useRegionProfiles,
} from "../hooks/useRegionProfiles";
import {
  EMPTY_EXPERIMENT_CATALOG,
  readExperimentCatalogState,
  toExperimentListQuery,
  writeExperimentCatalogState,
} from "../api/experimentState";
import {
  describeFields,
  initialValues,
  jsonErrors,
  missingRequired,
  outOfRange,
  overrideCount,
  toOptions,
} from "../api/schemaForm";
import type { RawValues } from "../api/schemaForm";
import { SchemaForm } from "../components/SchemaForm";
import { Tabs } from "../components/Tabs";
import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  ErrorBox,
  Field,
  Input,
  PageHeader,
  Panel,
  Section,
  Select,
  SkeletonRows,
  Table,
  cn,
  type Column,
  type Tone,
} from "../components/ui";
import { useDatasets, useSplits } from "../hooks/useCatalog";
import { TabScroll } from "./dataset/TabScroll";
import {
  useCreateExperiment,
  useDeleteExperiment,
  useExperimentDeletionPreview,
  useExperiments,
  useModelTypes,
} from "../hooks/useExperiments";

type ExperimentRow = NonNullable<ReturnType<typeof useExperiments>["data"]>[number];

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
  const [pendingDelete, setPendingDelete] = useState<ExperimentRow | null>(null);
  const deletionPreview = useExperimentDeletionPreview(pendingDelete?.id);
  const [searchParams, setSearchParams] = useSearchParams();
  const state = readExperimentCatalogState(searchParams);
  const experiments = useExperiments(toExperimentListQuery(state, datasetId));

  const datasetNames = new Map((datasets.data ?? []).map((entry) => [entry.id, entry.name]));
  const methodNames = new Map((methods.data?.methods ?? []).map((entry) => [entry.key, entry.title]));
  const activeFilters = [state.query || undefined, state.modelType, state.status].filter(Boolean).length;

  const update = (patch: Partial<typeof state>) =>
    setSearchParams(writeExperimentCatalogState({ ...state, ...patch }), { replace: true });

  const columns: Column<ExperimentRow>[] = [
    {
      key: "name",
      header: "Name",
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
      header: "Method",
      cell: (row) => (
        <span className="flex flex-col">
          <span>{methodNames.get(row.model_type) ?? row.model_type}</span>
          <span className="font-mono text-[11px] text-fg-subtle">{row.model_type}</span>
        </span>
      ),
    },
    {
      key: "created",
      header: "Created",
      cell: (row) => (
        <time dateTime={row.created_at} className="whitespace-nowrap text-xs text-fg-muted">
          {formatDate(row.created_at)}
        </time>
      ),
    },
    {
      key: "status",
      header: "Status",
      cell: (row) => <Badge tone={statusTone(row.status)}>{row.status}</Badge>,
    },
    {
      key: "auroc",
      header: "AUROC",
      numeric: true,
      width: "6rem",
      cell: (row) =>
        // A metric that could not be computed is a dash, never a zero.
        row.headline_roc_auc === null || row.headline_roc_auc === undefined ? (
          <span className="text-fg-subtle">—</span>
        ) : (
          row.headline_roc_auc.toFixed(3)
        ),
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
        title={experiments.data ? `${experiments.data.length} experiments` : "Experiment history"}
        actions={
          activeFilters > 0 ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSearchParams(writeExperimentCatalogState(EMPTY_EXPERIMENT_CATALOG))}
            >
              Clear {activeFilters}
            </Button>
          ) : undefined
        }
      >
        <div className="mb-4 grid gap-3 border-b border-line pb-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Search">
            <div className="relative">
              <Search
                className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-fg-subtle"
                aria-hidden
              />
              <Input
                value={state.query}
                onChange={(event) => update({ query: event.target.value })}
                placeholder="Name or notes"
                className="pl-8"
              />
            </div>
          </Field>
          <Field as="group" label="Method">
            <Select
              aria-label="Method"
              value={state.modelType ?? ""}
              placeholder="Any method"
              options={(methods.data?.methods ?? []).map((method) => ({
                value: method.key,
                label: method.title,
                note: method.key,
              }))}
              onValueChange={(value) => update({ modelType: value || undefined })}
            />
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
          <Field as="group" label="Order">
            <Select
              aria-label="Order"
              value={state.sort}
              options={[
                { value: "newest", label: "Newest first" },
                { value: "oldest", label: "Oldest first" },
                { value: "name", label: "Name" },
              ]}
              onValueChange={(value) => update({ sort: value as typeof state.sort })}
            />
          </Field>
        </div>

        {remove.error && <ErrorBox>{remove.error.message}</ErrorBox>}
        {experiments.isPending && <SkeletonRows rows={3} />}
        {experiments.error && <ErrorBox>{experiments.error.message}</ErrorBox>}
        {experiments.data && (
          <Table
            columns={columns}
            rows={experiments.data}
            rowKey={(row) => row.id}
            caption="Experiments"
            empty={
              activeFilters > 0
                ? "No experiments match these filters."
                : "No experiments yet. Create one to train a method on a split."
            }
          />
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

function statusTone(status: string): Tone {
  if (status === "trained") return "normal";
  if (status === "failed") return "defect";
  if (status === "training") return "info";
  return "unlabeled";
}

const DATE_FORMAT = new Intl.DateTimeFormat(undefined, {
  dateStyle: "medium",
  timeStyle: "short",
});

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : DATE_FORMAT.format(date);
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (amount < 1024) break;
    amount /= 1024;
    unit = next;
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
}

type ConfigTab = "method" | "preprocessing" | "evaluation";

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

  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState<number | undefined>(initialDatasetId);
  const [splitId, setSplitId] = useState<number | undefined>();
  const [regionProfileId, setRegionProfileId] = useState<number | undefined>();
  const [methodKey, setMethodKey] = useState<string | undefined>();
  const [configValues, setConfigValues] = useState<RawValues>({});
  const [preprocessingValues, setPreprocessingValues] = useState<RawValues>({});
  const [evaluationValues, setEvaluationValues] = useState<RawValues>({});
  const [tab, setTab] = useState<ConfigTab>("method");

  const splits = useSplits(datasetId);
  const regionProfiles = useRegionProfiles(datasetId);
  const regionBuild = useRegionBuild(regionProfileId);
  const method: ModelDescription | undefined = catalog.data?.methods.find(
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
  const evaluationFields = useMemo(
    () => (catalog.data ? describeFields(catalog.data.evaluation_schema) : []),
    [catalog.data],
  );

  // Default to the first method the moment the catalog lands, so the form is never a
  // blank screen waiting for a choice nobody knew they had to make.
  useEffect(() => {
    if (methodKey === undefined && catalog.data && catalog.data.methods.length > 0) {
      setMethodKey(catalog.data.methods[0]?.key);
    }
  }, [catalog.data, methodKey]);

  useEffect(() => setConfigValues(initialValues(configFields)), [configFields]);
  useEffect(() => setPreprocessingValues(initialValues(preprocessingFields)), [preprocessingFields]);
  useEffect(() => setEvaluationValues(initialValues(evaluationFields)), [evaluationFields]);

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
  if (name.trim() === "") missing.push("a name");
  if (datasetId === undefined) missing.push("a dataset");
  if (splitId === undefined) missing.push("a split");
  if (regionProfileId === undefined) missing.push("a prepared input");
  else if (!regionBuild.data || regionBuild.data.failed > 0) {
    missing.push("a complete region build");
  }
  if (method && !method.availability.available) missing.push("a method you can run");

  const ready =
    missing.length === 0 && methodKey !== undefined && blocking.length === 0;

  const submit = () => {
    if (
      !ready ||
      methodKey === undefined ||
      datasetId === undefined ||
      splitId === undefined ||
      regionProfileId === undefined
    ) {
      return;
    }
    create.mutate(
      {
        name: name.trim(),
        dataset_id: datasetId,
        split_id: splitId,
        region_profile_id: regionProfileId,
        model_type: methodKey,
        config: toOptions(configFields, configValues),
        preprocessing: toOptions(preprocessingFields, preprocessingValues),
        evaluation: toOptions(evaluationFields, evaluationValues),
      },
      { onSuccess: (created) => void navigate(`/experiments/${created.id}`) },
    );
  };

  const noSplits = datasetId !== undefined && splits.data?.length === 0;

  return (
    <Panel title={title}>
      <div className="flex flex-col gap-7">
        {catalog.error && <ErrorBox>{catalog.error.message}</ErrorBox>}

        <Section step={1} title="What to train on">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Field label="Name">
              <Input
                aria-label="Name"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="efficientad on candle"
              />
            </Field>

            <Field as="group" label="Dataset">
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
              label="Prepared input"
              description={
                datasetId !== undefined && regionProfiles.data?.length === 0 ? (
                  <>
                    No region profiles yet.{" "}
                    <Link
                      className="text-signal underline underline-offset-2"
                      to={`/datasets/${datasetId}/prepare`}
                    >
                      Prepare the dataset
                    </Link>
                    .
                  </>
                ) : regionProfileId !== undefined && regionBuild.error ? (
                  <>
                    This revision is not built.{" "}
                    <Link
                      className="text-signal underline underline-offset-2"
                      to={`/datasets/${datasetId}/prepare`}
                    >
                      Build it
                    </Link>
                    .
                  </>
                ) : regionBuild.data?.failed ? (
                  `${regionBuild.data.failed} images failed preparation; create a corrected revision.`
                ) : undefined
              }
            >
              <Select
                aria-label="Prepared input"
                value={regionProfileId === undefined ? "" : String(regionProfileId)}
                placeholder={datasetId === undefined ? "Pick a dataset first" : "Choose a revision…"}
                disabled={datasetId === undefined}
                options={(regionProfiles.data ?? []).map((profile) => ({
                  value: String(profile.id),
                  label: `${profile.name} · r${profile.revision_no}`,
                  note: `${profile.prepared_width}×${profile.prepared_height}`,
                }))}
                onValueChange={(value) =>
                  setRegionProfileId(value === "" ? undefined : Number(value))
                }
              />
            </Field>

            <Field
              as="group"
              label="Split"
              description={
                noSplits ? (
                  <>
                    This dataset has no splits.{" "}
                    <Link
                      className="text-signal underline underline-offset-2"
                      to={`/datasets/${datasetId}/splits`}
                    >
                      Create one
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
                options={(splits.data ?? []).map((split) => ({
                  value: String(split.id),
                  label: split.name,
                  note: split.strategy,
                }))}
                onValueChange={(value) => setSplitId(value === "" ? undefined : Number(value))}
              />
            </Field>
          </div>
        </Section>

        <Section step={2} title="Method">
          {catalog.isPending && <SkeletonRows rows={2} />}
          <div className="grid gap-3 sm:grid-cols-2">
            {catalog.data?.methods.map((entry) => (
              <MethodCard
                key={entry.key}
                method={entry}
                selected={entry.key === methodKey}
                onSelect={() => setMethodKey(entry.key)}
              />
            ))}
          </div>
        </Section>

        <Section
          step={3}
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
                  label: "Model input",
                  count: overrideCount(preprocessingFields, preprocessingValues),
                },
                {
                  id: "evaluation",
                  label: "Evaluation",
                  count: overrideCount(evaluationFields, evaluationValues),
                },
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
            {tab === "evaluation" && (
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
          <Button
            variant="primary"
            loading={create.isPending}
            disabled={!ready}
            onClick={submit}
          >
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
  selected,
  onSelect,
}: {
  method: ModelDescription;
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
        {capabilities.dataset_specific && <Badge tone="warning">dataset-specific</Badge>}
        {capabilities.produces_anomaly_map && <Badge tone="info">anomaly maps</Badge>}
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
