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

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { ModelDescription } from "../api/client";
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
import { useCreateExperiment, useExperiments, useModelTypes } from "../hooks/useExperiments";

type ExperimentRow = NonNullable<ReturnType<typeof useExperiments>["data"]>[number];

export function ExperimentsRoute() {
  const experiments = useExperiments();

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
    {
      key: "method",
      header: "Method",
      cell: (row) => <span className="font-mono text-xs text-fg-muted">{row.model_type}</span>,
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
  ];

  return (
    <div className="flex flex-col gap-6">
      <PageHeader title="Experiments" />

      <CreateExperiment />

      <Panel title="All experiments">
        {experiments.isPending && <SkeletonRows rows={3} />}
        {experiments.error && <ErrorBox>{experiments.error.message}</ErrorBox>}
        {experiments.data && (
          <Table
            columns={columns}
            rows={experiments.data}
            rowKey={(row) => row.id}
            caption="Experiments"
            empty="No experiments yet. Create one above to train a method on a split."
          />
        )}
      </Panel>
    </div>
  );
}

function statusTone(status: string): Tone {
  if (status === "trained") return "normal";
  if (status === "failed") return "defect";
  if (status === "training") return "info";
  return "unlabeled";
}

type ConfigTab = "method" | "preprocessing" | "evaluation";

function CreateExperiment() {
  const navigate = useNavigate();
  const catalog = useModelTypes();
  const datasets = useDatasets();
  const create = useCreateExperiment();

  const [name, setName] = useState("");
  const [datasetId, setDatasetId] = useState<number | undefined>();
  const [splitId, setSplitId] = useState<number | undefined>();
  const [methodKey, setMethodKey] = useState<string | undefined>();
  const [configValues, setConfigValues] = useState<RawValues>({});
  const [preprocessingValues, setPreprocessingValues] = useState<RawValues>({});
  const [evaluationValues, setEvaluationValues] = useState<RawValues>({});
  const [tab, setTab] = useState<ConfigTab>("method");

  const splits = useSplits(datasetId);
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
  if (method && !method.availability.available) missing.push("a method you can run");

  const ready =
    missing.length === 0 && methodKey !== undefined && blocking.length === 0;

  const submit = () => {
    if (!ready || methodKey === undefined || datasetId === undefined || splitId === undefined) {
      return;
    }
    create.mutate(
      {
        name: name.trim(),
        dataset_id: datasetId,
        split_id: splitId,
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
    <Panel title="New experiment">
      <div className="flex flex-col gap-7">
        {catalog.error && <ErrorBox>{catalog.error.message}</ErrorBox>}

        <Section step={1} title="What to train on">
          <div className="grid gap-4 sm:grid-cols-3">
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
                options={(datasets.data ?? []).map((dataset) => ({
                  value: String(dataset.id),
                  label: dataset.name,
                }))}
                onValueChange={(value) => {
                  setDatasetId(value === "" ? undefined : Number(value));
                  // A split belongs to one dataset, so the old choice is now meaningless
                  // rather than merely stale.
                  setSplitId(undefined);
                }}
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
                  label: "Preprocessing",
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
