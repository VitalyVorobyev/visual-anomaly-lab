/**
 * The experiment list, and the form that creates one.
 *
 * The method picker, the configuration form and the preprocessing form are all generated
 * from schemas the backend serves (`useModelTypes`). Nothing in this file names a method,
 * knows a hyperparameter, or has an opinion about what EfficientAD needs — which is the
 * claim ADR-0007 makes, tested by the fact that `pixel_reference` and
 * `efficientad_anomalib` have nothing in common and both render here.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";

import type { ModelDescription } from "../api/client";
import {
  describeFields,
  initialValues,
  jsonErrors,
  missingRequired,
  toOptions,
} from "../api/schemaForm";
import type { RawValues } from "../api/schemaForm";
import { SchemaForm } from "../components/SchemaForm";
import {
  Badge,
  Button,
  Empty,
  ErrorBox,
  Field,
  Panel,
  inputClasses,
} from "../components/ui";
import { useDatasets, useSplits } from "../hooks/useCatalog";
import { useCreateExperiment, useExperiments, useModelTypes } from "../hooks/useExperiments";

export function ExperimentsRoute() {
  const experiments = useExperiments();

  return (
    <div className="flex flex-col gap-6">
      <CreateExperiment />

      <Panel title="Experiments">
        {experiments.isPending && <p className="text-sm text-slate-500">Loading…</p>}
        {experiments.error && <ErrorBox>{experiments.error.message}</ErrorBox>}
        {experiments.data?.length === 0 && (
          <Empty>No experiments yet. Create one above to train a method on a split.</Empty>
        )}

        {experiments.data && experiments.data.length > 0 && (
          <ul className="divide-y divide-slate-200 dark:divide-slate-700">
            {experiments.data.map((experiment) => (
              <li key={experiment.id} className="flex items-center gap-3 py-2">
                <Link
                  to={`/experiments/${experiment.id}`}
                  className="text-sm font-medium hover:underline"
                >
                  {experiment.name}
                </Link>
                <span className="font-mono text-xs text-slate-500">{experiment.model_type}</span>
                <Badge tone={statusTone(experiment.status)}>{experiment.status}</Badge>
                <span className="ml-auto font-mono text-sm">
                  {experiment.headline_roc_auc === null ||
                  experiment.headline_roc_auc === undefined ? (
                    <span className="text-slate-400">not scored</span>
                  ) : (
                    <>AUROC {experiment.headline_roc_auc.toFixed(3)}</>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}

function statusTone(status: string) {
  if (status === "trained") return "normal" as const;
  if (status === "failed") return "defect" as const;
  if (status === "training") return "info" as const;
  return "unlabeled" as const;
}

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
  ];
  const ready =
    name.trim() !== "" &&
    datasetId !== undefined &&
    splitId !== undefined &&
    methodKey !== undefined &&
    method?.availability.available === true &&
    blocking.length === 0;

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

  return (
    <Panel title="New experiment">
      {catalog.error && <ErrorBox>{catalog.error.message}</ErrorBox>}

      <div className="flex flex-col gap-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name">
            <input
              className={inputClasses}
              aria-label="Name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="efficientad on candle"
            />
          </Field>

          <Field label="Dataset">
            <select
              className={inputClasses}
              aria-label="Dataset"
              value={datasetId ?? ""}
              onChange={(event) => {
                const next = event.target.value === "" ? undefined : Number(event.target.value);
                setDatasetId(next);
                // A split belongs to one dataset, so the old choice is now meaningless
                // rather than merely stale.
                setSplitId(undefined);
              }}
            >
              <option value="">Choose a dataset…</option>
              {datasets.data?.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Split">
            <select
              className={inputClasses}
              aria-label="Split"
              value={splitId ?? ""}
              disabled={datasetId === undefined}
              onChange={(event) =>
                setSplitId(event.target.value === "" ? undefined : Number(event.target.value))
              }
            >
              <option value="">Choose a split…</option>
              {splits.data?.map((split) => (
                <option key={split.id} value={split.id}>
                  {split.name} · {split.strategy}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Method">
            <select
              className={inputClasses}
              aria-label="Method"
              value={methodKey ?? ""}
              onChange={(event) => setMethodKey(event.target.value)}
            >
              {catalog.data?.methods.map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.title}
                  {entry.availability.available ? "" : " (unavailable)"}
                </option>
              ))}
            </select>
          </Field>
        </div>

        {datasetId !== undefined && splits.data?.length === 0 && (
          <ErrorBox>
            This dataset has no splits yet.{" "}
            <Link className="underline" to={`/datasets/${datasetId}/splits`}>
              Create one first.
            </Link>
          </ErrorBox>
        )}

        {method && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-slate-600 dark:text-slate-300">{method.summary}</p>
            <div className="flex flex-wrap gap-1.5">
              {method.capabilities.dataset_specific && (
                <Badge tone="warning">assumes this dataset&rsquo;s geometry</Badge>
              )}
              {method.capabilities.produces_anomaly_map && <Badge tone="info">anomaly maps</Badge>}
              {method.capabilities.produces_diagnostics && <Badge tone="info">diagnostics</Badge>}
              <Badge tone="neutral">prefers {method.capabilities.preferred_device}</Badge>
            </div>
            {/* Listed rather than hidden: "why can't I pick EfficientAD" has to be
                answerable from the screen. */}
            {!method.availability.available && <ErrorBox>{method.availability.reason}</ErrorBox>}
          </div>
        )}

        {method && (
          <details open className="rounded border border-slate-200 px-3 py-2 dark:border-slate-700">
            <summary className="cursor-pointer text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
              {method.title} configuration
            </summary>
            <div className="mt-3">
              <SchemaForm
                fields={configFields}
                values={configValues}
                onChange={setConfigValues}
              />
            </div>
          </details>
        )}

        <details className="rounded border border-slate-200 px-3 py-2 dark:border-slate-700">
          <summary className="cursor-pointer text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            Preprocessing — every method sees exactly this
          </summary>
          <div className="mt-3">
            <SchemaForm
              fields={preprocessingFields}
              values={preprocessingValues}
              onChange={setPreprocessingValues}
            />
          </div>
        </details>

        <details className="rounded border border-slate-200 px-3 py-2 dark:border-slate-700">
          <summary className="cursor-pointer text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            Evaluation — how the stored scores are read back
          </summary>
          <div className="mt-3">
            <SchemaForm
              fields={evaluationFields}
              values={evaluationValues}
              onChange={setEvaluationValues}
            />
          </div>
        </details>

        {blocking.length > 0 && <ErrorBox>Check: {blocking.join(", ")}.</ErrorBox>}
        {create.error && <ErrorBox>{create.error.message}</ErrorBox>}

        <div>
          <Button variant="primary" disabled={!ready || create.isPending} onClick={submit}>
            {create.isPending ? "Creating…" : "Create experiment"}
          </Button>
        </div>
      </div>
    </Panel>
  );
}
