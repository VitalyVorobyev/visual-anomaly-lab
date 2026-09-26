/**
 * Splits: start from a preset, tune one by hand if needed, and see exactly what each contains.
 *
 * Splits are immutable — changing one means creating another — so what a split will contain
 * is shown before it exists. The top of the screen is a card per preset this dataset can
 * serve, each with the composition its dry run produced and one press to create it; the
 * strategy form is folded away under "Custom split" and previews its params as they change.
 * Below, every split reports its task, its composition and the experiments that ran on it,
 * and one no experiment ran on can be deleted.
 */

import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import type { SplitDetail, SplitParamsInput, SplitPreset, Task } from "../api/client";
import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  Disclosure,
  Empty,
  ErrorBox,
  Field,
  Input,
  NumberInput,
  Panel,
  Select,
  Skeleton,
  Slider,
} from "@vitavision/lab-ui";
import {
  compositionMode,
  SplitComposition,
  type ClassInfo,
} from "../components/SplitComposition";
import { useClassCoverage } from "../hooks/useAnnotations";
import { useDataset, useSplits } from "../hooks/useCatalog";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import {
  DEFAULT_SPLIT_PARAMS,
  useCreateSplit,
  useDeleteSplit,
  useSplitDeletionPreview,
  useSplitPresets,
  useSplitPreview,
  type SplitRequest,
} from "../hooks/useSplitPresets";
import { hasClasses, hasLabels } from "../api/truth";
import { TabScroll } from "./dataset/TabScroll";
import { EMPTY_BROWSE, writeBrowseState } from "../api/browseState";

type Strategy = "normal_only_train" | "imported" | "few_shot" | "class_stratified";

const STRATEGIES: Strategy[] = ["normal_only_train", "imported", "few_shot", "class_stratified"];

const STRATEGY_OPTION: Record<Strategy, { label: string; task: string }> = {
  normal_only_train: { label: "Draw one, normals only in train", task: "Anomaly" },
  imported: { label: "Adopt the published partition", task: "Anomaly" },
  few_shot: { label: "Draw references of a class", task: "Few-shot" },
  class_stratified: { label: "Draw annotated samples by class", task: "Segment · detect" },
};

const TASK_BADGE: Record<Task, string> = {
  anomaly: "Anomaly",
  few_shot_segmentation: "Few-shot",
  semantic_segmentation: "Segmentation",
  object_detection: "Detection",
};

function TaskBadges({ tasks }: { tasks: readonly Task[] }) {
  return (
    <span className="flex flex-wrap gap-1">
      {tasks.map((task) => (
        <Badge key={task}>{TASK_BADGE[task]}</Badge>
      ))}
    </span>
  );
}

export function SplitsRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const [search] = useSearchParams();

  const dataset = useDataset(datasetId);
  const splits = useSplits(datasetId);
  const presets = useSplitPresets(datasetId);
  const classes: ClassInfo[] = useMemo(
    () =>
      (dataset.data?.class_counts ?? []).map(({ key, name, color }) => ({ key, name, color })),
    [dataset.data],
  );

  // A strategy can be offered when the dataset's truth can feed it (ADR-0041): verdicts for
  // the anomaly draws, a manifest that partitions for the published one, classes for the rest.
  const truth = dataset.data?.truth;
  const published = (presets.data ?? []).some((preset) => preset.key === "published");
  const available = STRATEGIES.filter((strategy) =>
    strategy === "normal_only_train"
      ? hasLabels(truth)
      : strategy === "imported"
        ? published
        : hasClasses(truth),
  );
  const requested = STRATEGIES.find((entry) => entry === search.get("strategy"));

  return (
    <TabScroll className="flex flex-col gap-6">
      <section aria-labelledby="presets-heading" className="flex flex-col gap-3">
        <div>
          <h2 id="presets-heading" className="text-sm font-medium text-fg">
            New split
          </h2>
          <p className="text-xs text-fg-muted">
            Presets this dataset can serve, with what each would contain. One press creates it;
            pressing again draws the next seed.
          </p>
        </div>
        {presets.error && <ErrorBox>{presets.error.message}</ErrorBox>}
        {presets.isPending && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Skeleton className="h-44" />
            <Skeleton className="h-44" />
          </div>
        )}
        {presets.data?.length === 0 && (
          <Callout title="No preset fits this dataset yet">
            A split needs truth to divide by.{" "}
            <Link
              className="text-signal underline underline-offset-2"
              to={`/datasets/${datasetId}`}
            >
              Label samples normal or defect
            </Link>{" "}
            for anomaly detection, or{" "}
            <Link
              className="text-signal underline underline-offset-2"
              to={`/datasets/${datasetId}/annotate`}
            >
              annotate classes
            </Link>{" "}
            for few-shot, segmentation and detection.
          </Callout>
        )}
        {presets.data && presets.data.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {presets.data.map((preset) => (
              <PresetCard
                key={preset.key}
                preset={preset}
                datasetId={datasetId}
                classes={classes}
              />
            ))}
          </div>
        )}
        {dataset.data && available.length > 0 && (
          <Disclosure summary="Custom split" defaultOpen={requested !== undefined}>
            <CustomSplit
              datasetId={datasetId}
              available={available}
              requested={requested}
              classes={classes}
            />
          </Disclosure>
        )}
      </section>

      <section aria-labelledby="splits-heading" className="flex flex-col gap-3">
        <h2 id="splits-heading" className="text-sm font-medium text-fg">
          Splits
        </h2>
        {splits.error && <ErrorBox>{splits.error.message}</ErrorBox>}
        {splits.isPending && <Skeleton className="h-32" />}
        {splits.data?.length === 0 && (
          <Empty>
            No splits yet.{" "}
            {(presets.data ?? []).length > 0
              ? "Create one from a preset above — no settings needed."
              : "Once the dataset has truth to divide by, presets appear above."}
          </Empty>
        )}
        {(splits.data ?? []).map((split) => (
          <SplitCard key={split.id} split={split} datasetId={datasetId} classes={classes} />
        ))}
      </section>
    </TabScroll>
  );
}

function PresetCard({
  preset,
  datasetId,
  classes,
}: {
  preset: SplitPreset;
  datasetId: number;
  classes: readonly ClassInfo[];
}) {
  const create = useCreateSplit(datasetId);
  const defaultClass = preset.params.label_key ?? "";
  const [classKey, setClassKey] = useState(defaultClass);
  // Another class is another dry run; the preset's own answers for its default class.
  const params: SplitParamsInput =
    classKey && classKey !== defaultClass
      ? { ...preset.params, label_key: classKey }
      : preset.params;
  const other = useSplitPreview(
    datasetId,
    classKey && classKey !== defaultClass ? { params } : undefined,
  );
  const shown =
    classKey && classKey !== defaultClass
      ? other.data
      : { composition: preset.composition, name: preset.name, error: null };
  const mode = compositionMode(preset.params.strategy, params, preset.composition);

  return (
    <article
      aria-label={preset.label}
      className="flex flex-col gap-3 rounded-panel border border-line bg-surface p-4"
    >
      <header className="flex flex-col gap-1.5">
        <div className="flex items-start justify-between gap-2">
          <h3 className="text-sm font-medium text-fg">{preset.label}</h3>
          <TaskBadges tasks={preset.tasks} />
        </div>
        <p className="text-xs text-fg-muted">{preset.meaning}</p>
      </header>
      {preset.classes.length > 0 && (
        <Field as="group" label="Class">
          <Select
            aria-label={`${preset.label} class`}
            value={classKey}
            onValueChange={setClassKey}
            options={preset.classes.map((entry) => ({
              value: entry.key,
              label: entry.name,
              note: `${entry.samples} samples`,
            }))}
          />
        </Field>
      )}
      {shown?.error ? (
        <p className="text-xs text-warn">{shown.error}</p>
      ) : shown ? (
        <SplitComposition
          composition={shown.composition}
          mode={mode}
          classes={classes}
          normalsOnlyTrain={preset.tasks.includes("anomaly")}
        />
      ) : (
        <Skeleton className="h-16" />
      )}
      {create.error && <ErrorBox>{create.error.message}</ErrorBox>}
      <footer className="mt-auto flex items-end justify-between gap-3">
        <span className="min-w-0 break-words font-mono text-xs text-fg-subtle">
          {shown?.name}
        </span>
        <Button
          variant="primary"
          size="sm"
          loading={create.isPending}
          disabled={!shown || Boolean(shown.error)}
          onClick={() => create.mutate({ params })}
        >
          Create
        </Button>
      </footer>
    </article>
  );
}

function CustomSplit({
  datasetId,
  available,
  requested,
  classes,
}: {
  datasetId: number;
  available: Strategy[];
  requested: Strategy | undefined;
  classes: readonly ClassInfo[];
}) {
  const create = useCreateSplit(datasetId);
  const [name, setName] = useState("");
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [strategy, setStrategy] = useState<Strategy>(
    requested && available.includes(requested) ? requested : (available[0] ?? "normal_only_train"),
  );
  // A strategy that stopped being available — the truth changed — gives way to one that is.
  useEffect(() => {
    const [first] = available;
    if (first !== undefined && !available.includes(strategy)) setStrategy(first);
  }, [available, strategy]);
  const [trainFraction, setTrainFraction] = useState(0.6);
  const [valFraction, setValFraction] = useState(0.2);
  const [valDefectFraction, setValDefectFraction] = useState(0.3);
  const [holdout, setHoldout] = useState(0);
  const [labelKey, setLabelKey] = useState("");
  const [shots, setShots] = useState(5);
  const [annotatedTrain, setAnnotatedTrain] = useState(0.7);

  const drawn = strategy === "normal_only_train";
  const references = strategy === "few_shot";
  const byClass = strategy === "class_stratified";
  // A class can supply references once some sample shows it (ADR-0040).
  const coverage = useClassCoverage(datasetId);
  const drawable = (coverage.data ?? []).filter((entry) => entry.present > 0);
  const chosenClass = drawable.find((entry) => entry.label_key === labelKey);
  // The class most samples show, so the preview has something to draw from the start.
  const frequent = drawable.reduce<(typeof drawable)[number] | undefined>(
    (best, entry) => (best === undefined || entry.present > best.present ? entry : best),
    undefined,
  );
  useEffect(() => {
    if (labelKey === "" && frequent !== undefined) setLabelKey(frequent.label_key);
  }, [labelKey, frequent]);

  const request: SplitRequest | undefined = useMemo(() => {
    if (references && !labelKey) return undefined;
    const params: SplitParamsInput = {
      ...DEFAULT_SPLIT_PARAMS,
      strategy,
      train_normal_fraction: trainFraction,
      val_normal_fraction: valFraction,
      val_defect_fraction: valDefectFraction,
      // Only meaningful for `imported`, and zero everywhere else so a drawn split's stored
      // params do not imply a holdout was considered.
      holdout_from_train: strategy === "imported" ? holdout : 0,
      train_fraction: annotatedTrain,
      ...(references ? { label_key: labelKey, shots } : {}),
    };
    return { params, ...(seed === undefined ? {} : { seed }) };
  }, [
    strategy,
    references,
    labelKey,
    shots,
    trainFraction,
    valFraction,
    valDefectFraction,
    holdout,
    annotatedTrain,
    seed,
  ]);
  const settled = useDebouncedValue(request);
  const preview = useSplitPreview(datasetId, settled);

  const seedField = (label: string) => (
    <Field label={label} description="Empty draws the first seed these settings have not used.">
      <NumberInput
        value={seed ?? ""}
        placeholder={preview.data ? String(preview.data.seed) : "auto"}
        onChange={(event) =>
          setSeed(event.target.value === "" ? undefined : Number(event.target.value))
        }
      />
    </Field>
  );

  return (
    <form
      className="mt-3 flex flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault();
        if (request === undefined) return;
        create.mutate(
          { ...request, name },
          {
            onSuccess: () => setName(""),
          },
        );
      }}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <Field as="group" label="Strategy">
          <Select
            aria-label="Strategy"
            value={strategy}
            onValueChange={(value) => setStrategy(value as Strategy)}
            options={available.map((value) => ({
              value,
              label: STRATEGY_OPTION[value].label,
              note: STRATEGY_OPTION[value].task,
            }))}
          />
        </Field>
        <Field label="Name" description="Optional.">
          <Input
            value={name}
            placeholder={preview.data?.name ?? "Derived from the settings"}
            onChange={(event) => setName(event.target.value)}
          />
        </Field>
        {references && (
          <>
            <Field
              as="group"
              label="Class"
              description={
                coverage.data !== undefined && drawable.length === 0
                  ? "No sample shows a class yet. Annotate references first."
                  : chosenClass
                    ? `${chosenClass.present} samples show it · ${chosenClass.absent} confirmed without it · ${chosenClass.unlabeled} unanswered`
                    : undefined
              }
            >
              <Select
                aria-label="Class"
                value={labelKey}
                placeholder="Choose a class"
                options={drawable.map((entry) => ({
                  value: entry.label_key,
                  label: entry.label_key,
                  note: `${entry.present} present`,
                }))}
                onValueChange={setLabelKey}
              />
            </Field>
            <Field label="References (shots)">
              <NumberInput
                min={1}
                value={shots}
                onChange={(event) => setShots(Number(event.target.value))}
              />
            </Field>
            {seedField("Seed")}
          </>
        )}
        {strategy === "imported" && (
          <>
            <Fraction
              label="Hold out this share of the published training normals"
              value={holdout}
              onChange={setHoldout}
            />
            {seedField("Seed (for the holdout only)")}
          </>
        )}
        {byClass && (
          <>
            {seedField("Seed")}
            <Fraction
              label="Annotated samples used for training"
              value={annotatedTrain}
              onChange={setAnnotatedTrain}
              min={0.05}
              max={0.95}
            />
          </>
        )}
        {drawn && (
          <>
            {seedField("Seed")}
            <Fraction
              label="Normals used for training"
              value={trainFraction}
              onChange={setTrainFraction}
            />
            <Fraction
              label="Normals held out for validation"
              value={valFraction}
              onChange={setValFraction}
            />
            <Fraction
              label="Defects in validation"
              value={valDefectFraction}
              onChange={setValDefectFraction}
            />
          </>
        )}
      </div>

      <p className="text-xs text-fg-muted">
        {references ? (
          <>
            Draws this many samples that show the class, under the seed, as the references a
            few-shot run learns from; every other sample is a query it is scored on. The same class
            with three seeds is three reference draws, which is how sensitivity to the choice of
            references is measured.{" "}
            {chosenClass && (
              <>
                To pick them by eye instead,{" "}
                <Link
                  className="text-signal underline underline-offset-2"
                  to={`/datasets/${datasetId}/studio/${chosenClass.label_key}`}
                >
                  open the reference studio
                </Link>
                .
              </>
            )}
          </>
        ) : byClass ? (
          <>
            For a segmentation or detection run, which learns from annotated samples of every
            class. Only samples whose annotation answers for every class of the dataset are drawn,
            and they are drawn by the set of classes each one shows, so every mix of classes trains
            and tests in proportion. Any class that two or more samples show trains; it is tested
            too, unless every sample showing it is the only one teaching another class — then its
            test metrics read as a dash. A class only one sample shows goes where the draw puts it.
            Samples without a full annotation go to test, where they are scored but measured
            against nothing; there is no validation subset.
          </>
        ) : drawn ? (
          <>
            Training is normals only, assignment is per sample so no two views of one part can
            straddle the boundary, and the draw is stratified by capture group. The seed and these
            fractions are stored with the split, because a seed alone reproduces nothing.
          </>
        ) : (
          <>
            Takes the partition the source dataset published, read from the manifest this dataset
            was imported from, so a number computed here is comparable to the one they published.
            Official one-class protocols usually have no validation subset at all, and an empty
            one is expected rather than a fault.{" "}
            {holdout > 0 ? (
              <>
                The holdout above moves {(holdout * 100).toFixed(0)}% of the published{" "}
                <em>training</em> normals into validation, for methods that calibrate on held-out
                normals. The published <em>test</em> subset is untouched.
              </>
            ) : (
              <>
                Leave the holdout at zero to reproduce the source exactly. Raise it if a method
                needs held-out normals to calibrate — it comes out of train, never out of test.
              </>
            )}
          </>
        )}
      </p>

      <div aria-live="polite" className="rounded-control border border-line bg-raised p-3">
        {request === undefined ? (
          <p className="text-xs text-fg-subtle">Choose a class to preview the draw.</p>
        ) : preview.data?.error ? (
          <p className="text-xs text-warn">{preview.data.error}</p>
        ) : preview.data ? (
          <SplitComposition
            composition={preview.data.composition}
            mode={compositionMode(strategy, { label_key: labelKey }, preview.data.composition)}
            classes={classes}
            normalsOnlyTrain={drawn || strategy === "imported"}
          />
        ) : preview.error ? (
          <p className="text-xs text-defect">{preview.error.message}</p>
        ) : (
          <Skeleton className="h-16" />
        )}
      </div>

      {create.error && <ErrorBox>{create.error.message}</ErrorBox>}

      <div>
        <Button
          type="submit"
          variant="primary"
          loading={create.isPending}
          disabled={request === undefined || Boolean(preview.data?.error)}
        >
          Create
        </Button>
      </div>
    </form>
  );
}

function Fraction({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
}) {
  return (
    <Field as="group" label={label} annotation={value.toFixed(2)}>
      <Slider
        min={min}
        max={max}
        step={0.05}
        value={value}
        aria-label={label}
        onValueChange={onChange}
      />
    </Field>
  );
}

/** What reproduces a split, which is not always its seed. */
function provenance(split: SplitDetail): string {
  // An imported split has no seed that means anything — what reproduces it is the manifest
  // that asserted the partition.
  if (split.strategy === "imported") return `imported · ${split.params.manifest_id ?? "no manifest"}`;
  if (split.strategy === "few_shot") {
    return `seed ${split.seed} · ${split.params.shots ?? "?"} × ${split.params.label_key ?? "?"}`;
  }
  if (split.strategy === "manual") return `${split.params.sample_ids.length} references · manual`;
  if (split.strategy === "class_stratified") {
    return `seed ${split.seed} · by ${split.params.classes.length} classes`;
  }
  return `seed ${split.seed} · ${split.strategy}`;
}

function SplitCard({
  split,
  datasetId,
  classes,
}: {
  split: SplitDetail;
  datasetId: number;
  classes: readonly ClassInfo[];
}) {
  const [confirming, setConfirming] = useState(false);
  const deletion = useSplitDeletionPreview(confirming ? split.id : undefined);
  const remove = useDeleteSplit(datasetId);
  const experiments = split.experiments ?? [];

  return (
    <Panel
      title={split.name}
      actions={
        <span className="flex items-center gap-3">
          <span className="font-mono text-xs text-fg-muted">{provenance(split)}</span>
          <Button variant="ghost" size="sm" onClick={() => setConfirming(true)}>
            Delete
          </Button>
        </span>
      }
      bodyClassName="flex flex-col gap-3"
    >
      <div className="flex flex-wrap items-center gap-2 text-xs text-fg-muted">
        <TaskBadges tasks={split.tasks ?? []} />
        <span>
          {experiments.length === 0 ? (
            "No experiment has run on it."
          ) : (
            <>
              Used by{" "}
              {experiments.map((entry, index) => (
                <span key={entry.experiment_id}>
                  {index > 0 && ", "}
                  <Link
                    className="text-fg underline-offset-2 hover:text-signal hover:underline"
                    to={`/experiments/${entry.experiment_id}`}
                  >
                    {entry.name}
                  </Link>
                </span>
              ))}
            </>
          )}
        </span>
      </div>
      <SplitComposition
        composition={split.composition}
        mode={compositionMode(split.strategy, split.params, split.composition)}
        classes={classes}
        normalsOnlyTrain={split.strategy === "normal_only_train" || split.strategy === "imported"}
        rowAction={(row) =>
          row.total > 0 ? (
            // Into the browser *filtered to this row*.
            <Link
              to={{
                pathname: `/datasets/${datasetId}`,
                search: writeBrowseState({
                  ...EMPTY_BROWSE,
                  splitId: split.id,
                  subset: row.subset,
                }).toString(),
              }}
              className="text-fg-muted transition-colors hover:text-signal"
            >
              browse
            </Link>
          ) : null
        }
      />
      <ConfirmDialog
        open={confirming}
        onOpenChange={(open) => !open && setConfirming(false)}
        title="Delete this split?"
        description={
          <>
            <span className="font-medium text-fg">{split.name}</span> and its sample assignments
            will be removed. Samples, labels and annotations are never touched.
            {deletion.isPending && (
              <span className="mt-3 block text-fg-subtle">Checking what uses it…</span>
            )}
            {deletion.error && (
              <span className="mt-3 block text-defect">{deletion.error.message}</span>
            )}
            {deletion.data && (
              <span className="mt-3 block rounded-control border border-line bg-raised px-3 py-2">
                <span className="block font-mono text-xs text-fg">
                  {deletion.data.assignments} sample assignments
                </span>
                {/* A refusal names the runs holding the split, because "delete them first" is
                    only actionable if you know which ones. */}
                {deletion.data.blocker && (
                  <span className="mt-1 block text-xs text-warn">{deletion.data.blocker}</span>
                )}
              </span>
            )}
            {remove.error && <span className="mt-3 block text-defect">{remove.error.message}</span>}
          </>
        }
        confirmLabel="Delete split"
        destructive
        loading={remove.isPending}
        disabled={!deletion.data?.can_delete}
        onConfirm={() => {
          if (!deletion.data?.can_delete) return;
          remove.mutate(split.id, { onSuccess: () => setConfirming(false) });
        }}
      />
    </Panel>
  );
}
