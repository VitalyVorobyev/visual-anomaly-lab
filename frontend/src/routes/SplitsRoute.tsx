/**
 * Splits: create one, and see exactly what it contains.
 *
 * Splits are immutable — changing one means creating another — so this screen only
 * creates and reports. The composition table is the honest part: a split that says
 * "train: 36 normal, 0 defect" has stated its own correctness, and the M2 exit criterion
 * is that this still reads the same after the application is closed and reopened.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import { api, unwrap } from "../api/client";
import type { SplitDetail } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import {
  Badge,
  Button,
  Empty,
  ErrorBox,
  Field,
  Input,
  NumberInput,
  Panel,
  Select,
  Slider,
  Table,
  type Column,
} from "@vitavision/lab-ui";
import { useClassCoverage } from "../hooks/useAnnotations";
import { useSplits } from "../hooks/useCatalog";
import { TabScroll } from "./dataset/TabScroll";
import { EMPTY_BROWSE, writeBrowseState } from "../api/browseState";

type Strategy = "normal_only_train" | "imported" | "few_shot" | "class_stratified";

const STRATEGIES: Strategy[] = ["normal_only_train", "imported", "few_shot", "class_stratified"];

/** A prerequisite link can name the strategy its task needs, so the form opens on it. */
function initialStrategy(requested: string | null): Strategy {
  return STRATEGIES.find((entry) => entry === requested) ?? "normal_only_train";
}

export function SplitsRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const queryClient = useQueryClient();
  const [search] = useSearchParams();

  const splits = useSplits(datasetId);

  const [name, setName] = useState("default");
  const [seed, setSeed] = useState(0);
  const [strategy, setStrategy] = useState<Strategy>(() =>
    initialStrategy(search.get("strategy")),
  );
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

  const create = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/splits", {
          body: {
            dataset_id: datasetId,
            name,
            seed,
            params: {
              strategy,
              train_normal_fraction: trainFraction,
              val_normal_fraction: valFraction,
              val_defect_fraction: valDefectFraction,
              // Only meaningful for `imported`, and zero everywhere else so a drawn
              // split's stored params do not imply a holdout was considered.
              holdout_from_train: strategy === "imported" ? holdout : 0,
              ...(references ? { label_key: labelKey, shots } : {}),
              // Required on the wire, and meaningless for every strategy but `class_stratified`.
              train_fraction: annotatedTrain,
              // Assigned rather than left out: unlabelled samples are excluded from
              // every metric later, but they have to be scored to appear in the
              // ranked lists.
              unlabeled_subset: "test",
            },
          },
        }),
        "the new split",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });

  return (
    <TabScroll className="flex flex-col gap-4">
      <Panel title="Create a split">
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            create.mutate();
          }}
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Name">
              <Input value={name} onChange={(event) => setName(event.target.value)} />
            </Field>
            <Field as="group" label="Strategy">
              <Select
                aria-label="Strategy"
                value={strategy}
                onValueChange={(value) => setStrategy(value as Strategy)}
                options={[
                  { value: "normal_only_train", label: "Draw one", note: "normal_only_train" },
                  { value: "imported", label: "Adopt the published one", note: "imported" },
                  { value: "few_shot", label: "Draw references for a class", note: "few_shot" },
                  {
                    value: "class_stratified",
                    label: "Draw annotated samples by class",
                    note: "class_stratified",
                  },
                ]}
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
                <Field label="Seed">
                  <NumberInput
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </Field>
              </>
            )}
            {strategy === "imported" && (
              <>
                <Fraction
                  label="Hold out this share of the published training normals"
                  value={holdout}
                  onChange={setHoldout}
                />
                <Field label="Seed (for the holdout only)">
                  <NumberInput
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </Field>
              </>
            )}
            {byClass && (
              <>
                <Field label="Seed">
                  <NumberInput
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </Field>
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
                <Field label="Seed">
                  <NumberInput
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </Field>
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
                few-shot run learns from; every other sample is a query it is scored on. The same
                class with three seeds is three reference draws, which is how sensitivity to the
                choice of references is measured.{" "}
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
                For a segmentation run, which learns from annotated samples of every class. Only
                samples whose annotation answers for every class of the dataset are drawn, and they
                are drawn by the set of classes each one shows, so every mix of classes trains and
                tests in proportion. Any class that two or more samples show trains; it is tested
                too, unless every sample showing it is the only one teaching another class — then
                its test metrics read as a dash. A class only one sample shows goes where the draw
                puts it. Samples without a full annotation go to test, where they are scored but
                measured against nothing; there is no validation subset.
              </>
            ) : drawn ? (
              <>
                Training is normals only, assignment is per sample so no two views of one
                part can straddle the boundary, and the draw is stratified by capture
                group. The seed and these fractions are stored with the split, because a
                seed alone reproduces nothing.
              </>
            ) : (
              <>
                Takes the partition the source dataset published, read from the manifest
                this dataset was imported from — the point is to reproduce someone else's
                split exactly, so that a number computed here is comparable to the one they
                published. Official one-class protocols usually have no validation subset
                at all, and an empty one is expected rather than a fault.{" "}
                {holdout > 0 ? (
                  <>
                    The holdout above moves {(holdout * 100).toFixed(0)}% of the published{" "}
                    <em>training</em> normals into validation, for methods that calibrate on
                    held-out normals. The published <em>test</em> subset is untouched, so the
                    reported figure is still the one the protocol defines.
                  </>
                ) : (
                  <>
                    Leave the holdout at zero to reproduce the source exactly. Raise it if a
                    method needs held-out normals to calibrate — it comes out of train, never
                    out of test.
                  </>
                )}
              </>
            )}
          </p>

          {create.error && <ErrorBox>{create.error.message}</ErrorBox>}

          <div>
            <Button
              type="submit"
              variant="primary"
              disabled={create.isPending || (references && chosenClass === undefined)}
            >
              {create.isPending ? "Creating…" : "Create"}
            </Button>
          </div>
        </form>
      </Panel>

      {splits.error && <ErrorBox>{splits.error.message}</ErrorBox>}
      {splits.data?.length === 0 && <Empty>No splits yet.</Empty>}
      {(splits.data ?? []).map((split) => (
        <SplitCard key={split.id} split={split} datasetId={datasetId} />
      ))}
    </TabScroll>
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

function SplitCard({ split, datasetId }: { split: SplitDetail; datasetId: number }) {
  return (
    <Panel
      title={split.name}
      actions={
        <span className="font-mono text-xs text-fg-muted">
          {/* An imported split has no seed that means anything — what reproduces it is
              the manifest that asserted the partition. */}
          {split.strategy === "imported"
            ? `${split.strategy} · ${split.params.manifest_id ?? "no manifest"}`
            : split.strategy === "few_shot"
              ? `seed ${split.seed} · ${split.params.shots ?? "?"} × ${split.params.label_key ?? "?"}`
              : split.strategy === "manual"
                ? `${split.params.sample_ids.length} references · manual`
                : split.strategy === "class_stratified"
                  ? `seed ${split.seed} · by ${split.params.classes.length} classes`
                  : `seed ${split.seed} · ${split.strategy}`}
        </span>
      }
    >
      <Table
        caption={`Composition of ${split.name}`}
        rows={split.composition}
        rowKey={(row) => row.subset}
        columns={compositionColumns(datasetId, split)}
      />
    </Panel>
  );
}

type CompositionRow = SplitDetail["composition"][number];

function compositionColumns(datasetId: number, split: SplitDetail): Column<CompositionRow>[] {
  const splitId = split.id;
  // Only a split whose train is normals promises no defect there; a split for segmentation
  // trains on defects on purpose.
  const normalsTrain = split.strategy === "normal_only_train" || split.strategy === "imported";
  return [
    {
      key: "subset",
      header: "Subset",
      cell: (row) => (
        <Badge tone={row.subset === "train" ? "info" : "neutral"}>{row.subset}</Badge>
      ),
    },
    { key: "total", header: "Total", numeric: true, cell: (row) => row.total },
    { key: "normal", header: "Normal", numeric: true, cell: (row) => row.normal },
    {
      key: "defect",
      header: "Defect",
      numeric: true,
      // A defect here would teach the model that defects are normal.
      cell: (row) =>
        normalsTrain && row.subset === "train" && row.defect === 0 ? (
          <span className="text-normal">0 ✓</span>
        ) : (
          row.defect
        ),
    },
    { key: "unlabeled", header: "Unlabeled", numeric: true, cell: (row) => row.unlabeled },
    {
      key: "browse",
      header: "",
      // Into the browser *filtered to this row*. It used to open the whole dataset, which
      // made "browse" beside a subset's counts a link to something other than that subset.
      cell: (row) => (
        <Link
          to={{
            pathname: `/datasets/${datasetId}`,
            search: writeBrowseState({ ...EMPTY_BROWSE, splitId, subset: row.subset }).toString(),
          }}
          className="text-xs text-fg-muted transition-colors hover:text-signal"
        >
          browse
        </Link>
      ),
    },
  ];
}
