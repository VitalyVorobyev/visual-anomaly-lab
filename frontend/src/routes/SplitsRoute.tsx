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
import { Link, useParams } from "react-router";

import { api, unwrap } from "../api/client";
import type { SplitDetail } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { Badge, Button, Empty, ErrorBox, Field, Panel, inputClasses } from "../components/ui";
import { useDataset, useSplits } from "../hooks/useCatalog";

export function SplitsRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const queryClient = useQueryClient();

  const dataset = useDataset(datasetId);
  const splits = useSplits(datasetId);

  const [name, setName] = useState("default");
  const [seed, setSeed] = useState(0);
  const [trainFraction, setTrainFraction] = useState(0.6);
  const [valFraction, setValFraction] = useState(0.2);
  const [valDefectFraction, setValDefectFraction] = useState(0.3);

  const create = useMutation({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/splits", {
          body: {
            dataset_id: datasetId,
            name,
            seed,
            params: {
              strategy: "normal_only_train",
              train_normal_fraction: trainFraction,
              val_normal_fraction: valFraction,
              val_defect_fraction: valDefectFraction,
              // Assigned rather than left out: unlabelled samples are excluded from every
              // metric later, but they have to be scored to appear in the ranked lists.
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
    <div className="flex flex-col gap-4">
      <div>
        <Link
          to={`/datasets/${datasetId}`}
          className="text-xs text-slate-500 hover:underline dark:text-slate-400"
        >
          ← back to the browser
        </Link>
        <h2 className="text-lg font-semibold tracking-tight">
          Splits{dataset.data ? ` — ${dataset.data.name}` : ""}
        </h2>
      </div>

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
              <input
                className={inputClasses}
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </Field>
            <Field label="Seed">
              <input
                className={`${inputClasses} font-mono`}
                type="number"
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
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400">
            Training is normals only, assignment is per sample so no two views of one part
            can straddle the boundary, and the draw is stratified by capture group. The
            seed and these fractions are stored with the split, because a seed alone
            reproduces nothing.
          </p>

          {create.error && <ErrorBox>{create.error.message}</ErrorBox>}

          <div>
            <Button type="submit" variant="primary" disabled={create.isPending}>
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
    </div>
  );
}

function Fraction({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={`${label} (${value.toFixed(2)})`}>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </Field>
  );
}

function SplitCard({ split, datasetId }: { split: SplitDetail; datasetId: number }) {
  return (
    <Panel
      title={split.name}
      actions={
        <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
          seed {split.seed} · {split.strategy}
        </span>
      }
    >
      <table className="w-full text-left text-sm">
        <thead className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
          <tr>
            <th className="pb-2">Subset</th>
            <th className="pb-2">Total</th>
            <th className="pb-2">Normal</th>
            <th className="pb-2">Defect</th>
            <th className="pb-2">Unlabeled</th>
            <th className="pb-2" />
          </tr>
        </thead>
        <tbody>
          {split.composition.map((row) => (
            <tr key={row.subset} className="border-t border-slate-100 dark:border-slate-800">
              <td className="py-1.5">
                <Badge tone={row.subset === "train" ? "info" : "neutral"}>{row.subset}</Badge>
              </td>
              <td className="py-1.5 font-mono text-xs">{row.total}</td>
              <td className="py-1.5 font-mono text-xs">{row.normal}</td>
              <td className="py-1.5 font-mono text-xs">
                {/* A defect here would teach the model that defects are normal. */}
                {row.subset === "train" && row.defect === 0 ? "0 ✓" : row.defect}
              </td>
              <td className="py-1.5 font-mono text-xs">{row.unlabeled}</td>
              <td className="py-1.5 text-right">
                <Link
                  to={`/datasets/${datasetId}`}
                  className="text-xs text-slate-500 hover:underline dark:text-slate-400"
                >
                  browse
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Panel>
  );
}
