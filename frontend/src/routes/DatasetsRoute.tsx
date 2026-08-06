/** The catalog: every imported dataset, with its composition. */

import { Link } from "react-router";

import { Button, CountRun, Empty, ErrorBox, Panel } from "../components/ui";
import { useDatasets, useDeleteDataset } from "../hooks/useCatalog";

export function DatasetsRoute() {
  const datasets = useDatasets();
  const remove = useDeleteDataset();

  if (datasets.isPending) return <Empty>Loading…</Empty>;
  if (datasets.error) return <ErrorBox>{datasets.error.message}</ErrorBox>;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight">Datasets</h2>
        <Link to="/import">
          <Button variant="primary">Import…</Button>
        </Link>
      </div>

      {remove.error && <ErrorBox>{remove.error.message}</ErrorBox>}

      {datasets.data.length === 0 ? (
        <Empty>Nothing imported yet.</Empty>
      ) : (
        <ul className="flex flex-col gap-3">
          {datasets.data.map((dataset) => (
            <li key={dataset.id}>
              <Panel
                title={<Link to={`/datasets/${dataset.id}`}>{dataset.name}</Link>}
                actions={
                  <Button
                    variant="danger"
                    disabled={remove.isPending}
                    onClick={() => remove.mutate(dataset.id)}
                  >
                    Delete
                  </Button>
                }
              >
                <div className="flex flex-col gap-2">
                  <CountRun
                    counts={[
                      ["normal", dataset.label_counts["normal"] ?? 0, "normal"],
                      ["defect", dataset.label_counts["defect"] ?? 0, "defect"],
                      ["unlabeled", dataset.label_counts["unlabeled"] ?? 0, "unlabeled"],
                    ]}
                  />
                  <p className="text-sm text-slate-500 dark:text-slate-400">
                    {dataset.samples} samples · {dataset.images} images
                    {dataset.adapter ? ` · ${dataset.adapter}` : ""}
                  </p>
                  {/* Referenced in place, never copied (ADR-0001). */}
                  <p className="font-mono text-xs break-all text-slate-400">{dataset.root_path}</p>
                </div>
              </Panel>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
