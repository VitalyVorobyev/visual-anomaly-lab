import { Callout, PageHeader, Panel, SkeletonRows } from "@vitavision/lab-ui";
import { apiBaseUrl } from "../api/client";
import { useHealth } from "../hooks/useHealth";

/** One fact about the sidecar. A definition list, because that is what this is. */
function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1 border-b border-line py-2.5 last:border-0 sm:flex-row sm:items-baseline sm:gap-4">
      <dt className="w-40 shrink-0 text-xs font-medium text-fg-muted">{label}</dt>
      <dd className="min-w-0 font-mono text-sm break-all text-fg">{value}</dd>
    </div>
  );
}

export function HealthRoute() {
  const { data, error, isPending } = useHealth();

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Backend health"
        meta={
          <span>
            Live from the sidecar at <code className="font-mono text-fg">{apiBaseUrl}</code>.
          </span>
        }
      />

      {error && (
        <Callout tone="error" title="The backend is not reachable">
          Start it with <code className="font-mono text-fg">scripts/dev-backend.sh</code>, or check
          the sidecar log if you are running the desktop app.
        </Callout>
      )}

      <Panel>
        {isPending && <SkeletonRows rows={6} />}
        {data && (
          <dl className="flex flex-col">
            <Fact label="Status" value={data.status} />
            <Fact label="Version" value={data.version} />
            <Fact label="Schema version" value={String(data.schema_version)} />
            <Fact label="Database" value={data.db_path} />
            <Fact label="Data directory" value={data.data_dir} />
            <Fact label="Started at" value={new Date(data.started_at).toLocaleString()} />
          </dl>
        )}
      </Panel>
    </div>
  );
}
