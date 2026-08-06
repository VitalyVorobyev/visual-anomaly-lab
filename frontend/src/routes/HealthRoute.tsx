import { apiBaseUrl } from "../api/client";
import { useHealth } from "../hooks/useHealth";

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-b border-slate-200 py-3 last:border-0 dark:border-slate-700">
      <dt className="text-xs font-medium tracking-wide text-slate-500 uppercase dark:text-slate-400">
        {label}
      </dt>
      <dd className="mt-1 font-mono text-sm break-all text-slate-900 dark:text-slate-100">
        {value}
      </dd>
    </div>
  );
}

export function HealthRoute() {
  const { data, error, isPending } = useHealth();

  return (
    <section className="space-y-4">
      <header>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Backend health</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Live from the sidecar at <code className="font-mono">{apiBaseUrl}</code>.
        </p>
      </header>

      {isPending && <p className="text-sm text-slate-500">Contacting the backend…</p>}

      {error && (
        <div className="rounded-md border border-red-300 bg-red-50 p-4 dark:border-red-800 dark:bg-red-950">
          <p className="text-sm font-medium text-red-800 dark:text-red-200">
            The backend is not reachable.
          </p>
          <p className="mt-1 text-sm text-red-700 dark:text-red-300">
            Start it with <code className="font-mono">scripts/dev-backend.sh</code>, or check the
            sidecar log if you are running the desktop app.
          </p>
        </div>
      )}

      {data && (
        <dl className="rounded-md border border-slate-200 px-4 dark:border-slate-700">
          <Field label="Status" value={data.status} />
          <Field label="Version" value={data.version} />
          <Field label="Schema version" value={String(data.schema_version)} />
          <Field label="Database" value={data.db_path} />
          <Field label="Data directory" value={data.data_dir} />
          <Field label="Started at" value={new Date(data.started_at).toLocaleString()} />
        </dl>
      )}
    </section>
  );
}
