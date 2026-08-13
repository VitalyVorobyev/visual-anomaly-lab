/**
 * The reference packs that are not in the catalogue yet.
 *
 * This was a full panel of inert boxes: every pack, registered or not, with its status
 * badge, its absolute path and `12/12 datasets registered` — a third of the window spent
 * restating what the list below already showed, in tiles that could not be clicked to
 * reach any of it.
 *
 * A registered pack is now a *collection* in the list, so it has nothing left to say here.
 * What remains is the one thing the list genuinely cannot show: a benchmark that is on
 * disk and not yet indexed, or one that is not on disk at all. Both are actionable, and
 * both are one line. When every pack is registered this renders nothing.
 */

import { ExternalLink, LibraryBig } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";

import { queryKeys } from "../../api/queryKeys";
import { JobProgress } from "../../components/JobProgress";
import { Button, ErrorBox, Skeleton } from "../../components/ui";
import { useReferencePacks, useRegisterReferencePacks } from "../../hooks/useCatalog";
import { isTerminal, useJob } from "../../hooks/useJob";

export function ReferencePackStrip() {
  const catalog = useReferencePacks();
  const register = useRegisterReferencePacks();
  const [jobId, setJobId] = useState<number>();
  const clearJob = useCallback(() => setJobId(undefined), []);

  const packs = catalog.data?.packs ?? [];
  const available = packs.filter((pack) => pack.status === "available");
  const missing = packs.filter((pack) => pack.status === "absent" || pack.status === "incomplete");

  if (catalog.isPending) return <Skeleton className="h-9 w-full" />;
  if (catalog.error) return <ErrorBox>{catalog.error.message}</ErrorBox>;
  if (available.length === 0 && missing.length === 0 && jobId === undefined) return null;

  return (
    <section className="flex flex-col gap-2 rounded-panel border border-line bg-surface px-3.5 py-3">
      <div className="flex items-center gap-2">
        <LibraryBig className="size-3.5 shrink-0 text-signal" aria-hidden />
        <h2 className="text-xs font-semibold tracking-tight text-fg">Reference benchmarks</h2>
      </div>

      {register.error && <ErrorBox>{register.error.message}</ErrorBox>}

      {available.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-fg-muted">
            {available.map((pack) => pack.title).join(", ")} found on disk —{" "}
            {catalog.data?.pending_datasets} dataset
            {catalog.data?.pending_datasets === 1 ? "" : "s"} ready to index. Registration reads
            the files in place and never changes them.
          </p>
          <Button
            variant="primary"
            size="sm"
            loading={register.isPending}
            disabled={jobId !== undefined}
            onClick={() =>
              register.mutate(
                { pack_keys: available.map((pack) => pack.key) },
                { onSuccess: (job) => setJobId(job.id) },
              )
            }
          >
            Register {catalog.data?.pending_datasets}
          </Button>
        </div>
      )}

      {missing.map((pack) => (
        <p key={pack.key} className="flex flex-wrap items-center gap-x-2 text-xs text-fg-muted">
          <span className="text-fg">{pack.title}</span>
          <span className="text-fg-subtle">
            {pack.status === "absent"
              ? "not found in the local datasets folder"
              : `incomplete — ${pack.missing.length} required path${
                  pack.missing.length === 1 ? "" : "s"
                } missing`}
          </span>
          <a
            href={pack.install_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-signal hover:underline"
          >
            Get it <ExternalLink className="size-3" />
          </a>
        </p>
      ))}

      {jobId !== undefined && (
        <div className="border-t border-line pt-3">
          <ReferenceRegistrationJob jobId={jobId} onFinished={clearJob} />
        </div>
      )}
    </section>
  );
}

function ReferenceRegistrationJob({
  jobId,
  onFinished,
}: {
  jobId: number;
  onFinished: () => void;
}) {
  const queryClient = useQueryClient();
  const { job, lines, error } = useJob(jobId);

  useEffect(() => {
    if (!job || !isTerminal(job.status)) return;
    void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
    void queryClient.invalidateQueries({ queryKey: queryKeys.referencePacks() });
    if (job.status === "succeeded") {
      const timer = window.setTimeout(onFinished, 1800);
      return () => window.clearTimeout(timer);
    }
  }, [job?.status, onFinished, queryClient]);

  return (
    <div className="flex flex-col gap-3">
      <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
      {job && isTerminal(job.status) && job.status !== "succeeded" && (
        <Button className="self-end" size="sm" variant="ghost" onClick={onFinished}>
          Dismiss
        </Button>
      )}
    </div>
  );
}
