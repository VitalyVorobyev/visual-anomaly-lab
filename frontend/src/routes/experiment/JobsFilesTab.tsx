/**
 * What has run, what each run said, and what it left on disk.
 *
 * All of this used to be the top of the Overview tab, where a run list and a log tail took
 * the whole first screen and the experiment's actual findings started below the fold. The
 * job log is diagnostic material — you go looking for it when something is wrong — and
 * putting it in front of the metrics answered a question nobody had asked yet.
 *
 * So it is a tab of its own, and Overview is about what the experiment achieved. Nothing
 * here is new; it is the same `Runs` list, the same `JobProgress`, and the same
 * `ArtifactsPanel`, with the log now selectable per job rather than only for whichever run
 * happened to be followed.
 */

import { useState } from "react";

import type { JobSummary } from "../../api/client";
import { JobProgress } from "../../components/JobProgress";
import { Badge, Empty, Panel } from "@vitavision/lab-ui";
import { useJob } from "../../hooks/useJob";
import { ArtifactsPanel } from "./ArtifactsPanel";
import { jobTone } from "./OverviewTab";

export function JobsFilesTab({
  experimentId,
  jobs,
  followingJobId,
  onFollow,
}: {
  experimentId: number;
  jobs: JobSummary[];
  followingJobId: number | undefined;
  onFollow: (jobId: number) => void;
}) {
  // Whichever run the page is already following, or the most recent one, so opening the
  // tab shows a log rather than an instruction to pick one.
  const [selected, setSelected] = useState<number | undefined>();
  const shown = selected ?? followingJobId ?? jobs[0]?.id;

  return (
    <div className="flex flex-col gap-6">
      <Panel title="Runs">
        {jobs.length === 0 ? (
          <Empty>Nothing has run yet. Train first, then score.</Empty>
        ) : (
          <ul className="divide-y divide-line text-sm">
            {jobs.map((job) => (
              <li key={job.id} className="flex items-center gap-3 py-2">
                <span className="font-mono text-xs text-fg-muted">#{job.id}</span>
                <span className="w-16">{job.kind}</span>
                <Badge tone={jobTone(job.status)}>{job.status}</Badge>
                <span className="truncate text-xs text-fg-muted">
                  {job.message ?? job.error ?? ""}
                </span>
                {job.id !== shown && (
                  <button
                    type="button"
                    className="ml-auto shrink-0 text-xs text-fg-muted hover:text-fg hover:underline"
                    onClick={() => {
                      setSelected(job.id);
                      onFollow(job.id);
                    }}
                  >
                    show log
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {shown !== undefined && <JobConsole jobId={shown} experimentId={experimentId} />}

      <ArtifactsPanel experimentId={experimentId} />
    </div>
  );
}

function JobConsole({ jobId, experimentId }: { jobId: number; experimentId: number }) {
  // The experiment id is what makes the screen refresh itself when this job ends: whatever
  // is following the socket sees the terminal frame, and that is the only moment anything
  // here knows the run is over.
  const { job, lines, error } = useJob(jobId, { experimentId });
  return (
    <Panel title={`Job #${jobId}`}>
      <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
    </Panel>
  );
}
