/**
 * Where the run's output actually is.
 *
 * A training run can spend eleven minutes producing a 31 MB checkpoint and, until now, the
 * app never said where it went — the path was on `ExperimentDetail` the whole time and no
 * screen showed it. This is that path, what is under it, and a way to open it.
 *
 * A listing, never a download. ADR-0019 ruled out mounting the artifact directory over
 * HTTP and named exposing the checkpoints as one of its reasons; nothing here changes
 * that. Opening the folder is the desktop shell's job (ADR-0014), and the browser gets the
 * path as selectable text — a different affordance, not a disabled button.
 */

import { useState } from "react";
import type { ReactNode } from "react";
import { FolderOpen, Trash2 } from "lucide-react";

import type { ArtifactGroup } from "../../api/client";
import { onDemandNote } from "../../api/diagnostics";
import { hasRevealPath, revealPath } from "../../api/shell";
import {
  Button,
  ConfirmDialog,
  Disclosure,
  Empty,
  Panel,
  SkeletonRows,
  Tooltip,
} from "../../components/ui";
import {
  useArtifacts,
  useClearDiagnostics,
  useDiagnostics,
} from "../../hooks/useExperiments";

export function ArtifactsPanel({ experimentId }: { experimentId: number }) {
  const artifacts = useArtifacts(experimentId);
  const [error, setError] = useState<string | null>(null);
  const canReveal = hasRevealPath();

  if (artifacts.isPending) return <SkeletonRows rows={3} />;
  if (!artifacts.data) return null;
  const { root, exists, groups, total_bytes } = artifacts.data;

  const reveal = (path: string) => {
    setError(null);
    revealPath(path).catch((cause: unknown) =>
      setError(cause instanceof Error ? cause.message : String(cause)),
    );
  };

  return (
    <Panel
      title="Files"
      actions={
        canReveal && exists ? (
          <Button onClick={() => reveal(root)}>
            <FolderOpen className="size-4" />
            Reveal in Finder
          </Button>
        ) : undefined
      }
    >
      {!exists ? (
        <Empty>
          This run has written nothing yet. The directory appears when it first produces
          something.
        </Empty>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <span className="text-xs text-fg-muted">
              {formatBytes(total_bytes)} in
              {canReveal ? " this run's directory:" : " this run's directory — select to copy:"}
            </span>
            {/* Selectable rather than a button: in a browser this *is* the affordance. */}
            <code className="rounded border border-line bg-raised px-2 py-1 font-mono text-xs break-all select-all">
              {root}
            </code>
          </div>

          {error !== null && <p className="text-xs text-warn">{error}</p>}

          <ul className="flex flex-col divide-y divide-line">
            {groups.map((group) => (
              <GroupRow
                key={group.name}
                group={group}
                onReveal={canReveal ? () => reveal(group.path) : undefined}
                action={
                  group.name === "diagnostics" ? (
                    <ClearDiagnostics experimentId={experimentId} />
                  ) : undefined
                }
              />
            ))}
          </ul>
        </div>
      )}
    </Panel>
  );
}

function GroupRow({
  group,
  onReveal,
  action,
}: {
  group: ArtifactGroup;
  onReveal: (() => void) | undefined;
  action?: ReactNode;
}) {
  const summary = (
    <span className="flex flex-1 items-baseline gap-2">
      <span className="font-medium text-fg">{group.title}</span>
      <span className="font-mono text-[11px] text-fg-subtle">
        {group.file_count} file{group.file_count === 1 ? "" : "s"} ·{" "}
        {formatBytes(group.total_bytes)}
      </span>
    </span>
  );

  return (
    <li className="flex items-center gap-2 py-1.5">
      {/* A group whose files were listed opens; one that is only counted has nothing to
          open, so it is a plain row rather than a disclosure that reveals an empty box. */}
      {group.files.length > 0 ? (
        <Disclosure summary={summary} className="flex-1">
          <ul className="flex flex-col gap-0.5 font-mono text-[11px]">
            {group.files.map((file) => (
              <li key={file.name} className="flex items-baseline justify-between gap-3">
                <span className="truncate text-fg-muted">{file.name}</span>
                <span className="shrink-0 text-fg-subtle tabular-nums">
                  {formatBytes(file.bytes)}
                </span>
              </li>
            ))}
          </ul>
        </Disclosure>
      ) : (
        <span className="flex flex-1 items-baseline gap-2 py-1 text-xs">{summary}</span>
      )}
      {action}
      {onReveal && (
        <Button variant="ghost" onClick={onReveal} aria-label={`Reveal ${group.title}`}>
          <FolderOpen className="size-4" />
        </Button>
      )}
    </li>
  );
}

/**
 * Reclaim the disk per-image diagnostics cost (ADR-0027).
 *
 * The default scope keeps the model-scoped entries, and the confirmation says so: the
 * architecture tree and the score-normalization table are kilobytes, and losing them here
 * would read as the Architecture tab breaking rather than as a disk clear working.
 *
 * It reports what it actually freed rather than claiming success, because the number is
 * the reason somebody pressed it.
 */
function ClearDiagnostics({ experimentId }: { experimentId: number }) {
  const clear = useClearDiagnostics(experimentId);
  const diagnostics = useDiagnostics(experimentId);
  const [asking, setAsking] = useState(false);
  const [freed, setFreed] = useState<string | null>(null);

  // Counted separately from the run's own sample, because they are a different fact: the
  // run's count is explained by its budget, these are questions somebody asked (ADR-0027).
  const asked = onDemandNote(diagnostics.data);

  return (
    <>
      {asked !== null && <span className="text-[11px] text-fg-subtle">{asked}</span>}
      {freed !== null && <span className="text-[11px] text-fg-muted">{freed}</span>}
      {clear.error && <span className="max-w-xs text-[11px] text-warn">{clear.error.message}</span>}

      <Tooltip content="Anomaly maps are not touched: each one is referenced by a scored image, so deleting them would break the overlay.">
        <Button
          variant="ghost"
          onClick={() => setAsking(true)}
          disabled={clear.isPending}
          aria-label="Clear per-image diagnostics"
        >
          <Trash2 className="size-4" />
        </Button>
      </Tooltip>

      <ConfirmDialog
        open={asking}
        onOpenChange={setAsking}
        title="Clear per-image diagnostics?"
        confirmLabel="Clear"
        destructive
        loading={clear.isPending}
        description={
          "Every per-image pane goes, including any image you diagnosed on demand. " +
          "The architecture view and the other run-wide diagnostics stay, and the " +
          "anomaly maps are untouched. A future inference run records them again."
        }
        onConfirm={() => {
          setFreed(null);
          clear.mutate("image", {
            onSuccess: (result) => setFreed(`${formatBytes(result.bytes_reclaimed)} freed`),
          });
        }}
      />
    </>
  );
}

/** Binary units, because this is disk and the file manager beside it will agree. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}
