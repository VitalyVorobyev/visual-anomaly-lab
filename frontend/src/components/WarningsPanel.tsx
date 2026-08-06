/**
 * The import review's warnings panel.
 *
 * ADR-0006 accepts a real risk with the review step: for a regular dataset it is pure
 * ceremony, and it will be tempting to click through without reading — at which point the
 * step provides false assurance rather than safety. So the panel does two things: it says
 * plainly when there is nothing to report, and when there *is*, it requires an explicit
 * acknowledgement rather than merely displaying a notice.
 *
 * Warnings are never fatal. A sample with a different channel count from its siblings is
 * legitimate data (ADR-0005); the operator is being asked to look, not to fix.
 */

import type { ManifestWarning } from "../api/client";
import { Badge, Panel } from "./ui";
import type { Tone } from "./ui";

const WARNING_TONE: Record<string, Tone> = {
  // Not a defect in the data — the reference dataset genuinely has a capture group with
  // one channel fewer — so it is informational rather than a caution.
  variable_channel_count: "info",
  unassigned_channel: "warning",
  unknown_channel_name: "warning",
  duplicate_hash: "warning",
  unreadable_file: "warning",
  empty_file: "warning",
};

/** Whether the commit button should be disabled. */
export function commitBlocked(warnings: ManifestWarning[], acknowledged: boolean): boolean {
  return warnings.length > 0 && !acknowledged;
}

export function WarningsPanel({
  warnings,
  acknowledged,
  onAcknowledge,
}: {
  warnings: ManifestWarning[];
  acknowledged: boolean;
  onAcknowledge: (value: boolean) => void;
}) {
  if (warnings.length === 0) {
    return (
      <Panel title="Warnings">
        <p className="text-sm text-slate-500 dark:text-slate-400">
          None. Every sample has the same number of images, every file was readable, and no
          two files have identical content.
        </p>
      </Panel>
    );
  }

  return (
    <Panel title={`Warnings (${warnings.length})`}>
      <ul className="flex flex-col gap-3">
        {warnings.map((warning) => (
          <li key={warning.code} className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <Badge tone={WARNING_TONE[warning.code] ?? "warning"}>{warning.code}</Badge>
            </div>
            <p className="text-sm">{warning.message}</p>
            {warning.paths.length > 0 && (
              <ul className="max-h-32 overflow-y-auto rounded bg-slate-50 p-2 font-mono text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                {warning.paths.map((path: string) => (
                  <li key={path} className="truncate">
                    {path}
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>

      <label className="mt-4 flex items-center gap-2 text-sm">
        <input
          type="checkbox"
          checked={acknowledged}
          onChange={(event) => onAcknowledge(event.target.checked)}
        />
        I have read these and want to import anyway
      </label>
    </Panel>
  );
}
