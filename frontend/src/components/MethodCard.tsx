/**
 * One method, as a thing you press — on the create form and on the guided run's Method step.
 *
 * A native radio underneath, so a keyboard reaches the group with Tab and moves inside it
 * with the arrow keys, and the whole card is the hit target. The badges come from the
 * registry's `status` and `recommended_for` (`api/methodChoice.ts`); nothing here names a
 * method.
 */

import { Badge, cn } from "@vitavision/lab-ui";

import type { ModelDescription } from "../api/client";

export function MethodCard({
  method,
  recommended,
  selected,
  onSelect,
  name = "method",
}: {
  method: ModelDescription;
  /** The registry's default for the task being configured. */
  recommended: boolean;
  selected: boolean;
  onSelect: () => void;
  /** The radio group's name, so two pickers on one page do not become one group. */
  name?: string;
}) {
  const capabilities = method.capabilities;
  const unavailable = !method.availability.available;

  return (
    <label
      className={cn(
        "relative flex cursor-pointer flex-col gap-2 rounded-panel border p-3 transition-colors",
        "has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-signal",
        selected
          ? "border-signal bg-signal/5 ring-1 ring-signal"
          : "border-line bg-raised/40 hover:border-line-strong",
      )}
    >
      <input
        type="radio"
        name={name}
        value={method.key}
        checked={selected}
        onChange={onSelect}
        className="absolute inset-0 cursor-pointer opacity-0"
      />

      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-semibold tracking-tight text-fg">{method.title}</span>
        <span className="font-mono text-[11px] text-fg-subtle">{method.key}</span>
      </div>

      <p className="text-xs leading-snug text-fg-muted">{method.summary}</p>

      <div className="flex flex-wrap gap-1.5">
        {/* The registry's verdict, first on the card: a gate decided it, and it is what a
            reader choosing between methods needs before any capability. */}
        {recommended && <Badge tone="normal">recommended</Badge>}
        {method.status === "experimental" && <Badge tone="warning">experimental</Badge>}
        {method.status === "floor" && <Badge tone="neutral">floor</Badge>}
        {capabilities.dataset_specific && <Badge tone="warning">dataset-specific</Badge>}
        {/* A segmenter's map is a foreground probability, not an anomaly map. */}
        {capabilities.produces_anomaly_map && (
          <Badge tone="info">
            {capabilities.tasks.includes("anomaly") ? "anomaly maps" : "probability maps"}
          </Badge>
        )}
        {capabilities.produces_diagnostics && <Badge tone="info">diagnostics</Badge>}
        {capabilities.channel_aware && <Badge tone="info">channel-aware</Badge>}
        {capabilities.portable_formats.map((format) => (
          <Badge key={format} tone="neutral">
            {format.toUpperCase()} export
          </Badge>
        ))}
        {!capabilities.requires_training && <Badge tone="neutral">no training</Badge>}
        <Badge tone="neutral">
          <span className="font-mono">{capabilities.preferred_device}</span>
        </Badge>
      </div>

      {/* Stated on the card rather than in a banner elsewhere: this is where the reader
          asks the question, so this is where it has to be answered. Nothing went wrong —
          a dependency is simply not installed — so it is a caveat, not an error. */}
      {unavailable && method.availability.reason && (
        <p className="rounded-control border border-warn/30 bg-warn/8 px-2 py-1.5 text-xs leading-snug text-fg-muted">
          {method.availability.reason}
        </p>
      )}
    </label>
  );
}
