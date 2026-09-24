/** The region list, with the mask's appearance beside the regions it governs. */

import { CircleDot, Eye, EyeOff } from "lucide-react";

import type { AnnotationLabel, AnnotationShape } from "../../api/client";
import { Badge, Empty, Slider, Tooltip, cn, focusRing } from "@vitavision/lab-ui";
import { useUpdateAnnotationLabel } from "../../hooks/useAnnotations";

export function RegionsSection({
  datasetId,
  shapes,
  labels,
  selectedId,
  onSelect,
  regionsHidden,
  onRegionsHidden,
  maskOpacity,
  onMaskOpacity,
}: {
  datasetId: number;
  shapes: AnnotationShape[];
  labels: AnnotationLabel[];
  selectedId: string | null;
  onSelect: (shapeId: string) => void;
  regionsHidden: boolean;
  onRegionsHidden: (hidden: boolean) => void;
  maskOpacity: number;
  onMaskOpacity: (value: number) => void;
}) {
  const recolour = useUpdateAnnotationLabel(datasetId);
  return (
    <section className="min-h-0 flex-1 overflow-y-auto p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-fg-muted">Regions</h2>
        <Badge tone="neutral">{shapes.length}</Badge>
      </div>

      {/* Appearance, beside the regions it governs. The right weight depends on the imagery:
          heavy enough to see over a bright specular surface is heavy enough to hide the
          texture of a dark field.

          The eye is a separate question from the weight, and it lives here rather than on the
          canvas so that "why is nothing drawn" has its answer next to the region count that is
          still saying there are three of them. */}
      <div className="mb-3 flex flex-col gap-2 rounded-control bg-raised/60 p-2">
        <div className="flex items-center gap-2">
          <Tooltip content={regionsHidden ? "Show the mask (H)" : "Hide the mask (H)"}>
            <button
              type="button"
              aria-label={regionsHidden ? "Show the mask" : "Hide the mask"}
              aria-pressed={regionsHidden}
              className={cn(
                "shrink-0 rounded-control p-1 text-fg-muted hover:bg-raised hover:text-fg",
                focusRing,
                regionsHidden && "bg-raised text-signal",
              )}
              onClick={() => onRegionsHidden(!regionsHidden)}
            >
              {regionsHidden ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
            </button>
          </Tooltip>
          <span className="shrink-0 text-[11px] text-fg-muted">Mask</span>
          <Slider
            aria-label="Mask opacity"
            min={0.1}
            max={1}
            step={0.05}
            value={maskOpacity}
            disabled={regionsHidden}
            onValueChange={onMaskOpacity}
            readout={regionsHidden ? "hidden" : `${Math.round(maskOpacity * 100)}%`}
          />
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {labels.map((label) => (
            <Tooltip key={label.key} content={`Colour of ${label.name}`}>
              <label
                className={cn(
                  "flex cursor-pointer items-center gap-1.5 rounded-control px-1.5 py-1 text-[11px] text-fg-muted hover:bg-raised",
                  focusRing,
                )}
              >
                <span
                  className="size-3 shrink-0 rounded-full border border-line-strong"
                  style={{ backgroundColor: label.color }}
                />
                <span className="max-w-24 truncate">{label.name}</span>
                <input
                  type="color"
                  value={label.color}
                  aria-label={`Colour of ${label.name}`}
                  disabled={recolour.isPending}
                  // Committed on `change`, not on `input`: a colour picker streams every value
                  // the pointer passes over, and each one would be a PUT.
                  onChange={(event) => recolour.mutate({ ...label, color: event.target.value })}
                  className="size-0 opacity-0"
                />
              </label>
            </Tooltip>
          ))}
        </div>
      </div>
      {shapes.length === 0 ? (
        <Empty>No editable regions. Choose Polygon or press P.</Empty>
      ) : (
        <div className="flex flex-col gap-1">
          {shapes.map((shape, index) => {
            const label = labels.find((item) => item.key === shape.label_key);
            return (
              <button
                key={shape.id}
                type="button"
                onClick={() => onSelect(shape.id)}
                className={cn(
                  "flex items-center gap-2 rounded-control px-2 py-1.5 text-left text-xs",
                  focusRing,
                  selectedId === shape.id ? "bg-raised text-fg" : "text-fg-muted hover:bg-raised/60",
                )}
              >
                <CircleDot className="size-3.5" style={{ color: label?.color }} />
                <span className="min-w-0 flex-1 truncate">
                  {index + 1}. {label?.name ?? shape.label_key}
                </span>
                <span className="font-mono text-[9px] text-fg-subtle">
                  {shape.operation === "subtract" ? "−" : "+"}
                  {shape.kind === "polygon" ? `${shape.points.length}v` : "mask"}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}
