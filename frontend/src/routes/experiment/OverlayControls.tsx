/**
 * What is drawn over the photograph, as one line of controls.
 *
 * This replaces a titled panel wrapping a single opacity slider. Opacity is a cosmetic
 * decision and did not deserve a continuous control taking a fifth of the window: the
 * useful question is *which layers*, not *how faint*, so each layer is a chip and the
 * alpha is fixed at a value that was chosen by looking at the result.
 *
 * The one continuous control left is the segmentation cut, and it is continuous because it
 * is not cosmetic — it changes what the model is shown to be claiming. It appears only when
 * the segmentation layer is on, so the common case is still three chips and nothing else.
 */

import type { MapScale } from "../../api/client";
import type { ResultsState } from "../../api/resultsState";
import { cutValue } from "../../api/resultsState";
import { Slider, ToggleChip } from "../../components/ui";

/**
 * The colours the *server* draws these layers in, repeated here for the swatches.
 *
 * Duplicated across the boundary on purpose and kept to two values: the alternative is an
 * endpoint that exists to tell the client what colour a PNG it already has was drawn in.
 * They must match `media/overlay.py` — `CONTOUR_RGB` and `PREDICTION_RGB`.
 */
export const TRUTH_SWATCH = "rgb(34 211 138)";
export const PREDICTION_SWATCH = "rgb(129 140 248)";
/** The heatmap has no single colour; its mid-scale orange stands for the ramp. */
export const HEATMAP_SWATCH = "rgb(220 82 52)";

export function OverlayControls({
  state,
  onChange,
  range,
  hasMask,
  hasMap,
}: {
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  range: MapScale | null | undefined;
  hasMask: boolean;
  hasMap: boolean;
}) {
  const cut = cutValue(state, range);

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <ToggleChip
          checked={state.heatmap}
          disabled={!hasMap}
          onCheckedChange={(heatmap) => onChange({ heatmap })}
          swatch={HEATMAP_SWATCH}
          title={hasMap ? "How anomalous, everywhere" : "This run recorded no anomaly map"}
        >
          heatmap
        </ToggleChip>
        <ToggleChip
          checked={state.region}
          disabled={!hasMap}
          onCheckedChange={(region) => onChange({ region })}
          swatch={PREDICTION_SWATCH}
          title={
            hasMap
              ? "Where the map crosses the cut — the model's own segmentation"
              : "This run recorded no anomaly map"
          }
        >
          prediction
        </ToggleChip>
        <ToggleChip
          checked={state.truth}
          disabled={!hasMask}
          onCheckedChange={(truth) => onChange({ truth })}
          swatch={TRUTH_SWATCH}
          title={hasMask ? "The annotated region" : "No ground-truth mask for this sample"}
        >
          ground truth
        </ToggleChip>
      </div>

      {/* Only while it can do something. A cut with no segmentation on screen is a control
          for an invisible effect. */}
      {state.region && (
        <div className="flex min-w-56 flex-1 items-center gap-2">
          <span className="shrink-0 text-xs text-fg-muted">cut</span>
          <Slider
            aria-label="Segmentation cut, as a fraction of the run's map range"
            min={0}
            max={1}
            step={0.01}
            value={state.cut}
            onValueChange={(next) => onChange({ cut: next })}
            readout={
              <span className="inline-block w-16 text-right font-mono text-xs">
                {cut === null ? "—" : cut.toFixed(3)}
              </span>
            }
          />
        </div>
      )}
    </div>
  );
}

/**
 * The numbers behind the picture, in words.
 *
 * Without this a map that is genuinely cold is indistinguishable on screen from one that
 * failed to render — which is exactly what score-driven alpha does to a low-scoring image,
 * and the reason a working overlay reads as a missing feature.
 */
export function MapScaleReadout({
  scale,
  range,
}: {
  scale: MapScale | null | undefined;
  range: MapScale | null | undefined;
}) {
  if (!scale && !range) return null;
  return (
    <p className="font-mono text-[11px] text-fg-subtle">
      {scale && (
        <>
          this map {scale.low.toFixed(3)} – {scale.high.toFixed(3)}
        </>
      )}
      {scale && range && " · "}
      {range && (
        <>
          run {range.low.toFixed(3)} – {range.high.toFixed(3)}
        </>
      )}
    </p>
  );
}
