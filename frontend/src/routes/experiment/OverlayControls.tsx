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
import { Slider, ToggleChip, cn } from "@vitavision/lab-ui";

import { classColour } from "../../components/viewer/labelPaint";

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
  classes,
  boxes,
}: {
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  range: MapScale | null | undefined;
  hasMask: boolean;
  hasMap: boolean;
  /**
   * A supervised segmentation run's pinned classes (ADR-0039). Given, the prediction and
   * the truth are its label maps — one colour per class, a legend instead of a swatch, and
   * no cut, because the method's classes are its own decision.
   */
  classes?: readonly string[] | undefined;
  /**
   * An object detection run's pinned classes and the sentence its cut was resolved by
   * (ADR-0039, ADR-0028). Given, the prediction and the truth are boxes, toned by verdict,
   * with the classes as a legend and the cut printed beside them.
   */
  boxes?: { classes: readonly string[]; rule: string | undefined } | undefined;
}) {
  if (boxes !== undefined) {
    return <BoxControls state={state} onChange={onChange} hasMap={hasMap} boxes={boxes} />;
  }
  if (classes !== undefined) {
    return <LabelControls state={state} onChange={onChange} hasMap={hasMap} classes={classes} />;
  }
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
        {/* No swatch, unlike its three neighbours: this layer is drawn in the colour of its
            own verdict — green on target, amber off it, blue where nothing was checked — so
            a single dot would name one of three and legend the other two wrongly. */}
        <ToggleChip
          checked={state.peak}
          disabled={!hasMap}
          onCheckedChange={(peak) => onChange({ peak })}
          title={
            hasMap
              ? "Where the map peaked, in the tolerance window the localization verdict was decided in"
              : "This run recorded no anomaly map"
          }
        >
          peak
        </ToggleChip>
      </div>

      {/* Only while it can do something. A cut with no segmentation on screen is a control
          for an invisible effect. */}
      {state.region && (
        <div className="flex min-w-56 flex-1 items-center gap-2">
          <span className="shrink-0 text-xs text-fg-muted">map cut</span>
          <Slider
            aria-label="Map cut, as a fraction of the run's map range"
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

function LabelControls({
  state,
  onChange,
  hasMap,
  classes,
}: {
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  hasMap: boolean;
  classes: readonly string[];
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <ToggleChip
          checked={state.region}
          onCheckedChange={(region) => onChange({ region })}
          title="The label map the method wrote, drawn solid"
        >
          prediction
        </ToggleChip>
        <ToggleChip
          checked={state.truth}
          onCheckedChange={(truth) => onChange({ truth })}
          title="The annotation over the run's classes, drawn dashed"
        >
          ground truth
        </ToggleChip>
        <ToggleChip
          checked={state.heatmap}
          disabled={!hasMap}
          onCheckedChange={(heatmap) => onChange({ heatmap })}
          swatch={HEATMAP_SWATCH}
          title={
            hasMap
              ? "How strongly each pixel was given any class"
              : "This run recorded no foreground map"
          }
        >
          foreground
        </ToggleChip>
      </div>
      <ClassLegend classes={classes} note="prediction solid · truth dashed" />
    </div>
  );
}

function ClassLegend({ classes, note }: { classes: readonly string[]; note: string }) {
  return (
    <ul aria-label="Classes" className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
      {classes.map((name, index) => (
        <li key={name} className="flex items-center gap-1.5 text-fg-muted">
          <span
            aria-hidden
            className="inline-block size-2.5 rounded-sm"
            style={{ backgroundColor: classColour(index + 1) }}
          />
          {name}
        </li>
      ))}
      <li className="text-fg-subtle">{note}</li>
    </ul>
  );
}

/** What each box tone means, in the tokens `boxTones.ts` draws with. */
const VERDICT_KEY = [
  { label: "match", swatch: "bg-normal" },
  { label: "false positive", swatch: "bg-defect" },
  { label: "missed", swatch: "bg-warn" },
];

function BoxControls({
  state,
  onChange,
  hasMap,
  boxes,
}: {
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  hasMap: boolean;
  boxes: { classes: readonly string[]; rule: string | undefined };
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-1.5">
          <ToggleChip
            checked={state.region}
            onCheckedChange={(region) => onChange({ region })}
            title="The detections the cut keeps, drawn solid and tagged with class and confidence"
          >
            prediction
          </ToggleChip>
          <ToggleChip
            checked={state.truth}
            onCheckedChange={(truth) => onChange({ truth })}
            title="The true boxes over the run's classes, drawn dashed"
          >
            ground truth
          </ToggleChip>
          <ToggleChip
            checked={state.heatmap}
            disabled={!hasMap}
            onCheckedChange={(heatmap) => onChange({ heatmap })}
            swatch={HEATMAP_SWATCH}
            title={hasMap ? "The method's own map, where it wrote one" : "This run recorded no map"}
          >
            map
          </ToggleChip>
        </div>
        <ul aria-label="Box verdicts" className="flex flex-wrap items-center gap-x-3 text-xs">
          {VERDICT_KEY.map((entry) => (
            <li key={entry.label} className="flex items-center gap-1.5 text-fg-muted">
              <span aria-hidden className={cn("inline-block h-0.5 w-3", entry.swatch)} />
              {entry.label}
            </li>
          ))}
        </ul>
        <ClassLegend classes={boxes.classes} note="class on the tag · prediction solid · truth dashed" />
      </div>
      {boxes.rule !== undefined && (
        <p className="text-xs text-fg-muted">
          Matched at IoU 0.5 · kept by the run&apos;s cut:{" "}
          <span className="font-mono text-fg">{boxes.rule}</span>
        </p>
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
