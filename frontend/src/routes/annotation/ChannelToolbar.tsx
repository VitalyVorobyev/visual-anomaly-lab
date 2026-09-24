/**
 * The channel strip over the canvas column, shown whenever a sample has more than one image:
 * which channel is edited, how a second one is shown beside it, and copying regions across.
 */

import { Copy } from "lucide-react";

import type { PaneMode } from "../../api/annotationPanes";
import type { SampleSummary } from "../../api/client";
import { ChannelTabs } from "../../components/ChannelTabs";
import { Button, SegmentedControl, Slider } from "@vitavision/lab-ui";
import type { ChannelPanes } from "./useChannelPanes";
import type { Workspace } from "./useWorkspace";

export function ChannelToolbar({
  sample,
  perSample,
  panes,
  workspace,
  shapeCount,
}: {
  sample: SampleSummary;
  perSample: boolean;
  panes: ChannelPanes;
  workspace: Workspace;
  shapeCount: number;
}) {
  const { paneMode, setPaneMode, overlayOpacity, setOverlayOpacity } = workspace;
  const { activeIndex, referenceIndex, copyable } = panes;
  return (
    <div className="flex h-11 shrink-0 items-center gap-3 border-b border-line bg-surface px-3">
      {/* The tabs are what gives way when the strip is over-subscribed, because they are the
          one thing here that reads perfectly well half-scrolled. Everything to the right is a
          control with a usable minimum size, and the blend slider in particular collapsed to
          a few pixels of track once the view switch stopped wrapping — a slider narrower than
          its own thumb is a rendering fault, not a tight fit. */}
      <div className="min-w-0 flex-1 overflow-x-auto">
        <ChannelTabs
          images={sample.images}
          active={activeIndex}
          onSelect={(index) => void panes.openChannel(index)}
        />
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {!perSample && (
          <Button
            icon={<Copy />}
            disabled={shapeCount === 0 || copyable.length === 0}
            title={
              shapeCount === 0
                ? "Draw a region first"
                : copyable.length === 0
                  ? "No other channel of this part shares this source frame"
                  : "Put these regions on the other channels of this part"
            }
            onClick={panes.copy.start}
          >
            Copy to…
          </Button>
        )}
        {paneMode === "overlay" && (
          <div className="flex w-56 shrink-0 items-center gap-2">
            <span className="shrink-0 text-[11px] text-fg-muted">Blend</span>
            <Slider
              aria-label="Overlay opacity"
              value={overlayOpacity}
              min={0}
              max={1}
              step={0.02}
              onValueChange={setOverlayOpacity}
              readout={`${Math.round(overlayOpacity * 100)}%`}
            />
          </div>
        )}
        {paneMode !== "single" && referenceIndex !== null && (
          // The same control as the strip on the left, because it makes the same kind of
          // choice. It was a dropdown, which put two unlike controls a few centimetres apart
          // in one row for no reason a reader could see.
          //
          // Every channel is listed, with the one already in the left pane disabled rather
          // than filtered out: the two strips then hold the same channels in the same order,
          // and the second does not reshuffle itself every time the first changes. What the
          // disabled tab shows is `resolveReference`'s wrap made visible — the reason a
          // two-channel part cannot put the same photograph in both panes.
          <div className="shrink-0">
            <ChannelTabs
              label="Second channel"
              images={sample.images}
              active={referenceIndex}
              onSelect={(index) => workspace.setReferenceIndex(index)}
              unavailable={{ index: activeIndex, reason: "Already in the left pane" }}
            />
          </div>
        )}
        <SegmentedControl
          aria-label="Channel view"
          value={paneMode}
          options={[
            { value: "single", label: "One" },
            { value: "compare", label: "Side by side" },
            { value: "overlay", label: "Blend" },
          ]}
          onValueChange={(value) => setPaneMode(value as PaneMode)}
        />
      </div>
    </div>
  );
}
