/**
 * One grouped sample: every channel it has, side by side or one at a time.
 *
 * **Channel tabs are rendered from the sample's own image list.** There is no constant
 * anywhere in this file for how many there should be, so the two-channel capture group in
 * the reference data renders through exactly the same code as every three-channel one —
 * which is the UI half of "channel count is data, never schema" (ADR-0005, §12).
 *
 * Zoom and pan are shared across channels rather than per channel: the views are
 * near-simultaneous images of the same physical object, so comparing them is only useful
 * if they move together.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import {
  PAGE_SIZE,
  readBrowseState,
  toSampleQuery,
  writeBrowseState,
} from "../api/browseState";
import type { ImageSummary, Label } from "../api/client";
import { imageUrl } from "../api/imageUrl";
import { ChannelTabs } from "../components/ChannelTabs";
import { Badge, Button, Empty, ErrorBox, Panel } from "../components/ui";
import { useSample, useSamples, useSetLabel } from "../hooks/useCatalog";

const MIN_ZOOM = 1;
const MAX_ZOOM = 12;
const ZOOM_SENSITIVITY = 0.0015;

const LABEL_TONE: Record<Label, "normal" | "defect" | "unlabeled"> = {
  normal: "normal",
  defect: "defect",
  unlabeled: "unlabeled",
};

/** Keyboard shortcuts, for fast passes over unlabelled data (§12). */
const LABEL_KEYS: Record<string, Label> = { n: "normal", d: "defect", u: "unlabeled" };

interface View {
  zoom: number;
  x: number;
  y: number;
}

const RESET: View = { zoom: 1, x: 0, y: 0 };

export function SampleRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const sampleId = Number(params["sampleId"]);

  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const browse = readBrowseState(searchParams);
  const search = writeBrowseState(browse).toString();

  const sample = useSample(datasetId, sampleId);
  // The grid's own query, rebuilt from the filters carried in the URL. Arriving from the
  // browser this is a cache hit, so knowing the neighbours costs nothing.
  const neighbours = useSamples(datasetId, toSampleQuery(browse));
  const setLabel = useSetLabel(datasetId);

  const [active, setActive] = useState(0);
  const [sideBySide, setSideBySide] = useState(false);
  const [view, setView] = useState<View>(RESET);
  const [autoAdvance, setAutoAdvance] = useState(true);
  /**
   * Crossing a page boundary cannot be done in one move: the neighbouring page is not
   * loaded yet. We shift the window and record which end of it we were heading for, and
   * an effect finishes the move once the data lands.
   */
  const [pendingEdge, setPendingEdge] = useState<"first" | "last" | null>(null);

  const siblings = neighbours.data?.items ?? [];
  const total = neighbours.data?.total ?? 0;
  const index = siblings.findIndex((item) => item.id === sampleId);

  const open = useCallback(
    (id: number) => navigate(`/datasets/${datasetId}/samples/${id}${search ? `?${search}` : ""}`, {
      // Replace rather than push: a labelling pass over two hundred samples should leave
      // one history entry to go back to, not two hundred.
      replace: true,
    }),
    [navigate, datasetId, search],
  );

  const step = useCallback(
    (delta: number) => {
      if (index < 0) return;
      const target = siblings[index + delta];
      if (target) {
        void open(target.id);
        return;
      }
      const nextOffset = browse.offset + delta * PAGE_SIZE;
      if (nextOffset < 0 || nextOffset >= total) return;
      setPendingEdge(delta > 0 ? "first" : "last");
      setSearchParams(writeBrowseState({ ...browse, offset: nextOffset }), { replace: true });
    },
    [index, siblings, browse, total, open, setSearchParams],
  );

  useEffect(() => {
    if (pendingEdge === null || neighbours.isFetching || siblings.length === 0) return;
    const target = pendingEdge === "first" ? siblings[0] : siblings[siblings.length - 1];
    setPendingEdge(null);
    if (target) void open(target.id);
  }, [pendingEdge, neighbours.isFetching, siblings, open]);

  const apply = useCallback(
    (label: Label) => {
      setLabel.mutate({ sampleId, label });
      // Advance from the list as it is *now*. Labelling invalidates the page, and under a
      // `label=unlabeled` filter the current sample is about to leave the set — so the
      // next id has to be read before that happens, not after.
      if (autoAdvance) step(1);
    },
    [setLabel, sampleId, autoAdvance, step],
  );

  /** The next preview, fetched before it is asked for, so paging feels instant. */
  useEffect(() => {
    const cover = index < 0 ? undefined : siblings[index + 1]?.images[0];
    if (!cover) return;
    const preload = new globalThis.Image();
    preload.src = imageUrl(cover.id, "preview");
  }, [index, siblings]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
      }
      const label = LABEL_KEYS[event.key.toLowerCase()];
      if (label) apply(label);
      if (event.key === "0") setView(RESET);
      if (event.key === "ArrowRight") step(1);
      if (event.key === "ArrowLeft") step(-1);
    };
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, [apply, step]);

  if (sample.error) return <ErrorBox>{sample.error.message}</ErrorBox>;
  if (sample.isPending || !sample.data) return <Empty>Loading…</Empty>;

  const current = sample.data;
  const images = current.images;
  const shown = images[Math.min(active, images.length - 1)];

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <Link
            to={`/datasets/${datasetId}${search ? `?${search}` : ""}`}
            className="text-xs text-slate-500 hover:underline dark:text-slate-400"
          >
            ← back to the browser
          </Link>
          <h2 className="font-mono text-lg font-semibold tracking-tight">
            {current.group_key}/{current.external_id}
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <Badge tone={LABEL_TONE[current.label]}>{current.label}</Badge>
          <Badge tone={current.label_source === "manual" ? "info" : "neutral"}>
            {current.label_source}
          </Badge>
        </div>
      </div>

      {/* Paging walks the filtered set the grid was showing, not the whole dataset. */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="flex items-center gap-2">
          <Button
            disabled={index <= 0 && browse.offset === 0}
            onClick={() => step(-1)}
            aria-label="Previous sample"
          >
            ←
          </Button>
          <Button
            disabled={index < 0 || browse.offset + index + 1 >= total}
            onClick={() => step(1)}
            aria-label="Next sample"
          >
            →
          </Button>
          <span className="text-sm text-slate-500 dark:text-slate-400">
            {index < 0 ? (
              // Labelling under a label filter drops the sample out of its own set. Say
              // so rather than silently disabling the arrows.
              <>no longer matches the current filters</>
            ) : (
              <>
                {browse.offset + index + 1} of {total}
              </>
            )}
          </span>
        </span>

        <label className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          <input
            type="checkbox"
            checked={autoAdvance}
            onChange={(event) => setAutoAdvance(event.target.checked)}
            className="h-4 w-4 accent-slate-900 dark:accent-slate-100"
          />
          Advance after labelling
        </label>
      </div>

      <Panel
        title="Label"
        actions={
          <span className="text-xs text-slate-500 dark:text-slate-400">
            keys: n · d · u — ← → to page — 0 resets the view
          </span>
        }
      >
        <div className="flex gap-2">
          {(["normal", "defect", "unlabeled"] as Label[]).map((label) => (
            <Button
              key={label}
              variant={current.label === label ? "primary" : "secondary"}
              disabled={setLabel.isPending}
              onClick={() => apply(label)}
            >
              {label}
            </Button>
          ))}
        </div>
        {setLabel.error && <ErrorBox>{setLabel.error.message}</ErrorBox>}
      </Panel>

      <Panel
        title={`Channels (${images.length})`}
        actions={
          <span className="flex items-center gap-2">
            <Button onClick={() => setView(RESET)}>Reset view</Button>
            <Button onClick={() => setSideBySide((value) => !value)}>
              {sideBySide ? "Single" : "Side by side"}
            </Button>
          </span>
        }
      >
        {images.length === 0 ? (
          <Empty>This sample has no images.</Empty>
        ) : (
          <div className="flex flex-col gap-3">
            {!sideBySide && (
              <ChannelTabs images={images} active={active} onSelect={setActive} />
            )}

            <div
              className={
                sideBySide
                  ? "grid gap-2"
                  : ""
              }
              style={
                sideBySide
                  ? { gridTemplateColumns: `repeat(${images.length}, minmax(0, 1fr))` }
                  : undefined
              }
            >
              {(sideBySide ? images : shown ? [shown] : []).map((image) => (
                <ZoomPan key={image.id} image={image} view={view} onView={setView} />
              ))}
            </div>
          </div>
        )}
      </Panel>

      <Panel title="Metadata">
        <table className="w-full text-left text-sm">
          <thead className="text-xs tracking-wide text-slate-500 uppercase dark:text-slate-400">
            <tr>
              <th className="pr-4 pb-2 whitespace-nowrap">Channel</th>
              <th className="pr-4 pb-2 whitespace-nowrap">Size</th>
              <th className="pr-4 pb-2 whitespace-nowrap">Depth</th>
              <th className="pr-4 pb-2 whitespace-nowrap">Bytes</th>
              <th className="pb-2">Path</th>
            </tr>
          </thead>
          <tbody>
            {images.map((image) => (
              <tr key={image.id} className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1.5 pr-4 font-mono text-xs whitespace-nowrap">
                  {image.channel ?? "—"}
                </td>
                <td className="py-1.5 pr-4 font-mono text-xs whitespace-nowrap">
                  {image.width}×{image.height}
                </td>
                <td className="py-1.5 pr-4 font-mono text-xs whitespace-nowrap">
                  {image.bit_depth}-bit
                </td>
                <td className="py-1.5 pr-4 font-mono text-xs whitespace-nowrap">
                  {(image.file_size / 1_000_000).toFixed(2)} MB
                </td>
                <td className="py-1.5 font-mono text-xs break-all text-slate-400">{image.path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}

/**
 * One image with shared zoom and pan.
 *
 * `preview` while zoomed out, `full` once the zoom would show real pixels — a lossless
 * PNG is what makes a defect judgeable, but fetching it for a thumbnail-sized view would
 * be several megabytes for nothing.
 */
function ZoomPan({
  image,
  view,
  onView,
}: {
  image: ImageSummary;
  view: View;
  onView: (view: View) => void;
}) {
  const dragging = useRef<{ x: number; y: number } | null>(null);
  const tier = view.zoom > 2 ? "full" : "preview";

  return (
    <div
      className="relative h-96 cursor-grab overflow-hidden rounded bg-slate-100 select-none dark:bg-slate-800"
      onWheel={(event) => {
        const next = Math.min(
          MAX_ZOOM,
          Math.max(MIN_ZOOM, view.zoom * (1 - event.deltaY * ZOOM_SENSITIVITY)),
        );
        onView(next === MIN_ZOOM ? RESET : { ...view, zoom: next });
      }}
      onPointerDown={(event) => {
        dragging.current = { x: event.clientX - view.x, y: event.clientY - view.y };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const origin = dragging.current;
        if (!origin) return;
        onView({ ...view, x: event.clientX - origin.x, y: event.clientY - origin.y });
      }}
      onPointerUp={() => {
        dragging.current = null;
      }}
    >
      <img
        src={imageUrl(image.id, tier)}
        alt={image.channel ?? "unassigned channel"}
        draggable={false}
        className="h-full w-full object-contain"
        style={{
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`,
          transformOrigin: "center",
        }}
      />
      <span className="absolute bottom-1 left-1 rounded bg-black/60 px-1 font-mono text-[10px] text-white">
        {image.channel ?? "unassigned"} · {view.zoom.toFixed(1)}×
      </span>
    </div>
  );
}
