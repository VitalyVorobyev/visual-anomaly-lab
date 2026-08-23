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
 *
 * **The image is the screen.** This used to be four stacked panels in a 72rem reading
 * column, with the picture third, boxed at a fixed `h-96` below a page header, a pager row
 * and a label panel — so the one thing the screen exists for was the smallest element on
 * it, and a 4 MP capture was judged through a letterbox. It is a canvas route now: a thin
 * band of identity and paging, the image taking every pixel that is left, and the controls
 * in a rail beside it. The rail is a *peer column*, which is the one case
 * `docs/architecture/frontend.md` allows to scroll on the page's own axis; the image
 * region itself never does.
 */

import { ArrowLeft, PenLine } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import { PAGE_SIZE, readBrowseState, toSampleQuery, writeBrowseState } from "../api/browseState";
import type { ImageSummary, Label, SampleSummary } from "../api/client";
import { preferredImageIndex } from "../api/defaultChannel";
import { imageUrl } from "../api/imageUrl";
import { ChannelTabs } from "../components/ChannelTabs";
import { Badge, Button, cn, Disclosure, Empty, ErrorBox, focusRing, FULL_TIER_ZOOM, RESET_VIEW, Skeleton, Switch, Tooltip, ZoomPanCanvas, type View } from "@vitavision/lab-ui";
import { useDataset, useSample, useSamples, useSetLabel } from "../hooks/useCatalog";

const LABEL_TONE: Record<Label, "normal" | "defect" | "unlabeled"> = {
  normal: "normal",
  defect: "defect",
  unlabeled: "unlabeled",
};

/** Keyboard shortcuts, for fast passes over unlabelled data (§12). */
const LABEL_KEYS: Record<string, Label> = { n: "normal", d: "defect", u: "unlabeled" };

/** The same map the other way, so each button can print the key that presses it. */
const KEY_FOR: Record<Label, string> = { normal: "n", defect: "d", unlabeled: "u" };

const LABELS: Label[] = ["normal", "defect", "unlabeled"];

/** Fit-to-window, which is what zoom 1 means in this frame. */
const RESET = RESET_VIEW;

export function SampleRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const sampleId = Number(params["sampleId"]);

  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const browse = readBrowseState(searchParams);
  const search = writeBrowseState(browse).toString();

  const sample = useSample(datasetId, sampleId);
  // Only for the name in the band. Arriving from the browser it is already in cache, and
  // on a deep link it is one small request beside the two this screen makes anyway.
  const dataset = useDataset(datasetId);
  // The grid's own query, rebuilt from the filters carried in the URL. Arriving from the
  // browser this is a cache hit, so knowing the neighbours costs nothing.
  const neighbours = useSamples(datasetId, toSampleQuery(browse));
  const setLabel = useSetLabel(datasetId);

  /**
   * `null` until the reader picks a channel, so the grid's filter can decide the first one.
   *
   * Opening a sample from a grid filtered to dark field and landing on bright field reads
   * as the filter being ignored. Once a tab *is* clicked the choice sticks, including
   * across paging to the next sample, which is what a labelling pass through one
   * illumination needs.
   */
  const [active, setActive] = useState<number | null>(null);
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
  // The sample is not in the page the browser is holding, so paging has no anchor to move
  // from and both arrows are dead rather than one of them being quietly inert. Two quite
  // different things get you here and the old copy only named one of them: labelling under
  // `label=unlabeled` drops the sample out of its own set, and a link opened cold starts at
  // offset 0 whatever page the sample is really on.
  const adrift = index < 0;
  const filtered =
    browse.label !== null || browse.channelId !== null || browse.splitId !== null;

  const open = useCallback(
    (id: number) =>
      navigate(`/datasets/${datasetId}/samples/${id}${search ? `?${search}` : ""}`, {
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

  const current = sample.data;
  const images = current?.images ?? [];
  // The rail's channel filter wins, because the reader asked for it on the way in and a
  // viewer that ignored it made the filter look inert. With no filter the dataset's own
  // default answers instead, and only then the first image.
  //
  // `undefined` rather than `null`: `BrowseState` is required-but-nullable on `undefined`,
  // and the old `=== null` test could never be true — the unfiltered case was reaching a
  // `findIndex` against `undefined`, landing on `-1`, and being rescued by the `Math.max`.
  // It gave the right answer for the wrong reason, and there is now a third case to tell
  // apart from it.
  const filteredChannel =
    browse.channelId === undefined
      ? -1
      : images.findIndex((image) => image.channel_id === browse.channelId);
  const activeIndex =
    active ??
    (filteredChannel === -1
      ? preferredImageIndex(images, dataset.data?.default_channel)
      : filteredChannel);
  const shown = images[Math.min(activeIndex, images.length - 1)];

  /** The next preview, fetched before it is asked for, so paging feels instant. */
  useEffect(() => {
    const cover = index < 0 ? undefined : siblings[index + 1]?.images[0];
    if (!cover) return;
    const preload = new globalThis.Image();
    preload.src = imageUrl(cover.id, "preview");
  }, [index, siblings]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      // Anywhere text is being entered, these are letters rather than commands. The old
      // guard named two element types and missed `<textarea>` and `contentEditable`, so
      // typing "n" in a free-text field would silently relabel the sample behind it.
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLSelectElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
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

  return (
    <div data-layout="sample" className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
      <header
        data-band="sample"
        className="flex shrink-0 items-center justify-between gap-x-4 border-b border-line px-4 py-2"
      >
        <div className="flex min-w-0 items-center gap-x-3">
          {/* The dataset's name lives on the button rather than beside the title: a VisA
              sample's own `group_key` already begins with it, so printing both reads
              `candle / candle/Data/Images/Normal/0951`. */}
          <Tooltip content={dataset.data ? `Back to ${dataset.data.name}` : "Back to the browser"}>
            <Link
              to={`/datasets/${datasetId}${search ? `?${search}` : ""}`}
              aria-label={
                dataset.data ? `Back to ${dataset.data.name}` : "Back to the browser"
              }
              className={cn(
                "shrink-0 rounded-control p-1 text-fg-muted transition-colors hover:bg-raised hover:text-fg",
                focusRing,
              )}
            >
              <ArrowLeft className="size-4" />
            </Link>
          </Tooltip>

          <h1 className="min-w-0 truncate font-mono text-sm text-fg">
            {current ? (
              `${current.group_key}/${current.external_id}`
            ) : (
              <Skeleton className="h-4 w-64" />
            )}
          </h1>

          {current && (
            <span className="flex shrink-0 items-center gap-1.5">
              <Badge tone={LABEL_TONE[current.label]}>{current.label}</Badge>
              <Badge tone={current.label_source === "manual" ? "info" : "neutral"}>
                {current.label_source}
              </Badge>
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-3">
          {/* Paging walks the filtered set the grid was showing, not the whole dataset. */}
          <span className="flex items-center gap-1.5">
            <Button
              size="sm"
              disabled={adrift || (index === 0 && browse.offset === 0)}
              onClick={() => step(-1)}
              aria-label="Previous sample"
            >
              ←
            </Button>
            <Button
              size="sm"
              disabled={adrift || browse.offset + index + 1 >= total}
              onClick={() => step(1)}
              aria-label="Next sample"
            >
              →
            </Button>
            <span className="font-mono text-xs whitespace-nowrap text-fg-muted tabular-nums">
              {adrift ? (
                // Say which of the two it is, rather than silently disabling the arrows.
                <>{filtered ? "off the filtered set" : "not on this page"}</>
              ) : (
                <>
                  {browse.offset + index + 1} of {total}
                </>
              )}
            </span>
          </span>

          {shown && (
            <Link to={`/datasets/${datasetId}/annotate/${sampleId}/${shown.id}`}>
              <Button size="sm" icon={<PenLine />}>
                Open in editor
              </Button>
            </Link>
          )}
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-1 flex-col overflow-hidden bg-canvas p-3">
          {sample.error ? (
            <ErrorBox>{sample.error.message}</ErrorBox>
          ) : !current ? (
            <Skeleton className="h-full w-full" />
          ) : images.length === 0 ? (
            <Empty>This sample has no images.</Empty>
          ) : (
            <div
              className={cn("min-h-0 flex-1", sideBySide && "grid gap-2")}
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
          )}
        </div>

        <aside
          data-scroll="rail"
          className="flex w-72 shrink-0 flex-col overflow-y-auto overscroll-contain border-l border-line bg-ground"
        >
          <RailSection title="Label" hint="n · d · u">
            <div className="flex flex-col gap-1.5">
              {LABELS.map((label) => (
                <Button
                  key={label}
                  className="justify-between"
                  variant={current?.label === label ? "primary" : "secondary"}
                  disabled={!current || setLabel.isPending}
                  onClick={() => apply(label)}
                >
                  {label}
                  <span
                    className={cn(
                      "font-mono text-[11px]",
                      current?.label === label ? "opacity-70" : "text-fg-subtle",
                    )}
                    aria-hidden
                  >
                    {KEY_FOR[label]}
                  </span>
                </Button>
              ))}
            </div>
            <Switch
              checked={autoAdvance}
              onCheckedChange={setAutoAdvance}
              label="Advance after labelling"
            />
            {setLabel.error && <ErrorBox>{setLabel.error.message}</ErrorBox>}
          </RailSection>

          {/* "Channels (1)" was a section heading that counted to one. The section is about
              looking at the image; the channel controls are part of that, and only exist
              when a sample has more than one. */}
          {images.length > 0 && (
            <RailSection title="View" hint={images.length > 1 ? `${images.length} channels` : ""}>
              {images.length > 1 && (
                <>
                  <ChannelTabs images={images} active={activeIndex} onSelect={setActive} />
                  <Button size="sm" onClick={() => setSideBySide((value) => !value)}>
                    {sideBySide ? "One at a time" : "Side by side"}
                  </Button>
                </>
              )}
              <Button
                size="sm"
                className="justify-between"
                onClick={() => setView(RESET)}
              >
                Reset view
                <span className="font-mono text-[11px] text-fg-subtle" aria-hidden>
                  0
                </span>
              </Button>
            </RailSection>
          )}

          {current && (
            <RailSection title={null}>
              {/* The path is here rather than on screen for the same reason the dataset
                  band moved its root path behind a mark: it is consulted, not read. */}
              <Disclosure summary="Files" count={images.length}>
                <ul className="flex flex-col gap-3">
                  {images.map((image) => (
                    <li key={image.id} className="flex flex-col gap-0.5">
                      <span className="font-mono text-xs text-fg">
                        {image.channel ?? "unassigned"}
                      </span>
                      <span className="font-mono text-[11px] text-fg-muted">
                        {image.width}×{image.height} · {image.bit_depth}-bit ·{" "}
                        {(image.file_size / 1_000_000).toFixed(2)} MB
                      </span>
                      <span className="font-mono text-[11px] break-all text-fg-subtle">
                        {image.path}
                      </span>
                    </li>
                  ))}
                </ul>
              </Disclosure>
              <SampleNotes sample={current} />
            </RailSection>
          )}
        </aside>
      </div>
    </div>
  );
}

/** One block of the rail. A rule between blocks, rather than a box around each. */
function RailSection({
  title,
  hint,
  children,
}: {
  title: string | null;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-2.5 border-b border-line p-4 last:border-b-0">
      {title !== null && (
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-xs font-semibold tracking-tight text-fg">{title}</h2>
          {hint && <span className="font-mono text-[11px] text-fg-subtle">{hint}</span>}
        </div>
      )}
      {children}
    </section>
  );
}

/**
 * The sample's own note, which the old screen fetched and never rendered.
 *
 * Read-only: `PATCH /samples/{id}` takes a label and nothing else, so there is no endpoint
 * to write this back. Showing it is still worth doing — an import that recorded why a part
 * was marked defect should not have to be read out of the database.
 */
function SampleNotes({ sample }: { sample: SampleSummary }) {
  if (!sample.notes) return null;
  return <p className="text-xs leading-relaxed text-fg-muted">{sample.notes}</p>;
}

/**
 * One image with shared zoom and pan.
 *
 * `preview` while zoomed out, `full` once the zoom would show real pixels — a lossless
 * PNG is what makes a defect judgeable, but fetching it for a thumbnail-sized view would
 * be several megabytes for nothing.
 *
 * This used to be a second, hand-written copy of `ZoomPanCanvas` with its own constants.
 * The two drifted exactly as far as you would expect: this one had a reset button and a
 * `0` key, that one had neither, and every fix to the gesture had to be written twice or
 * only landed on one screen. It is one component now, and the rail keeps its own reset
 * button — which is why `fitLabel` is off here.
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
  const tier = view.zoom > FULL_TIER_ZOOM ? "full" : "preview";

  return (
    <ZoomPanCanvas
      view={view}
      onView={onView}
      className="h-full min-h-0 border-0 bg-transparent"
      nativeWidth={image.width}
      fitLabel={null}
      label={`${image.channel ?? "unassigned"} · ${view.zoom.toFixed(1)}×`}
    >
      <img
        src={imageUrl(image.id, tier)}
        alt={image.channel ?? "unassigned channel"}
        draggable={false}
        className="h-full w-full object-contain"
      />
    </ZoomPanCanvas>
  );
}
