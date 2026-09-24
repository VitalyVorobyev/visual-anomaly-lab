/**
 * The reference studio: choose a class's references by looking at them, then freeze them
 * into a few-shot run (ADR-0040).
 *
 * A few-shot run is only as good as the handful of samples it learns from, and a seeded draw
 * picks them blind. Here the reader sees every sample that shows the class, its region
 * outlined on the stage, and picks — then one press makes the `manual` split, the
 * experiment and its Train & score, and ends on the run. Nothing here reads results: the
 * run page is the one place a run is read, so the studio cannot grow a second one.
 *
 * Built from what the other viewers use: `SampleTile` for the candidates, `SampleStage` for
 * the picture, `RailSection` for the rails, the lab-ui controls for the rest. The session is
 * the URL (`refs`, `focus`, `method`, `profile`), so a reload or a shared link keeps it.
 */

import { ArrowLeft } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import type { SampleSummary } from "../api/client";
import { preferredImageIndex } from "../api/defaultChannel";
import { classMaskUrl } from "../api/imageUrl";
import { RailSection } from "../components/viewer/RailSection";
import { SampleStage } from "../components/viewer/SampleStage";
import { useAnnotationLabels, useClassCoverage } from "../hooks/useAnnotations";
import { useDataset, useSample, useSamples } from "../hooks/useCatalog";
import { isUsableBuild } from "../hooks/useDatasetReadiness";
import { useModelTypes } from "../hooks/useExperiments";
import { useRegionBuild, useRegionProfiles } from "../hooks/useRegionProfiles";
import { useFreezeReferences } from "../hooks/useStudio";
import {
  Button,
  cn,
  Empty,
  ErrorBox,
  focusRing,
  ReadoutStrip,
  Select,
  Skeleton,
  Tooltip,
  type StageView,
} from "@vitavision/lab-ui";
import { SampleTile } from "./dataset/SampleTile";

/** ADR-0040: one to ten references define a class. */
export const MAX_REFERENCES = 10;
const PAGE = 48;
/** The method the studio offers first: the public gate's default (docs/measurements.md). */
const PREFERRED_METHOD = "fss_dino";

function readIds(raw: string | null): number[] {
  if (!raw) return [];
  const seen = new Set<number>();
  for (const part of raw.split(",")) {
    const value = Number(part);
    if (Number.isInteger(value) && value > 0) seen.add(value);
  }
  return [...seen];
}

export function StudioRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const classKey = params["labelKey"] ?? "";
  const navigate = useNavigate();
  const [search, setSearch] = useSearchParams();

  const references = readIds(search.get("refs"));
  const focus = Number(search.get("focus")) || references[0] || undefined;
  const [offset, setOffset] = useState(0);
  const [view, setView] = useState<StageView | null>(null);

  const dataset = useDataset(datasetId);
  const labels = useAnnotationLabels(datasetId);
  const coverage = useClassCoverage(datasetId);
  const candidates = useSamples(datasetId, {
    classKey,
    presence: "present",
    limit: PAGE,
    offset,
  });
  const focused = useSample(datasetId, focus);
  const catalog = useModelTypes();
  const profiles = useRegionProfiles(datasetId);

  const label = labels.data?.find((entry) => entry.key === classKey);
  const counts = coverage.data?.find((entry) => entry.label_key === classKey);
  const methods = useMemo(
    () =>
      (catalog.data?.methods ?? []).filter(
        (method) =>
          method.capabilities.tasks.includes("few_shot_segmentation") &&
          method.availability.available,
      ),
    [catalog.data],
  );
  const methodKey =
    search.get("method") ??
    (methods.find((method) => method.key === PREFERRED_METHOD) ?? methods[0])?.key;
  const method = methods.find((entry) => entry.key === methodKey);
  const profileId =
    Number(search.get("profile")) ||
    (profiles.data?.length === 1 ? profiles.data[0]?.id : undefined);
  const build = useRegionBuild(profileId);
  const freeze = useFreezeReferences();

  const update = (next: Record<string, string | undefined>) => {
    const merged = new URLSearchParams(search);
    for (const [key, value] of Object.entries(next)) {
      if (value === undefined || value === "") merged.delete(key);
      else merged.set(key, value);
    }
    setSearch(merged, { replace: true });
  };
  const toggle = (sampleId: number) => {
    const next = references.includes(sampleId)
      ? references.filter((id) => id !== sampleId)
      : [...references, sampleId];
    update({ refs: next.join(",") });
  };

  // Said beside the button, not hidden in a tooltip: why it cannot be pressed yet.
  const blocker =
    references.length === 0
      ? "Choose at least one reference."
      : references.length > MAX_REFERENCES
        ? `At most ${MAX_REFERENCES} references.`
        : method === undefined
          ? "No few-shot method is available."
          : profileId === undefined
            ? "Choose a region profile."
            : !isUsableBuild(build.data)
              ? "The region profile is not built."
              : null;

  const images = focused.data?.images ?? [];
  const shown = images[preferredImageIndex(images, dataset.data?.default_channel)];

  return (
    <div data-layout="studio" className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
      <header
        data-band="studio"
        className="flex shrink-0 items-center justify-between gap-x-4 border-b border-line px-4 py-2"
      >
        <div className="flex min-w-0 items-center gap-x-3">
          <Tooltip content="Back to the classes">
            <Link
              to={`/datasets/${datasetId}/annotate`}
              aria-label="Back to the classes"
              className={cn(
                "shrink-0 rounded-control p-1 text-fg-muted transition-colors hover:bg-raised hover:text-fg",
                focusRing,
              )}
            >
              <ArrowLeft className="size-4" />
            </Link>
          </Tooltip>
          <h1 className="min-w-0 truncate text-sm font-semibold text-fg">
            Reference studio · {label?.name ?? classKey}
          </h1>
          {counts && (
            <ReadoutStrip
              className="hidden min-w-0 md:flex"
              items={[
                { value: `${counts.present} show it` },
                { value: `${counts.absent} without it` },
                { value: `${counts.unlabeled} unanswered` },
              ]}
            />
          )}
        </div>
        <span className="shrink-0 text-xs text-fg-muted">{dataset.data?.name}</span>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside
          data-scroll="rail"
          aria-label="Samples that show the class"
          className="flex w-80 shrink-0 flex-col overflow-y-auto overscroll-contain border-r border-line"
        >
          <RailSection
            title="Samples that show it"
            hint={candidates.data ? `${candidates.data.total}` : undefined}
          >
            {candidates.error && <ErrorBox>{candidates.error.message}</ErrorBox>}
            {candidates.isPending && <Skeleton className="h-40 w-full" />}
            {candidates.data?.total === 0 && (
              <Empty>
                No sample shows this class yet.{" "}
                <Link className="text-signal underline" to={`/datasets/${datasetId}/annotate`}>
                  Annotate some
                </Link>
                .
              </Empty>
            )}
            <div className="grid grid-cols-2 gap-2">
              {(candidates.data?.items ?? []).map((sample: SampleSummary) => (
                <SampleTile
                  key={sample.id}
                  datasetId={datasetId}
                  sample={sample}
                  search=""
                  defaultChannel={dataset.data?.default_channel}
                  selected={references.includes(sample.id)}
                  active={sample.id === focus}
                  onSelect={() => toggle(sample.id)}
                  onOpen={() => update({ focus: String(sample.id) })}
                  selectLabel={`Use ${sample.group_key}/${sample.external_id} as a reference`}
                />
              ))}
            </div>
            {candidates.data && candidates.data.total > PAGE && (
              <div className="flex items-center justify-between gap-2">
                <Button
                  size="sm"
                  disabled={offset === 0}
                  onClick={() => setOffset(Math.max(0, offset - PAGE))}
                >
                  Previous
                </Button>
                <span className="font-mono text-xs text-fg-muted tabular-nums">
                  {offset + 1}–{Math.min(offset + PAGE, candidates.data.total)} of{" "}
                  {candidates.data.total}
                </span>
                <Button
                  size="sm"
                  disabled={offset + PAGE >= candidates.data.total}
                  onClick={() => setOffset(offset + PAGE)}
                >
                  Next
                </Button>
              </div>
            )}
          </RailSection>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col overflow-hidden bg-canvas p-3">
          {focus === undefined ? (
            <Empty>Open a sample on the left to see its region of {label?.name ?? classKey}.</Empty>
          ) : focused.error ? (
            <ErrorBox>{focused.error.message}</ErrorBox>
          ) : !shown ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <div className="relative h-full min-h-0 overflow-hidden">
              <SampleStage
                image={shown}
                alt={`${focused.data?.group_key}/${focused.data?.external_id}`}
                view={view}
                onView={setView}
                layers={[{ key: "class", src: classMaskUrl(shown.id, classKey) }]}
                label="Reference canvas"
              />
            </div>
          )}
        </main>

        <aside
          data-scroll="rail"
          aria-label="References"
          className="flex w-72 shrink-0 flex-col overflow-y-auto overscroll-contain border-l border-line"
        >
          <RailSection title="References" hint={`${references.length} of ${MAX_REFERENCES}`}>
            {references.length === 0 ? (
              <p className="text-xs text-fg-muted">
                Tick a sample on the left to use it as a reference. One to ten, each showing the
                class.
              </p>
            ) : (
              <ul className="flex flex-col gap-1">
                {references.map((sampleId) => (
                  <ReferenceRow
                    key={sampleId}
                    datasetId={datasetId}
                    sampleId={sampleId}
                    active={sampleId === focus}
                    onOpen={() => update({ focus: String(sampleId) })}
                    onRemove={() => toggle(sampleId)}
                  />
                ))}
              </ul>
            )}
          </RailSection>

          <RailSection title="Run">
            <Select
              aria-label="Method"
              value={methodKey ?? ""}
              placeholder="Choose a method"
              options={methods.map((entry) => ({
                value: entry.key,
                label: entry.title,
                note: entry.key,
              }))}
              onValueChange={(value) => update({ method: value })}
            />
            <Select
              aria-label="Region profile"
              value={profileId === undefined ? "" : String(profileId)}
              placeholder="Choose a region profile"
              options={(profiles.data ?? []).map((profile) => ({
                value: String(profile.id),
                label: `${profile.name} · r${profile.revision_no}`,
                note: `${profile.prepared_width}×${profile.prepared_height}`,
              }))}
              onValueChange={(value) => update({ profile: value })}
            />
            <Button
              variant="primary"
              disabled={blocker !== null || freeze.isPending}
              onClick={() => {
                if (blocker !== null || !method || profileId === undefined) return;
                freeze.mutate(
                  {
                    datasetId,
                    classKey,
                    references,
                    regionProfileId: profileId,
                    methodKey: method.key,
                    methodTitle: method.title,
                  },
                  { onSuccess: (experiment) => void navigate(`/experiments/${experiment.id}`) },
                );
              }}
            >
              {freeze.isPending ? "Freezing…" : "Freeze as experiment"}
            </Button>
            {blocker !== null ? (
              <p className="text-xs text-fg-muted">{blocker}</p>
            ) : (
              <p className="text-xs text-fg-muted">
                Makes a split of these references, a run of {method?.title} on{" "}
                {label?.name ?? classKey}, and starts Train &amp; score. Every other sample is
                a query.
              </p>
            )}
            {freeze.error && <ErrorBox>{freeze.error.message}</ErrorBox>}
          </RailSection>
        </aside>
      </div>
    </div>
  );
}

function ReferenceRow({
  datasetId,
  sampleId,
  active,
  onOpen,
  onRemove,
}: {
  datasetId: number;
  sampleId: number;
  active: boolean;
  onOpen: () => void;
  onRemove: () => void;
}) {
  const sample = useSample(datasetId, sampleId);
  const name = sample.data ? `${sample.data.group_key}/${sample.data.external_id}` : `#${sampleId}`;
  return (
    <li className="flex items-center gap-1.5">
      <button
        type="button"
        onClick={onOpen}
        aria-current={active ? "true" : undefined}
        className={cn(
          "min-w-0 flex-1 truncate rounded-sm px-1 py-0.5 text-left font-mono text-xs transition-colors hover:text-signal",
          active ? "text-fg" : "text-fg-muted",
          focusRing,
        )}
        title={name}
      >
        {sample.data?.external_id ?? `#${sampleId}`}
      </button>
      <Button size="sm" variant="ghost" onClick={onRemove} aria-label={`Remove ${name}`}>
        Remove
      </Button>
    </li>
  );
}
