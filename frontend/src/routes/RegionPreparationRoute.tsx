/** Dataset-local spatial preparation: define, inspect, then materialise model input. */

import { useQueryClient } from "@tanstack/react-query";
import { Check, Copy, Eye, Play, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router";

import { imageUrl, preparedImageUrl } from "../api/imageUrl";
import { Badge, Button, Callout, describeFields, Empty, ErrorBox, Field, initialValues, Input, jsonErrors, NumberInput, Panel, ReadoutStrip, SchemaForm, SegmentedControl, Select, toOptions, type OptionsSchema, type RawValues } from "@vitavision/lab-ui";
import type {
  JobDetail,
  RegionPreparationEntry,
  RegionProfileRevision,
  SpatialResample,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { JobProgress } from "../components/JobProgress";
import { useDataset } from "../hooks/useCatalog";
import { TabScroll } from "./dataset/TabScroll";
import { isTerminal, useJob } from "../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../hooks/useModelAssets";
import {
  useCreateRegionProfile,
  useRegionBuild,
  useRegionExtractors,
  useRegionProfiles,
  useStartRegionBuild,
  useStartRegionPreview,
} from "../hooks/useRegionProfiles";

type PreviewResult = {
  mode: "preview";
  sampled: number;
  dataset_images: number;
  succeeded: number;
  failed: number;
  entries: RegionPreparationEntry[];
};

const RESAMPLE_OPTIONS = [
  { value: "nearest", label: "Nearest" },
  { value: "bilinear", label: "Bilinear" },
  { value: "bicubic", label: "Bicubic" },
  { value: "lanczos", label: "Lanczos" },
];

export function RegionPreparationRoute() {
  const datasetId = Number(useParams()["datasetId"]);
  const dataset = useDataset(datasetId);
  const profiles = useRegionProfiles(datasetId);
  const extractors = useRegionExtractors();
  const assets = useModelAssets();
  const create = useCreateRegionProfile(datasetId);
  const preview = useStartRegionPreview();
  const build = useStartRegionBuild();
  const installAsset = useInstallModelAsset();
  const queryClient = useQueryClient();

  const [selectedId, setSelectedId] = useState<number>();
  const [extractorKey, setExtractorKey] = useState("identity");
  const [name, setName] = useState("model input");
  const [width, setWidth] = useState("256");
  const [height, setHeight] = useState("256");
  const [padding, setPadding] = useState("0.05");
  const [resample, setResample] = useState<SpatialResample>("bilinear");
  const [seed, setSeed] = useState("17");
  const [configValues, setConfigValues] = useState<RawValues>({});
  const [jobId, setJobId] = useState<number>();
  const [jobProfileId, setJobProfileId] = useState<number>();
  const [jobMode, setJobMode] = useState<"preview" | "build" | "asset">();
  const [view, setView] = useState("source");

  const selected = profiles.data?.find((profile) => profile.id === selectedId);
  const extractor = extractors.data?.find((item) => item.key === extractorKey);
  const configFields = useMemo(
    () => describeFields((extractor?.config_schema ?? {}) as OptionsSchema),
    [extractor],
  );
  const job = useJob(jobId);
  const buildReport = useRegionBuild(selectedId);

  useEffect(() => {
    if (selectedId !== undefined || !profiles.data?.length) return;
    setSelectedId(profiles.data[0]?.id);
  }, [profiles.data, selectedId]);

  useEffect(() => {
    if (!job.job || !isTerminal(job.job.status)) return;
    if (jobMode === "build" && job.job.status === "succeeded" && jobProfileId !== undefined) {
      void queryClient.invalidateQueries({ queryKey: queryKeys.regionBuild(jobProfileId) });
      setView("prepared");
    }
    if (jobMode === "asset") {
      void queryClient.invalidateQueries({ queryKey: queryKeys.modelAssets() });
    }
  }, [job.job, jobMode, jobProfileId, queryClient]);

  const previewResult =
    jobMode === "preview" && jobProfileId === selectedId ? asPreviewResult(job.job) : null;
  const entries = previewResult?.entries ?? buildReport.data?.preview_entries ?? [];
  const requiredAssets = (extractor?.required_assets ?? []).map((key) => ({
    key,
    asset: assets.data?.assets.find((item) => item.key === key),
  }));
  const errors = [
    create.error,
    preview.error,
    build.error,
    installAsset.error,
  ].filter((error): error is Error => error instanceof Error);

  const chooseExtractor = (key: string) => {
    setExtractorKey(key);
    const next = extractors.data?.find((item) => item.key === key);
    const fields = describeFields((next?.config_schema ?? {}) as OptionsSchema);
    setConfigValues(initialValues(fields));
  };

  const createRevision = async () => {
    const profile = await create.mutateAsync({
      name: name.trim(),
      extractor_type: extractorKey,
      extractor_config: toOptions(configFields, configValues),
      prepared_width: Number(width),
      prepared_height: Number(height),
      padding_fraction: Number(padding),
      resample,
      seed: Number(seed),
    });
    setSelectedId(profile.id);
    setJobId(undefined);
    setJobMode(undefined);
    setJobProfileId(undefined);
  };

  const run = async (mode: "preview" | "build") => {
    if (!selected) return;
    const summary = await (mode === "preview"
      ? preview.mutateAsync(selected.id)
      : build.mutateAsync(selected.id));
    setJobMode(mode);
    setJobProfileId(selected.id);
    setJobId(summary.id);
  };

  const revise = (profile: RegionProfileRevision) => {
    setName(profile.name);
    setExtractorKey(profile.extractor_type);
    setWidth(String(profile.prepared_width));
    setHeight(String(profile.prepared_height));
    setPadding(String(profile.padding_fraction));
    setResample(profile.resample);
    setSeed(String(profile.seed));
    // The next render has the right extractor schema; preserve values as strings so the
    // schema layer can apply its own numeric/enum conversion on submit.
    setConfigValues(
      Object.fromEntries(
        Object.entries(profile.extractor_config).map(([key, value]) => [
          key,
          typeof value === "boolean"
            ? value
            : typeof value === "string"
              ? value
              : JSON.stringify(value),
        ]),
      ),
    );
  };

  const install = async (assetKey: string) => {
    const summary = await installAsset.mutateAsync(assetKey);
    setJobMode("asset");
    setJobProfileId(undefined);
    setJobId(summary.id);
  };

  const invalid =
    name.trim() === "" ||
    !extractor?.availability.available ||
    jsonErrors(configFields, configValues).length > 0 ||
    !validNumber(width, 8, 2048) ||
    !validNumber(height, 8, 2048) ||
    !validNumber(padding, 0, 1) ||
    !Number.isInteger(Number(seed));

  return (
    <TabScroll
      measure="wide"
      className="grid items-start gap-5 lg:grid-cols-[21rem_minmax(0,1fr)]"
    >
      {dataset.error && (
        <div className="lg:col-span-2">
          <ErrorBox>{dataset.error.message}</ErrorBox>
        </div>
      )}
          <aside className="flex flex-col gap-4 lg:sticky lg:top-5">
            <Panel
              title="Profile revision"
              actions={
                <span className="font-mono text-xs text-fg-subtle">
                  {profiles.data?.length ?? 0} revisions
                </span>
              }
              bodyClassName="flex flex-col gap-4"
            >
              <Field label="Saved profile">
                <Select
                  aria-label="Saved profile"
                  value={selectedId === undefined ? "" : String(selectedId)}
                  onValueChange={(value) => {
                    setSelectedId(Number(value));
                    setJobId(undefined);
                    setJobMode(undefined);
                    setJobProfileId(undefined);
                  }}
                  placeholder={profiles.isPending ? "Loading…" : "No revisions yet"}
                  options={(profiles.data ?? []).map((profile) => ({
                    value: String(profile.id),
                    label: profile.name,
                    note: `r${profile.revision_no} · ${profile.extractor_type}`,
                  }))}
                />
              </Field>
              {selected && (
                <div className="flex items-center justify-between gap-3 rounded-control bg-raised px-3 py-2">
                  <div className="min-w-0 text-xs text-fg-muted">
                    <span className="font-mono text-fg">{selected.prepared_width}×{selected.prepared_height}</span>
                    <span> · {selected.resample} · pad {selected.padding_fraction}</span>
                  </div>
                  <Button size="sm" variant="ghost" icon={<Copy />} onClick={() => revise(selected)}>
                    Revise
                  </Button>
                </div>
              )}

              <div className="h-px bg-line" />

              <Field label="Name">
                <Input value={name} onChange={(event) => setName(event.target.value)} />
              </Field>
              <Field label="Region extractor">
                <Select
                  aria-label="Region extractor"
                  value={extractorKey}
                  onValueChange={chooseExtractor}
                  options={(extractors.data ?? []).map((item) => ({
                    value: item.key,
                    label: item.title,
                    note: item.availability.available ? undefined : item.availability.reason ?? "unavailable",
                    disabled: !item.availability.available,
                  }))}
                />
              </Field>
              {extractor && <p className="-mt-2 text-xs leading-5 text-fg-muted">{extractor.summary}</p>}

              {configFields.length > 0 && (
                <SchemaForm fields={configFields} values={configValues} onChange={setConfigValues} />
              )}

              {requiredAssets.map(({ key, asset }) => (
                <div key={key} className="flex items-center gap-2 rounded-control border border-line bg-raised p-2.5">
                  <Sparkles className="size-4 shrink-0 text-signal" />
                  <span className="min-w-0 flex-1 truncate text-xs">{asset?.title ?? key}</span>
                  {asset?.status === "ready" ? (
                    <Badge tone="normal">ready</Badge>
                  ) : (
                    <Button size="sm" loading={installAsset.isPending} onClick={() => void install(key)}>
                      Install
                    </Button>
                  )}
                </div>
              ))}

              <div className="grid grid-cols-2 gap-3">
                <Field label="Width" annotation="8–2048">
                  <NumberInput min={8} max={2048} value={width} onChange={(event) => setWidth(event.target.value)} />
                </Field>
                <Field label="Height" annotation="8–2048">
                  <NumberInput min={8} max={2048} value={height} onChange={(event) => setHeight(event.target.value)} />
                </Field>
                <Field label="Padding" annotation="0–1">
                  <NumberInput min={0} max={1} step="any" value={padding} onChange={(event) => setPadding(event.target.value)} />
                </Field>
                <Field label="Seed">
                  <NumberInput step={1} value={seed} onChange={(event) => setSeed(event.target.value)} />
                </Field>
              </div>
              <Field label="Resampling">
                <Select aria-label="Resampling" value={resample} onValueChange={(value) => setResample(value as SpatialResample)} options={RESAMPLE_OPTIONS} />
              </Field>

              <Button variant="primary" icon={<Check />} disabled={invalid} loading={create.isPending} onClick={() => void createRevision()}>
                Save immutable revision
              </Button>
            </Panel>
          </aside>

          <main className="flex min-w-0 flex-col gap-4">
            {errors.map((error, index) => <ErrorBox key={index}>{error.message}</ErrorBox>)}

            <Panel
              title={selected ? `${selected.name} · revision ${selected.revision_no}` : "Inspect a profile"}
              actions={
                selected ? (
                  <>
                    <Button icon={<Eye />} disabled={job.job !== undefined && !isTerminal(job.job.status)} loading={preview.isPending} onClick={() => void run("preview")}>
                      Preview 24
                    </Button>
                    <Button variant="primary" icon={<Play />} disabled={job.job !== undefined && !isTerminal(job.job.status)} loading={build.isPending} onClick={() => void run("build")}>
                      Build all
                    </Button>
                  </>
                ) : undefined
              }
              bodyClassName="flex flex-col gap-4"
            >
              {!selected && <Empty>Save a profile revision to inspect its crop and prepared input.</Empty>}
              {selected && !jobId && !buildReport.data && (
                <Callout>
                  Preview samples evenly across the dataset before building. A failure remains a failure; this workflow never substitutes the full frame silently.
                </Callout>
              )}
              {jobId !== undefined && (
                <JobProgress jobId={jobId} job={job.job} lines={job.lines} error={job.error} />
              )}
            </Panel>

            {(previewResult || buildReport.data) && (
              <Panel
                title="Visual audit"
                actions={
                  <SegmentedControl
                    aria-label="Preview image mode"
                    value={view}
                    options={[
                      { value: "source", label: "Source + crop" },
                      { value: "prepared", label: "Prepared" },
                    ]}
                    onValueChange={setView}
                  />
                }
                bodyClassName="flex flex-col gap-4"
              >
                <ReadoutStrip
                  items={[
                    { label: "shown", value: entries.length },
                    { label: "succeeded", value: previewResult?.succeeded ?? buildReport.data?.succeeded },
                    { label: "failed", value: previewResult?.failed ?? buildReport.data?.failed },
                    buildReport.data ? { label: "stored", value: formatBytes(buildReport.data.storage_bytes) } : { value: null },
                  ]}
                />
                {view === "prepared" && !buildReport.data && (
                  <Callout>Build this revision before opening its materialised prepared pixels.</Callout>
                )}
                <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
                  {entries.map((entry) => (
                    <PreparationCard
                      key={entry.image_id}
                      entry={entry}
                      profileId={selected?.id}
                      prepared={view === "prepared" && buildReport.data !== undefined}
                    />
                  ))}
                </div>
              </Panel>
            )}
          </main>
    </TabScroll>
  );
}

function PreparationCard({
  entry,
  profileId,
  prepared,
}: {
  entry: RegionPreparationEntry;
  profileId: number | undefined;
  prepared: boolean;
}) {
  const transform = entry.transform;
  return (
    <article className="overflow-hidden rounded-panel border border-line bg-surface">
      <div className="relative aspect-[4/3] bg-raised">
        {entry.status === "succeeded" && transform ? (
          <>
            <img
              src={prepared && profileId !== undefined ? preparedImageUrl(profileId, entry.image_id) : imageUrl(entry.image_id, "preview")}
              alt=""
              loading="lazy"
              className="absolute inset-0 h-full w-full object-contain"
            />
            {!prepared && (
              <svg
                viewBox={`0 0 ${transform.source_width} ${transform.source_height}`}
                preserveAspectRatio="xMidYMid meet"
                className="pointer-events-none absolute inset-0 h-full w-full text-signal"
                aria-hidden
              >
                <rect
                  x={transform.crop_left}
                  y={transform.crop_top}
                  width={transform.crop_right - transform.crop_left}
                  height={transform.crop_bottom - transform.crop_top}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={Math.max(transform.source_width, transform.source_height) / 220}
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            )}
          </>
        ) : (
          <div className="grid h-full place-items-center px-4 text-center text-xs text-defect">{entry.error ?? "Preparation failed"}</div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 px-2.5 py-2">
        <span className="font-mono text-xs text-fg">image {entry.image_id}</span>
        <span className="flex items-center gap-1.5">
          {entry.extractor_confidence !== null && (
            <span className="font-mono text-[10px] text-fg-subtle">{entry.extractor_confidence.toFixed(2)}</span>
          )}
          <Badge tone={entry.status === "succeeded" ? "normal" : "defect"}>{entry.status === "succeeded" ? "ok" : "failed"}</Badge>
        </span>
      </div>
    </article>
  );
}

function asPreviewResult(job: JobDetail | undefined): PreviewResult | null {
  if (job?.status !== "succeeded" || job.result["mode"] !== "preview") return null;
  const result = job.result as Partial<PreviewResult>;
  return Array.isArray(result.entries) ? (result as PreviewResult) : null;
}

function validNumber(value: string, min: number, max: number): boolean {
  const number = Number(value);
  return Number.isFinite(number) && number >= min && number <= max;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / 1024 ** 2).toFixed(1)} MiB`;
}
