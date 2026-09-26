/**
 * Dataset-local spatial preparation, preview first: tune where to look on one image at a
 * time, check it on a sample, and save it only once it is right.
 *
 * Every control change re-runs a synchronous preview of the *unsaved* configuration on the
 * image on the stage, debounced, so tuning writes nothing and waits behind no job. "Check
 * 24" is the sampled job on the same unsaved configuration. "Save profile" appears only
 * when the form differs from the saved revision it started from.
 *
 * A profile carries no size — the size is a run's. The preview size here, 448 × 448 unless
 * changed, is the frame every DINO method and most measured gates read; a run whose size
 * has no build prepares it in its own job, so "Build all" only saves that run the wait.
 */

import { useQueryClient } from "@tanstack/react-query";
import { Check, ChevronLeft, ChevronRight, Eye, Play, Shuffle, Sparkles, Trash2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router";

import {
  Badge,
  Button,
  Callout,
  ConfirmDialog,
  describeFields,
  Empty,
  ErrorBox,
  Field,
  initialValues,
  Input,
  jsonErrors,
  NumberInput,
  outOfRange,
  Panel,
  ReadoutStrip,
  SchemaForm,
  SegmentedControl,
  Select,
  type OptionsSchema,
  type RawValues,
} from "@vitavision/lab-ui";

import type {
  JobDetail,
  RegionPreparationEntry,
  RegionPreviewImage,
  RegionProfileRevision,
  SampleAlignment,
  SpatialResample,
} from "../api/client";
import { formatBytes } from "../api/format";
import { readPrepareState, writePrepareState, type PrepareState } from "../api/prepareState";
import { queryKeys } from "../api/queryKeys";
import { JobProgress } from "../components/JobProgress";
import { useDataset } from "../hooks/useCatalog";
import { useDebounced } from "../hooks/useDebounced";
import { useHotkeys } from "../hooks/useHotkeys";
import { isTerminal, useJob } from "../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../hooks/useModelAssets";
import {
  useCreateRegionProfile,
  useDeleteRegionProfile,
  useRandomPreviewImage,
  useRegionBuilds,
  useRegionExtractors,
  useRegionLivePreview,
  useRegionPreviewImages,
  useRegionProfileDeletionPreview,
  useRegionProfiles,
  useStartRegionBuild,
  useStartRegionCheck,
  type LivePreviewRequest,
} from "../hooks/useRegionProfiles";
import { TabScroll } from "./dataset/TabScroll";
import { CheckGrid } from "./prepare/CheckGrid";
import { Filmstrip, imageLabel } from "./prepare/Filmstrip";
import { LiveStage, type StageTarget } from "./prepare/LiveStage";
import { formValuesOf, recipeDiffers, recipeOf, suggestedName, type SavedRecipe } from "./prepare/recipe";

/** The size the preview, the check and "Build all" prepare at until another is typed. */
export const DEFAULT_PREPARE_SIZE = 448;

/** How long the form must be still before the live preview is asked again. */
export const LIVE_DEBOUNCE_MS = 250;

type CheckResult = {
  mode: "preview";
  width?: number;
  height?: number;
  sampled: number;
  dataset_images: number;
  succeeded: number;
  failed: number;
  entries: RegionPreparationEntry[];
  recipe?: SavedRecipe;
};

const RESAMPLE_OPTIONS = [
  { value: "nearest", label: "Nearest" },
  { value: "bilinear", label: "Bilinear" },
  { value: "bicubic", label: "Bicubic" },
  { value: "lanczos", label: "Lanczos" },
];

const ALIGNMENT_OPTIONS = [
  { value: "per_image", label: "Per image" },
  { value: "union", label: "Shared" },
];

export function RegionPreparationRoute() {
  const datasetId = Number(useParams()["datasetId"]);
  const dataset = useDataset(datasetId);
  const profiles = useRegionProfiles(datasetId);
  const extractors = useRegionExtractors();
  const assets = useModelAssets();
  const create = useCreateRegionProfile(datasetId);
  const check = useStartRegionCheck(datasetId);
  const build = useStartRegionBuild();
  const installAsset = useInstallModelAsset();
  const remove = useDeleteRegionProfile(datasetId);
  const random = useRandomPreviewImage(datasetId);
  const queryClient = useQueryClient();

  // Which revision is open and which job is being followed live in the URL, so a reload
  // or a trip to another tab mid-build comes back to the same console.
  const [params, setParams] = useSearchParams();
  const { profile: selectedId, job: jobId, jobProfile: jobProfileId, mode: jobMode } =
    readPrepareState(params);
  const updatePrep = (next: PrepareState) => setParams(writePrepareState(next), { replace: true });

  const [extractorKey, setExtractorKey] = useState("identity");
  const [name, setName] = useState("");
  const [width, setWidth] = useState(String(DEFAULT_PREPARE_SIZE));
  const [height, setHeight] = useState(String(DEFAULT_PREPARE_SIZE));
  const [padding, setPadding] = useState("0.05");
  const [resample, setResample] = useState<SpatialResample>("bilinear");
  const [alignment, setAlignment] = useState<SampleAlignment>("per_image");
  const [configValues, setConfigValues] = useState<RawValues>({});
  const [target, setTarget] = useState<StageTarget | undefined>(undefined);
  const [pendingDelete, setPendingDelete] = useState<RegionProfileRevision | null>(null);
  const deletionPreview = useRegionProfileDeletionPreview(pendingDelete?.id);

  const selected = profiles.data?.find((profile) => profile.id === selectedId);
  const extractor = extractors.data?.find((item) => item.key === extractorKey);
  const configFields = useMemo(
    () => describeFields((extractor?.config_schema ?? {}) as OptionsSchema),
    [extractor],
  );
  const job = useJob(jobId);
  const sizeValid = validNumber(width, 8, 2048) && validNumber(height, 8, 2048);
  const size = useMemo(
    () => (sizeValid ? { width: Number(width), height: Number(height) } : undefined),
    [sizeValid, width, height],
  );

  const builds = useRegionBuilds(selectedId);
  const strip = useRegionPreviewImages(datasetId, alignment);
  const stripImages = useMemo(() => strip.data?.images ?? [], [strip.data]);

  // The form always starts from a saved revision: the one open, else the newest.
  useEffect(() => {
    if (selectedId !== undefined || !profiles.data?.length) return;
    updatePrep({ ...readPrepareState(params), profile: profiles.data[0]?.id });
    // Keyed on the list arriving, not on `params`: once a revision is chosen this is done.
  }, [profiles.data, selectedId]);

  const load = useCallback((profile: RegionProfileRevision) => {
    setName("");
    setExtractorKey(profile.extractor_type);
    setPadding(String(profile.padding_fraction));
    setResample(profile.resample);
    setAlignment(profile.sample_alignment);
    setConfigValues(formValuesOf(profile));
  }, []);

  useEffect(() => {
    if (selected) load(selected);
    // Opening a revision loads it; a refetch of the same revision must not undo edits.
  }, [selected?.id, load]);

  // The first image of the strip goes on the stage until the reader picks another.
  useEffect(() => {
    if (target !== undefined || stripImages.length === 0) return;
    setTarget(stageTargetOf(stripImages[0] as RegionPreviewImage));
  }, [stripImages, target]);

  useEffect(() => {
    if (!job.job || !isTerminal(job.job.status)) return;
    if (jobMode === "build" && job.job.status === "succeeded" && jobProfileId !== undefined) {
      // Every size of that profile: the report at the size built, and the list of builds.
      void queryClient.invalidateQueries({ queryKey: ["region-profiles", jobProfileId] });
    }
    if (jobMode === "asset") {
      void queryClient.invalidateQueries({ queryKey: queryKeys.modelAssets() });
    }
  }, [job.job, jobMode, jobProfileId, queryClient]);

  const recipe = recipeOf({ extractorKey, configFields, configValues, padding, resample, alignment });
  // Nothing is "edited" until the saved revisions are known and the form has loaded one.
  const loaded = profiles.data !== undefined && (selected !== undefined || profiles.data.length === 0);
  const dirty = loaded && recipeDiffers(recipe, configFields, selected);

  const outOfBounds = outOfRange(configFields, configValues);
  const blocked =
    extractor === undefined
      ? extractors.error
        ? `The extractor catalogue could not be read: ${extractors.error.message}`
        : "Loading the extractor…"
      : !extractor.availability.available
        ? (extractor.availability.reason ?? "This extractor is not available.")
        : jsonErrors(configFields, configValues).length > 0 || outOfBounds.length > 0
          ? "Fix the options outside their range to preview."
          : !validNumber(padding, 0, 1)
            ? "Padding must be between 0 and 1."
            : size === undefined
              ? "The preview size must be between 8 and 2048."
              : null;

  // Debounce the request, not the controls: typing stays live, the backend sees settled values.
  const requestKey =
    blocked === null && target !== undefined && size !== undefined
      ? JSON.stringify({ recipe, imageId: target.imageId, size })
      : "";
  const settledKey = useDebounced(requestKey, LIVE_DEBOUNCE_MS);
  const liveRequest = settledKey === "" ? undefined : (JSON.parse(settledKey) as LivePreviewRequest);
  const live = useRegionLivePreview(datasetId, liveRequest);
  const livePending = requestKey !== "" && (requestKey !== settledKey || live.isFetching);

  const index = target === undefined ? -1 : stripImages.findIndex((image) => image.image_id === target.imageId);
  const step = (delta: number) => {
    if (stripImages.length === 0) return;
    const next = index === -1 ? 0 : (index + delta + stripImages.length) % stripImages.length;
    setTarget(stageTargetOf(stripImages[next] as RegionPreviewImage));
  };
  useHotkeys((event) => {
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      step(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      step(1);
    }
  });

  const pickRandom = async () => {
    const image = await random.mutateAsync();
    setTarget(stageTargetOf(image));
  };

  const chooseExtractor = (key: string) => {
    setExtractorKey(key);
    const next = extractors.data?.find((item) => item.key === key);
    const fields = describeFields((next?.config_schema ?? {}) as OptionsSchema);
    setConfigValues(initialValues(fields));
  };

  const suggested = suggestedName(recipe, extractor?.title ?? extractorKey);
  const effectiveName = name.trim() || suggested;

  const save = async () => {
    const profile = await create.mutateAsync({ ...recipe, name: effectiveName });
    updatePrep({ profile: profile.id });
  };

  const startCheck = async () => {
    if (size === undefined) return;
    const summary = await check.mutateAsync({ recipe, size });
    updatePrep({ profile: selectedId, job: summary.id, mode: "preview" });
  };

  const startBuild = async () => {
    if (!selected || size === undefined) return;
    const summary = await build.mutateAsync({ profileId: selected.id, size });
    updatePrep({ profile: selected.id, job: summary.id, jobProfile: selected.id, mode: "build" });
  };

  const install = async (assetKey: string) => {
    const summary = await installAsset.mutateAsync(assetKey);
    updatePrep({ profile: selectedId, job: summary.id, mode: "asset" });
  };

  const requiredAssets = (extractor?.required_assets ?? []).map((key) => ({
    key,
    asset: assets.data?.assets.find((item) => item.key === key),
  }));
  const errors = [create.error, check.error, build.error, installAsset.error, remove.error, random.error].filter(
    (error): error is Error => error instanceof Error,
  );

  const running = job.job !== undefined && !isTerminal(job.job.status);
  const checkResult = jobMode === "preview" ? asCheckResult(job.job) : null;
  const checkIsStale =
    checkResult?.recipe !== undefined && recipeDiffers(recipe, configFields, checkResult.recipe);
  /*
   * A check that succeeded and reported nothing to draw. Without this the screen answers
   * a press of Check 24 with no grid and no message beside a green badge — how a `done`
   * frame truncated in transport once went undiagnosed.
   */
  const checkReportedNothing = jobMode === "preview" && job.job?.status === "succeeded" && checkResult === null;
  const sizeLabel = size === undefined ? "—" : `${size.width}×${size.height}`;
  // From the list of builds rather than a per-size read, whose 404 means only "not built".
  const builtHere = builds.data?.find((entry) => entry.width === size?.width && entry.height === size?.height);
  const alreadyBuilt = builtHere !== undefined;
  const buildBlocker = !loaded
    ? null
    : selected === undefined
      ? "Save the profile to build it."
      : dirty
        ? "Build all prepares the saved revision; save these changes first."
        : alreadyBuilt
          ? `This size is built.`
          : null;

  return (
    <TabScroll measure="wide" className="grid items-start gap-5 lg:grid-cols-[22rem_minmax(0,1fr)]">
      {dataset.error && (
        <div className="lg:col-span-2">
          <ErrorBox>{dataset.error.message}</ErrorBox>
        </div>
      )}
      <aside className="flex flex-col gap-4 lg:sticky lg:top-5">
        <Panel
          title="Region profile"
          actions={
            <span className="font-mono text-xs text-fg-subtle">{profiles.data ? `${profiles.data.length} saved` : "…"}</span>
          }
          bodyClassName="flex flex-col gap-4"
        >
          <div className="flex items-end gap-1">
            <Field label="Saved profile" className="min-w-0 flex-1">
              <Select
                aria-label="Saved profile"
                value={selectedId === undefined ? "" : String(selectedId)}
                onValueChange={(value) => updatePrep({ profile: Number(value) })}
                placeholder={
                  profiles.error ? "Could not be read" : profiles.isPending ? "Loading…" : "No revisions yet"
                }
                options={(profiles.data ?? []).map((profile) => ({
                  value: String(profile.id),
                  label: profile.name,
                  note: `r${profile.revision_no} · ${profile.extractor_type}${profile.sample_alignment === "union" ? " · shared crop" : ""}`,
                }))}
              />
            </Field>
            {selected && (
              <Button
                variant="ghost"
                aria-label={`Delete ${selected.name} revision ${selected.revision_no}`}
                icon={<Trash2 />}
                onClick={() => setPendingDelete(selected)}
              />
            )}
          </div>
          {selected && (
            <p className="-mt-2 text-xs text-fg-muted">
              {dirty ? (
                <span className="text-warn">Edited · not saved</span>
              ) : (
                <span>
                  Showing r{selected.revision_no} as saved
                  {selected.sample_alignment === "union" && " · shared crop"}
                </span>
              )}
            </p>
          )}

          <div className="h-px bg-line" />

          <Field label="Region extractor">
            <Select
              aria-label="Region extractor"
              placeholder={extractors.error ? "Could not be read" : "Loading…"}
              value={extractors.data ? extractorKey : ""}
              onValueChange={chooseExtractor}
              options={(extractors.data ?? []).map((item) => ({
                value: item.key,
                label: item.title,
                note: item.availability.available ? undefined : (item.availability.reason ?? "unavailable"),
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

          <Field label="Padding" annotation="0–1">
            <NumberInput
              aria-label="Padding"
              min={0}
              max={1}
              step="any"
              value={padding}
              onChange={(event) => setPadding(event.target.value)}
            />
          </Field>
          <Field label="Resampling">
            <Select
              aria-label="Resampling"
              value={resample}
              onValueChange={(value) => setResample(value as SpatialResample)}
              options={RESAMPLE_OPTIONS}
            />
          </Field>
          <Field label="Crop per sample">
            <SegmentedControl
              aria-label="Crop per sample"
              value={alignment}
              options={ALIGNMENT_OPTIONS}
              onValueChange={(value) => setAlignment(value as SampleAlignment)}
            />
          </Field>
          <p className="-mt-2 text-xs leading-5 text-fg-muted">
            Shared gives every channel of a part the union of their crops, so they stay registered; its
            images must share one size.
          </p>

          <Field label="Preview size" annotation="8–2048 px">
            <div className="flex items-center gap-2">
              <NumberInput
                aria-label="Preview width"
                min={8}
                max={2048}
                value={width}
                onChange={(event) => setWidth(event.target.value)}
              />
              <span aria-hidden className="text-fg-subtle">
                ×
              </span>
              <NumberInput
                aria-label="Preview height"
                min={8}
                max={2048}
                value={height}
                onChange={(event) => setHeight(event.target.value)}
              />
            </div>
          </Field>
          <p className="-mt-2 text-xs leading-5 text-fg-muted">
            The preview, Check 24 and Build all prepare at {sizeLabel}. A profile has no size of its own: a
            run prepares it at the run&apos;s size when it trains.
          </p>
          {(builds.data?.length ?? 0) > 0 && (
            <div className="-mt-2 flex flex-wrap items-center gap-1.5 text-xs text-fg-muted">
              <span>Built at</span>
              {builds.data?.map((entry) => (
                <Button
                  key={`${entry.width}x${entry.height}`}
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setWidth(String(entry.width));
                    setHeight(String(entry.height));
                  }}
                >
                  {entry.width}×{entry.height}
                </Button>
              ))}
            </div>
          )}

          {/* Said out loud rather than left to a greyed-out button. */}
          {outOfBounds.length > 0 && (
            <p className="text-xs text-defect">Outside the allowed range: {outOfBounds.join(", ")}.</p>
          )}

          {dirty && (
            <>
              <div className="h-px bg-line" />
              <Field label="Save as">
                <Input
                  aria-label="Profile name"
                  value={name}
                  placeholder={suggested}
                  onChange={(event) => setName(event.target.value)}
                />
              </Field>
              <Button
                variant="primary"
                icon={<Check />}
                disabled={blocked !== null}
                loading={create.isPending}
                onClick={() => void save()}
              >
                Save profile
              </Button>
              {blocked !== null && <p className="-mt-2 text-xs text-fg-muted">{blocked}</p>}
            </>
          )}
        </Panel>
      </aside>

      <main className="flex min-w-0 flex-col gap-4">
        {errors.map((error, errorIndex) => (
          <ErrorBox key={errorIndex}>{error.message}</ErrorBox>
        ))}

        <Panel
          title="Live preview"
          actions={
            <>
              <span className="hidden font-mono text-xs text-fg-subtle sm:inline">
                {index === -1 ? (target ? "random" : "—") : `${index + 1} / ${stripImages.length}`}
                {strip.data && ` of ${strip.data.total}`}
              </span>
              <Button
                size="sm"
                variant="ghost"
                aria-label="Previous image"
                icon={<ChevronLeft />}
                disabled={stripImages.length === 0}
                onClick={() => step(-1)}
              />
              <Button
                size="sm"
                variant="ghost"
                aria-label="Next image"
                icon={<ChevronRight />}
                disabled={stripImages.length === 0}
                onClick={() => step(1)}
              />
              <Button
                size="sm"
                icon={<Shuffle />}
                disabled={!strip.data?.total}
                loading={random.isPending}
                onClick={() => void pickRandom()}
              >
                Random
              </Button>
            </>
          }
          bodyClassName="flex flex-col gap-3"
        >
          {strip.error ? (
            <ErrorBox>{strip.error.message}</ErrorBox>
          ) : strip.data && strip.data.total === 0 ? (
            <Empty>This dataset has no images to prepare.</Empty>
          ) : (
            <>
              <LiveStage
                target={target}
                preview={live.data}
                pending={livePending}
                error={live.error}
                blocked={blocked}
              />
              <Filmstrip
                images={stripImages}
                activeImageId={target?.imageId}
                onSelect={(image) => setTarget(stageTargetOf(image))}
              />
              {strip.data && (
                <p className="text-xs text-fg-muted">
                  ← → step through {stripImages.length} images spread evenly over the dataset. The dashed box
                  is where the extractor looked; the solid one is the crop after padding.
                </p>
              )}
            </>
          )}
        </Panel>

        <Panel
          title={`Check 24 · ${sizeLabel}`}
          actions={
            <>
              <Button
                icon={<Eye />}
                disabled={running || blocked !== null}
                loading={check.isPending}
                onClick={() => void startCheck()}
              >
                Check 24
              </Button>
              <Button
                variant="primary"
                icon={<Play />}
                disabled={running || size === undefined || !selected || buildBlocker !== null}
                loading={build.isPending}
                onClick={() => void startBuild()}
              >
                Build all
              </Button>
            </>
          }
          bodyClassName="flex flex-col gap-4"
        >
          <p className="text-xs leading-5 text-fg-muted">
            Check 24 runs this configuration — saved or not — on 24 images spread over the dataset, as a job,
            and lists the failures first. Build all prepares every image of the saved revision ahead of a run.
            {buildBlocker !== null && <span className="text-fg"> {buildBlocker}</span>}
          </p>
          {jobId !== undefined && <JobProgress jobId={jobId} job={job.job} lines={job.lines} error={job.error} />}
          {checkReportedNothing && (
            <Callout tone="warning">
              This check finished but returned no per-image entries, so there is nothing to audit. The job log
              above is the record of what it actually did.
            </Callout>
          )}
          {checkResult && (
            <>
              {checkIsStale && (
                <Callout tone="warning">The configuration has changed since this check ran.</Callout>
              )}
              <ReadoutStrip
                items={[
                  { label: "checked", value: checkResult.entries.length },
                  { label: "succeeded", value: checkResult.succeeded },
                  { label: "failed", value: checkResult.failed },
                  { label: "of", value: checkResult.dataset_images },
                  builtHere
                    ? { label: "built", value: formatBytes(builtHere.storage_bytes) }
                    : { value: null },
                ]}
              />
              <CheckGrid
                entries={checkResult.entries}
                activeImageId={target?.imageId}
                onOpen={(entry) =>
                  setTarget({
                    imageId: entry.image_id,
                    width: entry.transform?.source_width,
                    height: entry.transform?.source_height,
                    label: `image ${entry.image_id}`,
                  })
                }
              />
            </>
          )}
        </Panel>
      </main>

      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        title="Delete this profile revision?"
        description={
          pendingDelete && (
            <>
              <span className="font-medium text-fg">
                {pendingDelete.name} · revision {pendingDelete.revision_no}
              </span>{" "}
              and any prepared pixels it materialised will be removed. Source dataset files are never touched.
              {deletionPreview.isPending && (
                <span className="mt-3 block text-fg-subtle">Inspecting prepared files…</span>
              )}
              {deletionPreview.error && (
                <span className="mt-3 block text-defect">{deletionPreview.error.message}</span>
              )}
              {deletionPreview.data && (
                <span className="mt-3 block rounded-control border border-line bg-raised px-3 py-2">
                  <span className="block font-mono text-xs text-fg">
                    {deletionPreview.data.generated_files} prepared files ·{" "}
                    {formatBytes(deletionPreview.data.generated_bytes)}
                  </span>
                  {/* A refusal names what holds the profile, because "delete the
                      experiments first" is only actionable if you know which ones. */}
                  {deletionPreview.data.blocker && (
                    <span className="mt-1 block text-xs text-warn">{deletionPreview.data.blocker}</span>
                  )}
                </span>
              )}
            </>
          )
        }
        confirmLabel="Delete revision"
        destructive
        loading={remove.isPending}
        disabled={!deletionPreview.data?.can_delete}
        onConfirm={() => {
          if (pendingDelete === null || !deletionPreview.data?.can_delete) return;
          const deletedId = pendingDelete.id;
          remove.mutate(deletedId, {
            onSuccess: () => {
              // The selection and anything keyed to it described a revision that is gone.
              if (selectedId !== deletedId) return;
              updatePrep({});
            },
            onSettled: () => setPendingDelete(null),
          });
        }}
      />
    </TabScroll>
  );
}

function stageTargetOf(image: RegionPreviewImage): StageTarget {
  return { imageId: image.image_id, width: image.width, height: image.height, label: imageLabel(image) };
}

function asCheckResult(job: JobDetail | undefined): CheckResult | null {
  if (job?.status !== "succeeded" || job.result["mode"] !== "preview") return null;
  const result = job.result as Partial<CheckResult>;
  return Array.isArray(result.entries) ? (result as CheckResult) : null;
}

function validNumber(value: string, min: number, max: number): boolean {
  const number = Number(value);
  return value.trim() !== "" && Number.isFinite(number) && number >= min && number <= max;
}
