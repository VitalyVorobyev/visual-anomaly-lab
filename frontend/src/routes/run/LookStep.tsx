/**
 * Step 2, the look: where the run looks, seen on one image at the size it will read.
 *
 * "Full frame" by default — every dataset has it, and a run prepares it at its own size — or
 * any region profile saved on Prepare. The stage is Prepare's own live preview
 * (`POST /api/datasets/{id}/region-preview`, `LiveStage`): the source with the crop drawn,
 * beside the prepared frame at the size the chosen method resolves to. ←/→ step through the
 * images Prepare's filmstrip spreads over the dataset. Adjust goes to Prepare, which comes
 * back here with the profile it saved.
 */

import { Badge, ButtonLink, Button, Skeleton } from "@vitavision/lab-ui";
import { ChevronLeft, ChevronRight, SlidersHorizontal } from "lucide-react";
import { useEffect, useState } from "react";

import type { RegionPreviewImage, RegionProfileRevision } from "../../api/client";
import { useHotkeys } from "../../hooks/useHotkeys";
import { useRegionLivePreview, useRegionPreviewImages } from "../../hooks/useRegionProfiles";
import { imageLabel } from "../prepare/Filmstrip";
import { LiveStage } from "../prepare/LiveStage";
import { ChoiceCard, StepIntro } from "./ChoiceCard";
import type { GuidedRun } from "./useGuidedRun";

export function LookStep({ run, datasetId }: { run: GuidedRun; datasetId: number }) {
  const profiles = run.profiles.data ?? [];
  const profile = run.profile;
  const images = useRegionPreviewImages(datasetId, profile?.sample_alignment ?? "per_image");
  const strip = images.data?.images ?? [];
  const [index, setIndex] = useState(0);
  useEffect(() => {
    if (index >= strip.length && strip.length > 0) setIndex(0);
  }, [index, strip.length]);
  const image: RegionPreviewImage | undefined = strip[index];

  const size = run.inputSize.data
    ? { width: run.inputSize.data.width, height: run.inputSize.data.height }
    : undefined;
  const preview = useRegionLivePreview(
    datasetId,
    profile && image && size
      ? { recipe: recipeOf(profile), imageId: image.image_id, size }
      : undefined,
  );

  const step = (delta: number) => {
    if (strip.length === 0) return;
    setIndex((current) => (current + delta + strip.length) % strip.length);
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

  const adjust = `/datasets/${datasetId}/prepare?${new URLSearchParams({
    ...(profile ? { profile: String(profile.id) } : {}),
    return: "run",
  }).toString()}`;

  return (
    <div className="flex flex-col gap-5">
      <StepIntro title="Where should it look?">
        The whole image unless a region profile says otherwise. The right pane is exactly what
        the method will read.
      </StepIntro>

      {run.profiles.isPending ? (
        <Skeleton className="h-16" />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {profiles.map((entry) => (
            <ChoiceCard
              key={entry.id}
              name="look"
              value={String(entry.id)}
              selected={entry.id === profile?.id}
              onSelect={() => run.update({ profileId: entry.id })}
              title={
                <>
                  {entry.name} <span className="font-mono text-xs text-fg-subtle">r{entry.revision_no}</span>
                </>
              }
              badges={
                entry.extractor_type === "identity" && entry.name === "Full frame" ? (
                  <Badge tone="info">default</Badge>
                ) : undefined
              }
              description={describeProfile(entry)}
            />
          ))}
        </div>
      )}

      <section aria-label="Preview" className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-xs text-fg-muted">
            {size ? (
              <>
                Read at <span className="font-mono text-fg">{run.size.caption}</span>
              </>
            ) : run.inputSize.error ? (
              run.inputSize.error.message
            ) : (
              run.readError ? "The input size waits for the method catalogue." : "Resolving the method's input size…"
            )}
          </p>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              icon={<ChevronLeft />}
              aria-label="Previous image"
              disabled={strip.length < 2}
              onClick={() => step(-1)}
            />
            <span className="font-mono text-xs text-fg-muted tabular-nums">
              {strip.length === 0 ? "—" : `${index + 1} / ${strip.length}`}
            </span>
            <Button
              size="sm"
              variant="ghost"
              icon={<ChevronRight />}
              aria-label="Next image"
              disabled={strip.length < 2}
              onClick={() => step(1)}
            />
            <ButtonLink to={adjust} size="sm" variant="secondary" icon={<SlidersHorizontal />}>
              Adjust on Prepare
            </ButtonLink>
          </div>
        </div>
        <LiveStage
          target={
            image
              ? { imageId: image.image_id, width: image.width, height: image.height, label: imageLabel(image) }
              : undefined
          }
          preview={preview.data}
          pending={preview.isFetching}
          error={preview.error ?? images.error}
          blocked={
            size === undefined
              ? "The prepared frame waits for the method's input size."
              : profile === undefined
                ? "This dataset has no region profile to preview."
                : null
          }
        />
      </section>
    </div>
  );
}

function recipeOf(profile: RegionProfileRevision) {
  return {
    extractor_type: profile.extractor_type,
    extractor_config: profile.extractor_config,
    padding_fraction: profile.padding_fraction,
    resample: profile.resample,
    sample_alignment: profile.sample_alignment,
  };
}

function describeProfile(profile: RegionProfileRevision): string {
  if (profile.extractor_type === "identity" && profile.name === "Full frame") {
    return "The whole image, resized to the run's input size. Nothing to prepare in advance.";
  }
  const pad = Math.round(profile.padding_fraction * 1000) / 10;
  const shared = profile.sample_alignment === "union" ? " · one crop shared by a sample's channels" : "";
  return `${profile.extractor_type} · pad ${pad}%${shared}`;
}
