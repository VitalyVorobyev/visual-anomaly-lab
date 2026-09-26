/**
 * The images the live stage steps through: spread evenly over the dataset, never its first
 * folder, and over every channel (`preview_selection`, the same rule the sampled check uses).
 *
 * It scrolls sideways on its own; the page is the only thing that scrolls vertically.
 */

import { cn, focusRing } from "@vitavision/lab-ui";

import type { RegionPreviewImage } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";

export function imageLabel(image: RegionPreviewImage): string {
  const name = `${image.group_key}/${image.external_id}`;
  return image.channel ? `${name} · ${image.channel}` : name;
}

export function Filmstrip({
  images,
  activeImageId,
  onSelect,
}: {
  images: RegionPreviewImage[];
  activeImageId: number | undefined;
  onSelect: (image: RegionPreviewImage) => void;
}) {
  return (
    <div role="list" aria-label="Preview images" className="flex gap-2 overflow-x-auto pb-1">
      {images.map((image) => {
        const active = image.image_id === activeImageId;
        return (
          <div role="listitem" key={image.image_id} className="shrink-0">
            <button
              type="button"
              onClick={() => onSelect(image)}
              aria-label={imageLabel(image)}
              aria-current={active ? "true" : undefined}
              title={imageLabel(image)}
              className={cn(
                "block h-14 w-20 overflow-hidden rounded-control border-2 bg-raised transition-colors hover:border-signal",
                active ? "border-signal" : "border-transparent",
                focusRing,
              )}
            >
              <img
                src={imageUrl(image.image_id, "thumb")}
                alt=""
                loading="lazy"
                draggable={false}
                className="h-full w-full object-cover"
              />
            </button>
          </div>
        );
      })}
    </div>
  );
}
