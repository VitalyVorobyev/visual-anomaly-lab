/**
 * A part is one job, or each photograph is — and the queue has to say which.
 *
 * This is the rule that makes sample-scoped annotation usable rather than merely correct:
 * the backend would happily accept the same completion three times, but a reader working
 * through 300 parts would be shown 900 cards and would rightly conclude the feature does
 * nothing.
 */

import { describe, expect, it } from "vitest";

import { queueUnits } from "./annotationQueue";
import type { ImageSummary, SampleSummary } from "./client";

function image(id: number, channel: string | null): ImageSummary {
  return {
    id,
    channel,
    channel_id: channel ? id : null,
    width: 1280,
    height: 1024,
    bit_depth: 24,
    file_size: 3_932_214,
    path: `/somewhere/${id}.bmp`,
  };
}

function sample(id: number, images: ImageSummary[]): SampleSummary {
  return {
    id,
    dataset_id: 7,
    group_key: "set1/defect",
    external_id: String(id),
    label: "defect",
    label_source: "import",
    notes: null,
    images,
    annotation: "none",
  };
}

const THREE = sample(1, [image(11, "bright"), image(12, "dark"), image(13, "dome")]);
const TWO = sample(2, [image(21, "bright"), image(22, "dark")]);

describe("what one unit of annotation work is", () => {
  it("lists every photograph under image scope", () => {
    const units = queueUnits([THREE, TWO], false);

    expect(units.map((unit) => unit.image.id)).toEqual([11, 12, 13, 21, 22]);
  });

  it("collapses a part's channels to one entry under sample scope", () => {
    const units = queueUnits([THREE, TWO], true);

    expect(units.map((unit) => unit.sample.id)).toEqual([1, 2]);
    // The first channel in display order is what the card previews and the editor opens on.
    expect(units.map((unit) => unit.image.id)).toEqual([11, 21]);
  });

  it("counts a two-channel part exactly like a three-channel one", () => {
    // Channel count is data, never schema (ADR-0005): a short capture group is one job
    // under sample scope for the same reason a full one is, with no padding and no case.
    expect(queueUnits([TWO], true)).toHaveLength(1);
    expect(queueUnits([THREE], true)).toHaveLength(1);
  });

  it("drops a sample with no images rather than offering an empty job", () => {
    expect(queueUnits([sample(3, [])], true)).toEqual([]);
    expect(queueUnits([sample(3, [])], false)).toEqual([]);
  });
});
