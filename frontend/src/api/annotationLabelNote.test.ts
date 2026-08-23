/**
 * The label/document disagreement rule.
 *
 * Worth asserting rather than eyeballing because the *silence* is the load-bearing half: a mark
 * that appears on the agreeing cases too is a mark nobody reads.
 */

import { describe, expect, it } from "vitest";

import { labelNote } from "./annotationLabelNote";
import type { AnnotationDocument, AnnotationShape } from "./client";

const REGION: AnnotationShape = {
  id: "shape-1",
  label_key: "defect",
  kind: "polygon",
  operation: "add",
  points: [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
  ],
};

function document(
  shapes: AnnotationShape[] = [],
  base: AnnotationDocument["base"] = "empty",
): AnnotationDocument {
  return { schema_version: 1, image_width: 64, image_height: 64, base, shapes };
}

describe("the label note", () => {
  it("says nothing when a normal part carries no regions", () => {
    expect(labelNote("normal", document())).toBeNull();
  });

  it("says nothing when a defect part carries a region", () => {
    expect(labelNote("defect", document([REGION]))).toBeNull();
  });

  it("names the consequence of drawing a defect on a part called normal", () => {
    const note = labelNote("normal", document([REGION]));
    expect(note).toContain("1 defect region is still drawn");
    expect(note).toContain("read as ground truth");
  });

  it("counts more than one region in the plural", () => {
    expect(labelNote("normal", document([REGION, { ...REGION, id: "shape-2" }]))).toContain(
      "2 defect regions are still drawn",
    );
  });

  it("names the imported mask when a normal part is based on one and adds nothing", () => {
    expect(labelNote("normal", document([], "source_mask"))).toContain("an imported mask is");
  });

  it("tells a defect part with nothing drawn that pixel metrics will skip it", () => {
    expect(labelNote("defect", document())).toContain("skip a defect image");
  });

  it("does not accuse a defect part whose only truth is its imported mask", () => {
    expect(labelNote("defect", document([], "source_mask"))).toBeNull();
  });

  it("tells an unlabeled part that its regions count for nothing yet", () => {
    expect(labelNote("unlabeled", document([REGION]))).toContain("excluded from every metric");
  });

  it("leaves an unlabeled part with nothing drawn alone", () => {
    // Nothing has been claimed either way, so there is no disagreement to report — the queue
    // screen is where "this is still unlabeled" belongs.
    expect(labelNote("unlabeled", document())).toBeNull();
  });

  it("ignores a subtract region, which asserts no defect", () => {
    expect(labelNote("normal", document([{ ...REGION, operation: "subtract" }]))).toBeNull();
  });
});
