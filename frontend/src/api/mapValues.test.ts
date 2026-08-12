import { describe, expect, it } from "vitest";

import {
  PlaneFormatError,
  decodePlane,
  fractionOf,
  valueAt,
  valuesAt,
} from "./mapValues";

/** The encoder's output, built here so the two ends are checked against each other. */
function encode(planes: number[][][], stride = 1): ArrayBuffer {
  const channels = planes.length;
  const height = planes[0]?.length ?? 0;
  const width = planes[0]?.[0]?.length ?? 0;

  const buffer = new ArrayBuffer(24 + channels * width * height * 4);
  const header = new DataView(buffer);
  header.setUint8(0, 0x56); // V
  header.setUint8(1, 0x41); // A
  header.setUint8(2, 0x4d); // M
  header.setUint8(3, 0x31); // 1
  header.setUint32(4, width, true);
  header.setUint32(8, height, true);
  header.setUint32(12, stride, true);
  header.setUint32(16, channels, true);

  const body = new Float32Array(buffer, 24);
  let index = 0;
  for (const plane of planes) {
    for (const row of plane) {
      for (const value of row) body[index++] = value;
    }
  }
  return buffer;
}

describe("decodePlane", () => {
  it("reads the dimensions and the values", () => {
    const plane = decodePlane(
      encode([
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ]),
    );

    expect(plane.width).toBe(3);
    expect(plane.height).toBe(2);
    expect(plane.channels).toBe(1);
    expect(plane.stride).toBe(1);
    expect([...plane.values]).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("takes the channel count from the header rather than assuming it", () => {
    /**
     * A mono experiment has one plane and an RGB one has three. Nothing in the UI may
     * encode which — channel count is data, so the header carries it and both are the
     * same code path here.
     */
    const plane = decodePlane(
      encode([
        [[1, 2]],
        [[3, 4]],
        [[5, 6]],
      ]),
    );

    expect(plane.channels).toBe(3);
    expect(valuesAt(plane, 0.0, 0.0)).toEqual([1, 3, 5]);
    expect(valuesAt(plane, 0.9, 0.0)).toEqual([2, 4, 6]);
  });

  it("refuses bytes that are not a value plane", () => {
    expect(() => decodePlane(new ArrayBuffer(4))).toThrow(PlaneFormatError);
    expect(() => decodePlane(new ArrayBuffer(64))).toThrow(PlaneFormatError);
  });

  it("refuses a body shorter than its header claims", () => {
    const buffer = encode([[[1, 2, 3]]]);
    expect(() => decodePlane(buffer.slice(0, 26))).toThrow(PlaneFormatError);
  });
});

describe("valueAt", () => {
  const plane = decodePlane(
    encode([
      [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
      ],
    ]),
  );

  it("indexes by fractions of the frame", () => {
    // No letterbox correction: preprocessing resizes without preserving aspect ratio and
    // every layer is `object-fill`, so the plane covers the frame edge to edge.
    expect(valueAt(plane, 0.0, 0.0)).toBe(0);
    expect(valueAt(plane, 0.99, 0.0)).toBe(3);
    expect(valueAt(plane, 0.0, 0.99)).toBe(10);
    expect(valueAt(plane, 0.99, 0.99)).toBe(13);
    expect(valueAt(plane, 0.5, 0.5)).toBe(12);
  });

  it("is null outside the image rather than clamping to an edge value", () => {
    // Clamping would report a number for a pointer that is over the frame's background,
    // which reads as a measurement of something that is not there.
    expect(valueAt(plane, -0.01, 0.5)).toBeNull();
    expect(valueAt(plane, 1, 0.5)).toBeNull();
    expect(valueAt(plane, 0.5, 1.5)).toBeNull();
  });

  it("is null for a channel the plane does not have", () => {
    expect(valueAt(plane, 0.5, 0.5, 1)).toBeNull();
  });

  it("has no measurement outside the prepared region", () => {
    const uncovered = decodePlane(encode([[[Number.NaN]]]));

    expect(valueAt(uncovered, 0.5, 0.5)).toBeNull();
    expect(valuesAt(uncovered, 0.5, 0.5)).toEqual([]);
  });
});

describe("fractionOf", () => {
  it("places a value in the run's range", () => {
    expect(fractionOf(0.5, { low: 0, high: 2 })).toBe(0.25);
    expect(fractionOf(1, { low: 1, high: 5 })).toBe(0);
  });

  it("has no answer for a degenerate or absent range", () => {
    // A run that recorded no range, or one flat map — a percentage would be invented.
    expect(fractionOf(1, null)).toBeNull();
    expect(fractionOf(1, { low: 2, high: 2 })).toBeNull();
  });
});
