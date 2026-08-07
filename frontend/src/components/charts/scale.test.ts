import { describe, expect, it } from "vitest";

import {
  extent,
  formatTick,
  histogram,
  linePath,
  linearScale,
  logScale,
  niceStep,
  padDomain,
} from "./scale";

describe("niceStep", () => {
  it("rounds up to a step a reader can hold in their head", () => {
    expect(niceStep(0.237)).toEqual(0.25);
    expect(niceStep(1.1)).toEqual(2);
    expect(niceStep(3)).toEqual(5);
    expect(niceStep(7)).toEqual(10);
  });

  it("does not divide by zero on a degenerate range", () => {
    expect(niceStep(0)).toEqual(1);
    expect(niceStep(Number.NaN)).toEqual(1);
  });
});

describe("padDomain", () => {
  it("widens a constant series so its line is still drawable", () => {
    const [low, high] = padDomain(5, 5);
    expect(high).toBeGreaterThan(low);
  });

  it("widens a series pinned at zero", () => {
    expect(padDomain(0, 0)).toEqual([-1, 1]);
  });

  it("leaves a real range alone", () => {
    expect(padDomain(0, 1)).toEqual([0, 1]);
  });

  it("falls back rather than producing NaN pixels", () => {
    expect(padDomain(Number.NaN, 1)).toEqual([0, 1]);
  });
});

describe("extent", () => {
  it("is a usable domain for an empty series", () => {
    expect(extent([])).toEqual([0, 1]);
  });

  it("ignores non-finite values rather than poisoning the range", () => {
    expect(extent([1, Number.NaN, 3, Number.POSITIVE_INFINITY])).toEqual([1, 3]);
  });
});

describe("linearScale", () => {
  it("maps the domain onto the pixel range, endpoint to endpoint", () => {
    const scale = linearScale([0, 10], 0, 100);
    expect(scale.project(0)).toEqual(0);
    expect(scale.project(10)).toEqual(100);
    expect(scale.project(5)).toEqual(50);
  });

  it("inverts happily, which is how a y axis is drawn", () => {
    const scale = linearScale([0, 1], 200, 0);
    expect(scale.project(0)).toEqual(200);
    expect(scale.project(1)).toEqual(0);
  });

  it("produces ticks inside the domain with readable labels", () => {
    const ticks = linearScale([0, 1], 0, 100).ticks(4);
    expect(ticks.length).toBeGreaterThan(2);
    for (const tick of ticks) {
      expect(tick.value).toBeGreaterThanOrEqual(0);
      expect(tick.value).toBeLessThanOrEqual(1);
    }
    // Re-derived from the step index, so no 0.30000000000000004 reaches a label.
    expect(ticks.map((tick) => tick.label)).not.toContain("0.30000000000000004");
  });

  it("still yields ticks for a constant series", () => {
    expect(logScale([1, 1], 0, 100).ticks().length).toBeGreaterThan(0);
    expect(linearScale([7, 7], 0, 100).ticks().length).toBeGreaterThan(0);
  });
});

describe("logScale", () => {
  it("spreads decades evenly, which is the point for loss curves", () => {
    const scale = logScale([1e-4, 1e-1], 0, 300);
    const decade = scale.project(1e-2) - scale.project(1e-3);
    expect(decade).toBeCloseTo(scale.project(1e-1) - scale.project(1e-2), 6);
  });

  it("clamps a zero rather than projecting it to negative infinity", () => {
    expect(Number.isFinite(logScale([1e-6, 1], 0, 100).project(0))).toBe(true);
  });
});

describe("formatTick", () => {
  it("uses an exponent where a decimal would be a row of zeroes", () => {
    expect(formatTick(0.000032, 0.00001)).toEqual("3.2e-5");
  });

  it("matches its precision to the step", () => {
    expect(formatTick(0.5, 0.25)).toEqual("0.50");
    expect(formatTick(20, 10)).toEqual("20");
  });
});

describe("linePath", () => {
  const x = linearScale([0, 10], 0, 100);
  const y = linearScale([0, 10], 0, 100);

  it("starts with a move and continues with lines", () => {
    const path = linePath(
      [
        { x: 0, y: 0 },
        { x: 10, y: 10 },
      ],
      x,
      y,
    );
    expect(path.startsWith("M")).toBe(true);
    expect(path).toContain("L");
  });

  it("is empty for an empty series rather than malformed", () => {
    expect(linePath([], x, y)).toEqual("");
  });

  it("lifts the pen over a gap instead of bridging it", () => {
    const path = linePath(
      [
        { x: 0, y: 0 },
        { x: 5, y: Number.NaN },
        { x: 10, y: 10 },
      ],
      x,
      y,
    );
    // Two moves, no line: the missing point is a gap, not a straight-line assertion.
    expect(path.match(/M/g)).toHaveLength(2);
    expect(path).not.toContain("L");
  });
});

describe("histogram", () => {
  it("bins over a shared domain so two classes can be compared", () => {
    expect(histogram([0, 0.4, 0.9], [0, 1], 2)).toEqual([2, 1]);
  });

  it("puts the domain maximum in the last bin rather than off the end", () => {
    expect(histogram([1], [0, 1], 4)).toEqual([0, 0, 0, 1]);
  });

  it("survives a degenerate domain", () => {
    expect(histogram([5, 5], [5, 5], 4)).toEqual([2, 0, 0, 0]);
  });

  it("counts nothing for an empty input", () => {
    expect(histogram([], [0, 1], 3)).toEqual([0, 0, 0]);
  });
});
