/**
 * The arithmetic behind every chart: domains, ticks, and the map from a value to a pixel.
 *
 * Hand-rolled SVG rather than a charting library, as the M4 plan records. The chart types
 * this workbench needs are few and simple, and a library that lags the React and Tailwind
 * versions the frontend is deliberately on would mean either downgrading the project or
 * isolating a laggard — a cost that outweighs a few hundred lines of scale maths.
 *
 * Everything here is pure. That is the point: it is the part worth unit-testing, and it
 * leaves the components themselves as thin as an SVG element list.
 */

/** A tick's position in domain units, with the label to draw beside it. */
export interface Tick {
  value: number;
  label: string;
}

export interface Scale {
  /** Domain, after padding and after any log transform's guard. */
  domain: [number, number];
  /** Map one domain value onto the pixel range this scale was built for. */
  project: (value: number) => number;
  ticks: (count?: number) => Tick[];
}

/** Smallest value a log scale will accept, so a zero loss cannot become `-Infinity`. */
const LOG_FLOOR = 1e-12;

const TICK_STEPS = [1, 2, 2.5, 5, 10];

/** Past this a label is noise; anything finer is shown as an exponent instead. */
const MAX_DECIMALS = 6;

/**
 * A "nice" step at or above `rough` — one of 1, 2, 2.5 or 5 times a power of ten.
 *
 * Axis labels people can read at a glance are the whole reason: 0.25 / 0.50 / 0.75 rather
 * than 0.237 / 0.474 / 0.711.
 */
export function niceStep(rough: number): number {
  if (!Number.isFinite(rough) || rough <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalized = rough / magnitude;
  const step = TICK_STEPS.find((candidate) => candidate >= normalized - 1e-9) ?? 10;
  return step * magnitude;
}

/**
 * How many decimals a label needs so that neighbouring ticks do not print identically.
 *
 * Derived from the step rather than fixed, because one chart here is a loss around 1e-4
 * and another is a rate between 0 and 1. Enough decimals to print the *step* exactly is
 * the right rule, and `ceil(-log10(step))` is not it: a 0.25 step gets one decimal that
 * way, and the axis reads 0.3 / 0.5 / 0.8 — three labels for four evenly spaced ticks.
 */
function decimalsFor(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0;
  for (let decimals = 0; decimals < MAX_DECIMALS; decimals += 1) {
    if (Math.abs(Number(step.toFixed(decimals)) - step) < step * 1e-6) return decimals;
  }
  return MAX_DECIMALS;
}

export function formatTick(value: number, step: number): string {
  if (!Number.isFinite(value)) return "";
  const magnitude = Math.abs(value);
  // Outside the range where a decimal reads better than an exponent. A loss of 3.2e-5 is
  // legible; 0.000032 is a row of zeroes the eye has to count.
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(1);
  }
  return value.toFixed(decimalsFor(step));
}

/**
 * Widen a degenerate domain into one that can be drawn.
 *
 * A single-point series, or a constant one, has `min === max`. Left alone that divides by
 * zero and the line vanishes; widened, the point sits in the middle of a sensible axis,
 * which is what someone watching the first moments of a training run should see.
 */
export function padDomain(min: number, max: number): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    const margin = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return [min - margin, max + margin];
  }
  return [min, max];
}

export function extent(values: readonly number[]): [number, number] {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return [0, 1];
  return [Math.min(...finite), Math.max(...finite)];
}

/** A linear scale from `domain` onto `[rangeStart, rangeEnd]` pixels. */
export function linearScale(
  domain: [number, number],
  rangeStart: number,
  rangeEnd: number,
): Scale {
  const [low, high] = padDomain(domain[0], domain[1]);
  const span = high - low;
  return {
    domain: [low, high],
    project: (value) => rangeStart + ((value - low) / span) * (rangeEnd - rangeStart),
    ticks: (count = 5) => {
      const step = niceStep(span / Math.max(1, count));
      const first = Math.ceil(low / step) * step;
      const ticks: Tick[] = [];
      for (let value = first; value <= high + step * 1e-9; value += step) {
        // Re-derived from the step index rather than accumulated, so floating-point drift
        // cannot turn 0.30000000000000004 into a label.
        const exact = Math.round(value / step) * step;
        ticks.push({ value: exact, label: formatTick(exact, step) });
      }
      return ticks;
    },
  };
}

/**
 * A base-10 log scale, for the three EfficientAD loss terms.
 *
 * They span decades and start far apart, so on a linear axis two of the three are a flat
 * line along the bottom for the whole run. Non-positive values are clamped to `LOG_FLOOR`
 * rather than dropped: a loss that reaches exactly zero is a real observation, and a gap
 * in the line would read as missing data.
 */
export function logScale(domain: [number, number], rangeStart: number, rangeEnd: number): Scale {
  const low = Math.log10(Math.max(domain[0], LOG_FLOOR));
  const high = Math.log10(Math.max(domain[1], LOG_FLOOR));
  const [paddedLow, paddedHigh] = padDomain(low, high);
  const span = paddedHigh - paddedLow;

  return {
    domain: [10 ** paddedLow, 10 ** paddedHigh],
    project: (value) => {
      const transformed = Math.log10(Math.max(value, LOG_FLOOR));
      return rangeStart + ((transformed - paddedLow) / span) * (rangeEnd - rangeStart);
    },
    ticks: (count = 5) => {
      // One tick per decade, thinned when the range covers more decades than fit.
      const stride = Math.max(1, Math.ceil(span / Math.max(1, count)));
      const ticks: Tick[] = [];
      for (
        let exponent = Math.ceil(paddedLow);
        exponent <= paddedHigh + 1e-9;
        exponent += stride
      ) {
        const value = 10 ** exponent;
        ticks.push({ value, label: formatTick(value, value) });
      }
      return ticks;
    },
  };
}

/** An SVG `d` attribute through the projected points, skipping non-finite ones. */
export function linePath(
  points: readonly { x: number; y: number }[],
  xScale: Scale,
  yScale: Scale,
): string {
  const commands: string[] = [];
  let pen = "M";
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      // Lift the pen rather than bridging the gap: a straight line across missing data
      // is an assertion nobody made.
      pen = "M";
      continue;
    }
    commands.push(`${pen}${xScale.project(point.x).toFixed(2)} ${yScale.project(point.y).toFixed(2)}`);
    pen = "L";
  }
  return commands.join(" ");
}

/**
 * Bin values into `count` equal-width buckets over their own extent.
 *
 * Used for the score histogram, where the two classes must share one set of edges or the
 * bars cannot be compared. Callers pass the shared domain in for exactly that reason.
 */
export function histogram(
  values: readonly number[],
  domain: [number, number],
  count: number,
): number[] {
  const bins = new Array<number>(Math.max(1, count)).fill(0);
  const [low, high] = domain;
  const span = high - low;
  if (span <= 0) {
    bins[0] = values.length;
    return bins;
  }
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    const index = Math.min(bins.length - 1, Math.floor(((value - low) / span) * bins.length));
    if (index >= 0) bins[index] = (bins[index] ?? 0) + 1;
  }
  return bins;
}
