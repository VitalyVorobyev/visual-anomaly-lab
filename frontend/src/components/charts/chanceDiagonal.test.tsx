import type { Scale } from "@vitavision/charts";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";

import { chanceDiagonal } from "./chanceDiagonal";

it("draws from the origin to (1, 1) of the two scales", () => {
  const scale = (k: number) => ({ project: (v: number) => v * k }) as unknown as Scale;
  const svg = renderToStaticMarkup(<svg>{chanceDiagonal(scale(10), scale(20))}</svg>);
  expect(svg).toContain('x1="0"');
  expect(svg).toContain('x2="10"');
  expect(svg).toContain('y2="20"');
});
