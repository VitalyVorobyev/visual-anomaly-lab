/**
 * The dataset branch makes a claim about react-router's ranking, so the claim is asserted.
 *
 * `datasets/:datasetId` is now a layout route with five children. The two dataset screens
 * that are *not* tabs — the sample viewer and the annotation editor — are declared under
 * other layouts, and must keep matching there: a branch is matched only at its leaf, so the
 * three-segment `…/annotate` tab cannot claim the five-segment editor path. Believing that
 * is cheaper than testing it right up until the day it is wrong and the editor renders with
 * a dataset band across the top of a flush canvas.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { AppRoutes } from "./routes";
import { withProviders } from "./test-harness";

function renderAt(path: string) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>,
    ),
  );
}

function band(container: HTMLElement) {
  return container.querySelector('[data-band="dataset"]');
}

describe("the route table", () => {
  it("puts each of the five tabs inside the dataset band", () => {
    for (const path of [
      "/datasets/7",
      "/datasets/7/annotate",
      "/datasets/7/prepare",
      "/datasets/7/splits",
      "/datasets/7/experiments",
      "/datasets/7/experiments/new",
    ]) {
      const { container, unmount } = renderAt(path);

      expect(band(container), path).not.toBeNull();
      unmount();
    }
  });

  it("gives the sample viewer the canvas, not the band and not a reading column", () => {
    const { container } = renderAt("/datasets/7/samples/9");

    expect(band(container)).toBeNull();
    // The picture is the screen: a 72rem measure and a page scroller are what boxed it at
    // `h-96` under three panels of chrome.
    expect(container.querySelector('[data-layout="reading"]')).toBeNull();
    expect(container.querySelector('[data-layout="canvas"]')).not.toBeNull();
    expect(container.querySelector('[data-layout="sample"]')).not.toBeNull();
  });

  it("keeps the annotation editor out of the band", () => {
    const { container } = renderAt("/datasets/7/annotate/1/2");

    expect(band(container)).toBeNull();
    expect(container.querySelector('[data-layout="canvas"]')).not.toBeNull();
    expect(screen.getByText(/Loading annotation editor/)).toBeTruthy();
  });

  it("keeps the cross-dataset catalogue out of the band", () => {
    const { container } = renderAt("/experiments");

    expect(band(container)).toBeNull();
    expect(container.querySelector('[data-layout="reading"]')).not.toBeNull();
  });

  it("gives the shell a frame that cannot scroll the document", () => {
    const { container } = renderAt("/");
    const shell = container.querySelector('[data-layout="shell"]')!;

    // `100vh` could exceed the visible height and leak a second scrollbar onto the
    // document; `h-full` resolves against a `#root` that styles.css pins to what is visible.
    expect(shell.className).toContain("h-full");
    expect(shell.className).toContain("overflow-hidden");
    expect(shell.className).not.toContain("h-screen");
  });
});
