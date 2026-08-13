/**
 * The log tail is clipped, not scrolled.
 *
 * This component appears inside pages that already scroll — the Prepare tab, the import
 * screen — so a `max-h-* overflow-y-auto` console here is a second scrollbar on the same
 * axis as the first. Clipping to the tail keeps the terminal reading like a terminal, and
 * the disclosure puts every line back into the page's own flow rather than behind a second
 * scroller.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { JobDetail } from "../api/client";
import { JobProgress } from "./JobProgress";
import { withProviders } from "../test-harness";

const job = {
  id: 1,
  kind: "region_build",
  status: "running",
  progress: 0.5,
  message: "preparing",
  error: null,
  result: {},
} as unknown as JobDetail;

function renderLog(lines: string[]) {
  return render(withProviders(<JobProgress jobId={1} job={job} lines={lines} error={null} />));
}

const many = Array.from({ length: 500 }, (_, index) => `line ${index}`);

describe("JobProgress", () => {
  it("puts no scroller in the console", () => {
    const { container } = renderLog(many);

    const scrollers = [...container.querySelectorAll("*")].filter((element) =>
      String(element.className).includes("overflow-y-auto"),
    );

    expect(scrollers).toHaveLength(0);
  });

  it("shows the newest lines without being scrolled there", () => {
    const { container } = renderLog(many);
    const tail = container.querySelector("pre")!;

    expect(tail.textContent).toContain("line 499");
    expect(tail.textContent).not.toContain("line 0\n");
  });

  it("keeps the whole log reachable", () => {
    const { container } = renderLog(many);

    expect(screen.getByText("Full log")).toBeTruthy();
    // Two `<pre>`s: the clipped tail, and the full log inside the disclosure.
    const panes = container.querySelectorAll("pre");
    expect(panes).toHaveLength(2);
    expect(panes[1]!.textContent).toContain("line 0");
  });

  it("does not offer to expand a log that is already whole", () => {
    renderLog(["only line"]);

    expect(screen.queryByText("Full log")).toBeNull();
  });
});
