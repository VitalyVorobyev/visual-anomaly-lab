/**
 * The review step's honesty check.
 *
 * ADR-0006 names the risk plainly: for a regular dataset the review is ceremony, and it
 * will be tempting to click through. These cases pin the two behaviours that make it
 * more than a notice — nothing to report says so, and something to report has to be
 * acknowledged before commit is possible.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ManifestWarning } from "../api/client";
import { WarningsPanel, commitBlocked } from "./WarningsPanel";

const VARIABLE_CHANNELS: ManifestWarning = {
  code: "variable_channel_count",
  message: "13 of 302 samples do not have 3 images.",
  paths: ["unsorted/group/2 (2 images)"],
};

const UNREADABLE: ManifestWarning = {
  code: "unreadable_file",
  message: "1 file could not be read and was skipped.",
  paths: ["broken.bmp"],
};

describe("commitBlocked", () => {
  it("does not block when there is nothing to report", () => {
    expect(commitBlocked([], false)).toBe(false);
  });

  it("blocks until warnings are acknowledged", () => {
    expect(commitBlocked([VARIABLE_CHANNELS], false)).toBe(true);
    expect(commitBlocked([VARIABLE_CHANNELS], true)).toBe(false);
  });
});

describe("WarningsPanel", () => {
  it("says so plainly when a scan found nothing", () => {
    render(<WarningsPanel warnings={[]} acknowledged={false} onAcknowledge={vi.fn()} />);

    expect(screen.getByText(/None\./)).toBeTruthy();
    // No acknowledgement is demanded for a clean scan.
    expect(screen.queryByRole("checkbox")).toBeNull();
  });

  it("lists every warning with its code, message and paths", () => {
    render(
      <WarningsPanel
        warnings={[VARIABLE_CHANNELS, UNREADABLE]}
        acknowledged={false}
        onAcknowledge={vi.fn()}
      />,
    );

    expect(screen.getByText("variable_channel_count")).toBeTruthy();
    expect(screen.getByText("unreadable_file")).toBeTruthy();
    expect(screen.getByText(/13 of 302 samples/)).toBeTruthy();
    expect(screen.getByText("broken.bmp")).toBeTruthy();
    expect(screen.getByText(/Warnings \(2\)/)).toBeTruthy();
  });

  it("asks for an explicit acknowledgement", () => {
    const onAcknowledge = vi.fn();
    render(
      <WarningsPanel
        warnings={[VARIABLE_CHANNELS]}
        acknowledged={false}
        onAcknowledge={onAcknowledge}
      />,
    );

    const checkbox = screen.getByRole("checkbox") as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
    checkbox.click();

    expect(onAcknowledge).toHaveBeenCalledWith(true);
  });
});
