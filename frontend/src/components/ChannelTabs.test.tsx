/**
 * "Channel count is data, never schema" (ADR-0005), tested where a screen could break it.
 *
 * A dataset in the reference data has one capture group with two illuminations rather
 * than three. If any of these cases needed a special case in the component, that rule
 * would already be broken.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ImageSummary } from "../api/client";
import { ChannelTabs } from "./ChannelTabs";

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

describe("ChannelTabs", () => {
  it("renders one tab per image, whatever the count", () => {
    const three = [image(1, "bright"), image(2, "dark"), image(3, "dome")];

    render(<ChannelTabs images={three} active={0} onSelect={vi.fn()} />);

    expect(screen.getAllByRole("tab")).toHaveLength(3);
    expect(screen.getByRole("tab", { name: "dome" })).toBeTruthy();
  });

  it("renders a two-channel sample with no special case and no padding", () => {
    const two = [image(1, "bright"), image(2, "dark")];

    render(<ChannelTabs images={two} active={0} onSelect={vi.fn()} />);

    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual(["bright", "dark"]);
    // Nothing invents a missing third tab.
    expect(screen.queryByRole("tab", { name: "dome" })).toBeNull();
  });

  it("renders a single-view sample whose image has no channel at all", () => {
    render(<ChannelTabs images={[image(1, null)]} active={0} onSelect={vi.fn()} />);

    // Surfaced rather than hidden: a dataset the matcher did not recognize is still
    // browsable, and the operator can see that it was not recognized.
    expect(screen.getByRole("tab", { name: "unassigned" })).toBeTruthy();
  });

  it("marks exactly one tab selected", () => {
    const three = [image(1, "bright"), image(2, "dark"), image(3, "dome")];

    render(<ChannelTabs images={three} active={1} onSelect={vi.fn()} />);

    const selected = screen.getAllByRole("tab", { selected: true });
    expect(selected).toHaveLength(1);
    expect(selected[0]?.textContent).toBe("dark");
  });

  it("reports the index that was clicked", () => {
    const onSelect = vi.fn();
    render(
      <ChannelTabs images={[image(1, "bright"), image(2, "dark")]} active={0} onSelect={onSelect} />,
    );

    screen.getByRole("tab", { name: "dark" }).click();

    expect(onSelect).toHaveBeenCalledWith(1);
  });

});
