/**
 * The tile's one subtle contract: ticking the box selects, in the same click.
 *
 * The bug being guarded against was a real browser behaviour — cancelling a checkbox's
 * click to stop the surrounding anchor navigating makes the browser restore the previous
 * `checked` value after React has written the new one, so the tick arrived one render late.
 * A DOM emulator does not reproduce that, so the assertion that actually holds the line is
 * the structural one: **the box is not inside the link**. Nothing then needs cancelling,
 * and the race has nowhere to live. The rest pins what the grid depends on — the callback,
 * its modifier keys, and a plain click still opening the sample.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it, vi } from "vitest";

import { SampleTile } from "./SampleTile";

const SAMPLE = {
  id: 12,
  dataset_id: 7,
  group_key: "set1/no-defect",
  external_id: "12",
  label: "normal" as const,
  label_source: "import" as const,
  notes: null,
  images: [
    {
      id: 501,
      channel: "bright",
      channel_id: 1,
      width: 1280,
      height: 1024,
      bit_depth: 24,
      file_size: 2_400_000,
      path: "/roots/fixture/set1/bright/12.png",
    },
  ],
};

/** Renders the tile with selection wired to real state, as the grid wires it. */
function renderTile(selected = false) {
  const onSelect = vi.fn();
  const view = render(
    <MemoryRouter initialEntries={["/datasets/7"]}>
      <SampleTile
        datasetId={7}
        sample={SAMPLE}
        search="label=normal"
        selected={selected}
        onSelect={onSelect}
      />
    </MemoryRouter>,
  );
  return { onSelect, view };
}

const box = () => screen.getByRole("checkbox", { name: "Select set1/no-defect/12" });

describe("a sample tile", () => {
  it("shows the tick as soon as the selection is applied, not a render later", () => {
    const { onSelect, view } = renderTile();
    expect(box().getAttribute("aria-checked")).toBe("false");

    fireEvent.click(box());
    expect(onSelect).toHaveBeenCalledTimes(1);

    // What the caller does with that click: the same tile, now selected. Re-rendering with
    // the new prop is the only thing that may change the tick — nothing else in the tile
    // gets to leave it behind.
    view.rerender(
      <MemoryRouter initialEntries={["/datasets/7"]}>
        <SampleTile
          datasetId={7}
          sample={SAMPLE}
          search="label=normal"
          selected
          onSelect={onSelect}
        />
      </MemoryRouter>,
    );
    expect(box().getAttribute("aria-checked")).toBe("true");
  });

  it("keeps the box out of the link, so ticking it cannot navigate", () => {
    renderTile();
    const link = screen.getByRole("link");
    expect(link.getAttribute("href")).toBe("/datasets/7/samples/12?label=normal");
    expect(link.contains(box())).toBe(false);
  });

  it("passes the modifier keys through, so shift extends and the platform key toggles", () => {
    const { onSelect } = renderTile();

    fireEvent.click(box(), { shiftKey: true });
    expect(onSelect.mock.calls[0]?.[0]).toMatchObject({ shiftKey: true });

    fireEvent.click(box(), { metaKey: true });
    expect(onSelect.mock.calls[1]?.[0]).toMatchObject({ metaKey: true });
  });

  it("leaves a plain click on the tile to open the sample", () => {
    const { onSelect } = renderTile();
    fireEvent.click(screen.getByRole("link"));
    expect(onSelect).not.toHaveBeenCalled();
  });
});
