/**
 * The contract between the grid and the sample viewer.
 *
 * Both routes rebuild their `useSamples` query from the URL. If the two ever disagree the
 * viewer pages through a *different* set than the grid displayed, and nothing errors —
 * the arrows simply go somewhere surprising. These tests pin the round trip and the
 * validation that stops a hand-edited URL from reaching the API.
 */

import { describe, expect, it } from "vitest";

import {
  EMPTY_BROWSE,
  PAGE_SIZE,
  readBrowseState,
  toBulkFilters,
  toSampleQuery,
  writeBrowseState,
} from "./browseState";

const read = (query: string) => readBrowseState(new URLSearchParams(query));

describe("browse state", () => {
  it("round-trips a fully specified browse", () => {
    const state = {
      label: "defect",
      channelId: 3,
      splitId: 5,
      subset: "test",
      offset: 400,
    } as const;

    expect(read(writeBrowseState(state).toString())).toEqual(state);
  });

  it("writes nothing for an unfiltered first page", () => {
    expect(writeBrowseState(EMPTY_BROWSE).toString()).toBe("");
    expect(read("")).toEqual(EMPTY_BROWSE);
  });

  it("rejects values the API does not know", () => {
    // A hand-edited URL must not be able to smuggle an unknown label into a request.
    expect(read("label=maybe").label).toBeUndefined();
    expect(read("subset=holdout&split=1").subset).toBeUndefined();
    expect(read("channel=-4").channelId).toBeUndefined();
    expect(read("channel=abc").channelId).toBeUndefined();
    expect(read("offset=1.5").offset).toBe(0);
  });

  it("drops a subset that has no split to belong to", () => {
    // The API treats a subset without a split as meaningless; so does the URL, or a
    // leftover `subset=train` would silently narrow an unsplit view.
    expect(read("subset=train").subset).toBeUndefined();
    expect(read("subset=train&split=2").subset).toBe("train");

    const orphaned = { ...EMPTY_BROWSE, subset: "train" } as const;
    expect(writeBrowseState(orphaned).has("subset")).toBe(false);
  });

  it("asks for one page at the offset it was given", () => {
    const query = toSampleQuery({ ...EMPTY_BROWSE, label: "normal", offset: 200 });

    expect(query).toMatchObject({ label: "normal", limit: PAGE_SIZE, offset: 200 });
  });

  it("drops paging from the bulk filter", () => {
    // "Label all matching" means the whole matching set, not the page on screen.
    const filters = toBulkFilters({ ...EMPTY_BROWSE, label: "unlabeled", offset: 600 });

    expect(filters).toEqual({
      label: "unlabeled",
      channel_id: null,
      split_id: null,
      subset: null,
    });
  });
});
