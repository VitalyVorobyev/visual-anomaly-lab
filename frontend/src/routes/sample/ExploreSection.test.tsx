/**
 * Explore's Text mode, drawn from a session in each of its states: SAM 3 missing, being
 * downloaded, unavailable for a reason the reader has to act on, and answering.
 *
 * The session is a plain object rather than the hook: what is under test is what the rail
 * says in each state, and the hook's requests are exercised by the backend's own tests.
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { ExploreTextCapability, ExploreTextResponse } from "../../api/client";
import { withProviders } from "../../test-harness";
import { ExploreSection } from "./ExploreSection";
import type { ExploreSession } from "./useExploreSession";
import type { ExploreTextSession } from "./useExploreTextSession";

const READY: ExploreTextCapability = {
  available: true,
  reason: null,
  asset_key: "sam3",
  installable: false,
  gated: true,
  access_url: "https://huggingface.co/facebook/sam3",
  max_phrase_length: 80,
  default_threshold: 0.5,
};

const ANSWER: ExploreTextResponse = {
  image_id: 7,
  phrase: "candle",
  threshold: 0.5,
  device: "mps",
  cached: false,
  warm: false,
  encode_ms: 2100,
  prompt_ms: 150,
  elapsed_ms: 2400,
  map_id: "a".repeat(32),
  map_url: `/api/explore/maps/${"a".repeat(32)}.png`,
  instances: [
    { index: 1, score: 0.97, box: { x0: 0, y0: 0, x1: 10, y1: 10 }, area: 80 },
    { index: 2, score: 0.81, box: { x0: 5, y0: 5, x1: 20, y1: 20 }, area: 120 },
  ],
  dropped: 3,
};

const idle = { isPending: false, error: null, mutate: vi.fn() };

function textSession(overrides: Partial<ExploreTextSession>): ExploreTextSession {
  return {
    capability: READY,
    asset: {
      key: "sam3",
      license_name: "SAM License",
      license_url: "https://huggingface.co/facebook/sam3/blob/x/LICENSE",
    },
    modelAssets: { isPending: false },
    installAsset: idle,
    install: vi.fn(),
    followedAssetJobId: undefined,
    assetJob: { job: undefined },
    cancelAssetJob: idle,
    phrase: "",
    setPhrase: vi.fn(),
    canAsk: false,
    ask: vi.fn(),
    retry: vi.fn(),
    request: { ...idle, data: undefined, variables: undefined },
    encoding: false,
    answer: undefined,
    instance: null,
    setInstance: vi.fn(),
    clear: vi.fn(),
    ...overrides,
  } as unknown as ExploreTextSession;
}

function renderText(text: ExploreTextSession) {
  const session = {
    capability: {
      isPending: false,
      error: null,
      data: { available: true, backbones: [], text: text.capability },
    },
    on: true,
    setOn: vi.fn(),
    mode: "text",
    setMode: vi.fn(),
    backbone: undefined,
    text,
    opacity: 0.8,
    setOpacity: vi.fn(),
    sendable: text.instance !== null,
    send: vi.fn(),
    toShape: { isPending: false, error: null },
  } as unknown as ExploreSession;
  return render(withProviders(<ExploreSection session={session} />));
}

describe("Explore's Text mode", () => {
  it("offers the licensed download when SAM 3 is only missing", () => {
    const install = vi.fn();
    renderText(
      textSession({
        capability: {
          ...READY,
          available: false,
          installable: true,
          reason: "SAM 3 is not downloaded: 3.4 GB, once, under the SAM License.",
        },
        install,
      }),
    );
    expect(screen.getByText(/not downloaded: 3.4 GB/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "SAM License" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Request access" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Accept licence & download/ }));
    expect(install).toHaveBeenCalledOnce();
  });

  it("follows the download as a job, with a cancel", () => {
    renderText(
      textSession({
        capability: { ...READY, available: false, installable: true, reason: "missing" },
        followedAssetJobId: 12,
        assetJob: { job: { status: "running", progress: 0.4, message: "1300.0 / 3285.4 MiB" } },
      } as unknown as Partial<ExploreTextSession>),
    );
    expect(screen.getByText("1300.0 / 3285.4 MiB")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "Phrase" })).toBeNull();
  });

  it("says why when SAM 3 cannot run at all", () => {
    renderText(
      textSession({
        capability: {
          ...READY,
          available: false,
          reason: "Install the backend's 'dl' extra to run SAM 3.",
        },
      }),
    );
    expect(screen.getByText("SAM 3 is unavailable")).toBeTruthy();
    expect(screen.getByText(/'dl' extra/)).toBeTruthy();
  });

  it("asks on submit and names the wait for a first phrase", () => {
    const ask = vi.fn();
    renderText(textSession({ phrase: "candle", canAsk: true, ask, encoding: true }));
    fireEvent.submit(screen.getByRole("textbox", { name: "Phrase" }).closest("form")!);
    expect(ask).toHaveBeenCalledOnce();
    expect(screen.getByRole("status").textContent).toMatch(/Encoding this image with SAM 3/);
  });

  it("lists instances by score, says what was dropped, and picks one", () => {
    const setInstance = vi.fn();
    renderText(textSession({ answer: ANSWER, setInstance }));
    expect(screen.getByText(/2 instances of “candle” · 3 weaker not shown/)).toBeTruthy();
    const chips = within(screen.getByRole("group", { name: "Instances" })).getAllByRole("switch");
    expect(chips.map((chip) => chip.textContent)).toEqual(["#1 · 0.97", "#2 · 0.81"]);
    fireEvent.click(chips[1]!);
    expect(setInstance).toHaveBeenCalledWith(2);
    expect(screen.getByText(/encoded 2100 ms/)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Send to editor/ })).toHaveProperty("disabled", true);
  });

  it("sends the picked instance", () => {
    renderText(textSession({ answer: ANSWER, instance: 1 }));
    expect(screen.getByRole("button", { name: /Send to editor/ })).toHaveProperty("disabled", false);
  });

  it("says so when nothing reached the cut", () => {
    renderText(
      textSession({
        answer: { ...ANSWER, phrase: "stain", instances: [], map_id: null, map_url: null, dropped: 0 },
      }),
    );
    expect(screen.getByText(/Nothing scored ≥ 0.50 for “stain”/)).toBeTruthy();
  });
});
