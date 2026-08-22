import { afterEach, describe, expect, it, vi } from "vitest";
import { act } from "@testing-library/react";

afterEach(() => {
  vi.resetModules();
  delete window.__ANOMALY_LAB__;
  document.body.innerHTML = "";
});

describe("the entry point", () => {
  it("paints the shell's startup failure rather than mounting a workbench with no backend", async () => {
    document.body.innerHTML = '<div id="root"></div>';
    window.__ANOMALY_LAB__ = {
      startupError: {
        message: "`uv` was not found, so the backend could not be started.",
        detail: "Searched:\n  /usr/bin/uv",
      },
    };

    await act(async () => {
      await import("./main");
    });

    const root = document.getElementById("root");
    expect(root?.textContent).toContain("The lab could not start its backend.");
    expect(root?.textContent).toContain("`uv` was not found");
    expect(root?.textContent).toContain("/usr/bin/uv");
  });
});
