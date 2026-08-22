import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { CrashBoundary, CrashScreen, installCrashHandlers } from "./CrashScreen";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function Throws(): never {
  throw new Error("Failed to fetch dynamically imported module: AnnotationEditorRoute");
}

describe("CrashScreen", () => {
  it("names the failure it is actually reporting, not always the window", () => {
    render(
      <CrashScreen
        headline="The lab could not start its backend."
        message="`uv` was not found, so the backend could not be started."
        detail={"Searched:\n  /usr/bin/uv\n  /Users/someone/.local/bin/uv"}
      />,
    );

    const panel = screen.getByRole("alert");
    expect(panel.textContent).toContain("The lab could not start its backend.");
    expect(panel.textContent).toContain("`uv` was not found");
    expect(panel.textContent).toContain("/Users/someone/.local/bin/uv");
  });

  it("keeps the window headline when none is given", () => {
    render(<CrashScreen message="something threw" />);

    expect(screen.getByRole("alert").textContent).toContain("could not draw its window");
  });
});

describe("CrashBoundary", () => {
  it("shows what was thrown, where React 19 would otherwise unmount the root", () => {
    // React logs the caught error itself; the point of the test is the screen, not the log.
    vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <CrashBoundary>
        <Throws />
      </CrashBoundary>,
    );

    expect(screen.getByRole("alert").textContent).toContain(
      "Failed to fetch dynamically imported module",
    );
  });

  it("stays out of the way while nothing is wrong", () => {
    render(
      <CrashBoundary>
        <p>the workbench</p>
      </CrashBoundary>,
    );

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("the workbench")).toBeTruthy();
  });
});

describe("installCrashHandlers", () => {
  it("paints an uncaught error into an empty root — the black-window case", () => {
    const container = document.createElement("div");
    document.body.append(container);
    installCrashHandlers(container);

    window.dispatchEvent(
      new ErrorEvent("error", {
        error: new Error("styles.css failed to load"),
        message: "styles.css failed to load",
      }),
    );

    expect(container.textContent).toContain("styles.css failed to load");
    container.remove();
  });

  it("leaves a working screen alone: a live UI reports its own failures", () => {
    const container = document.createElement("div");
    container.append(document.createElement("main"));
    document.body.append(container);
    installCrashHandlers(container);

    window.dispatchEvent(
      new ErrorEvent("error", { error: new Error("a scoring run that failed"), message: "boom" }),
    );

    expect(container.textContent).not.toContain("a scoring run that failed");
    container.remove();
  });
});
