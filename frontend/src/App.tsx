/**
 * The shell: what is true on every screen.
 *
 * The old header set the wordmark at `text-sm` — the same size as the navigation links
 * beside it — so nothing in it was more important than anything else, and the active route
 * was distinguished only by font weight. It also had no way to tell whether the backend was
 * up: that fact lived exclusively on `/health`, which is the one screen you cannot reach a
 * conclusion from if the sidecar has died while you were on another one.
 */

import { NavLink, Outlet } from "react-router";

import { ThemeToggle, Tooltip, TooltipProvider, cn, focusRing } from "@vitavision/lab-ui";
import { useHealth } from "./hooks/useHealth";
import { THEME_STORAGE_KEY } from "./themeStorageKey";

const NAV = [
  { to: "/", label: "Datasets", end: true },
  { to: "/experiments", label: "Experiments", end: false },
  { to: "/compare", label: "Compare", end: false },
];

/**
 * The shell, with a height chain rather than a document that grows.
 *
 * `h-screen` + `min-h-0` down to `main` is what lets a route say "the canvas takes what is
 * left" and have that mean something. Without it `flex-1` is measured against content
 * height, the image sizes itself, and a screen whose whole job is looking at a picture
 * spends its first third on chrome.
 *
 * The scroll container moves from the document to `main`. The header is outside it and
 * already `sticky`, so it behaves identically; inner scrollers are unaffected.
 */
export function App() {
  return (
    <TooltipProvider>
      <div className="flex h-screen flex-col bg-ground text-fg">
        <header className="sticky top-0 z-40 border-b border-line bg-ground/85 backdrop-blur-sm">
          <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-3">
            <Wordmark />

            <nav className="flex items-center gap-1" aria-label="Main">
              {NAV.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    cn(
                      "rounded-control px-2.5 py-1 text-sm transition-colors",
                      focusRing,
                      isActive
                        ? "bg-raised font-medium text-fg"
                        : "text-fg-muted hover:bg-raised/60 hover:text-fg",
                    )
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>

            <div className="ml-auto flex items-center gap-3">
              <BackendStatus />
              <ThemeToggle storageKey={THEME_STORAGE_KEY} />
            </div>
          </div>
        </header>

        <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <Outlet />
        </main>
      </div>
    </TooltipProvider>
  );
}

/**
 * The default: a column of prose, forms and tables, capped at a readable width.
 *
 * Everything that is read rather than looked at lives here. `max-w-6xl` is the measure
 * that keeps a metric table and a paragraph scannable on a wide display.
 */
export function ReadingLayout() {
  return (
    <div
      data-layout="reading"
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
    >
      <div className="mx-auto w-full max-w-6xl px-6 py-8">
        <Outlet />
      </div>
    </div>
  );
}

/**
 * The layout for dataset work: controls support the content instead of pushing it down.
 *
 * A workspace owns the remaining viewport and lets the route name its one scroll region.
 * Keeping this contract separate from `ReadingLayout` is what prevents the dataset grid
 * from growing an inner scrollbar inside a page that already scrolls.
 */
export function WorkspaceLayout() {
  return (
    <div
      data-layout="workspace"
      className="flex min-h-0 flex-1 flex-col overflow-hidden"
    >
      <Outlet />
    </div>
  );
}

/**
 * The layout for screens whose content is an image.
 *
 * Two differences from the reading layout, both for the same reason. There is no width
 * cap: 72rem is right for prose and wrong for a photograph someone is inspecting, which
 * should use the window it was given. And the route gets the viewport's height as a flex
 * child rather than growing the page, so a canvas can fill what the header and the toolbar
 * leave rather than sizing itself and letting the page scroll.
 *
 * A nested layout route rather than a flag the shell reads off the URL: the routes that
 * want this are listed in one place, in `main.tsx`, next to the ones that do not.
 */
export function CanvasLayout({ flush = false }: { flush?: boolean }) {
  return (
    <div
      data-layout="canvas"
      className={cn(
        "flex min-h-0 flex-1 flex-col overflow-hidden",
        !flush && "gap-3 px-4 py-3",
      )}
    >
      <Outlet />
    </div>
  );
}

/**
 * The mark is an aperture: concentric rings around a point, which is both what a lens is
 * and what this application draws over and over — a map with a hot spot in it.
 */
function Wordmark() {
  return (
    <span className="flex shrink-0 items-center gap-2">
      <svg viewBox="0 0 16 16" className="size-4 text-signal" aria-hidden>
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.25" />
        <circle cx="8" cy="8" r="3.5" fill="none" stroke="currentColor" strokeWidth="1.25" />
        <circle cx="8" cy="8" r="1.25" fill="currentColor" />
      </svg>
      <span className="text-sm font-semibold tracking-tight text-fg">anomaly lab</span>
    </span>
  );
}

/**
 * Whether the sidecar is answering, in the corner, always.
 *
 * `useHealth` already polls every five seconds for the health screen; mounting it here
 * costs nothing extra because TanStack Query dedupes the two subscribers onto one request.
 */
function BackendStatus() {
  const { data, error, isPending } = useHealth();

  const state = error ? "down" : isPending && data === undefined ? "connecting" : "up";
  const copy = {
    up: "Backend is answering",
    down: "Backend is not reachable. Start it with scripts/dev-backend.sh.",
    connecting: "Contacting the backend…",
  }[state];

  return (
    <Tooltip content={copy}>
      <span className="flex items-center gap-1.5" role="status" aria-label={copy}>
        <span
          className={cn(
            "size-1.5 rounded-full",
            state === "up" && "bg-normal",
            state === "down" && "bg-defect",
            state === "connecting" && "animate-pulse bg-warn",
          )}
          aria-hidden
        />
        <span className="font-mono text-[11px] text-fg-subtle">
          {state === "up" ? "online" : state}
        </span>
      </span>
    </Tooltip>
  );
}

