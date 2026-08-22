// Self-hosted rather than fetched: the desktop shell serves from tauri://localhost with no
// network guarantee, so a webfont request is a font that sometimes does not arrive.
import "@fontsource-variable/ibm-plex-sans/wght.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter } from "react-router";

import { initTheme } from "@vitavision/lab-ui";

import { CrashBoundary, installCrashHandlers } from "./components/CrashScreen";
import { AppRoutes } from "./routes";
import "./styles.css";
import { THEME_STORAGE_KEY } from "./themeStorageKey";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

// Before the first render, so a module that throws on the way in still says so. The
// desktop shell has no console: an uncaught error there is a black window and nothing
// else. See `components/CrashScreen.tsx`.
installCrashHandlers(container);

// index.html already painted the stored choice before first paint; this subscribes so that
// a choice of "system" keeps following the OS after mount.
initTheme(THEME_STORAGE_KEY);

// HashRouter, not BrowserRouter: routing must not depend on the path the bundle happens
// to be served from. The desktop shell loads `…/index.html` (and production serves from
// `tauri://localhost`), where a path-based router matches no route and renders a silent
// blank page. Routing off the fragment behaves identically under http://, tauri:// and
// file://, and survives a reload on a nested route.
createRoot(container).render(
  <StrictMode>
    {/* Outside everything it guards, and outside the router in particular: a route that
        throws must still be caught, and a router that fails to construct must still be
        reported rather than unmounting the root into a black window. */}
    <CrashBoundary>
      <QueryClientProvider client={queryClient}>
        <HashRouter>
          <AppRoutes />
        </HashRouter>
      </QueryClientProvider>
    </CrashBoundary>
  </StrictMode>,
);
