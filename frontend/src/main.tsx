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

import { AppRoutes } from "./routes";
import "./styles.css";
import { initTheme } from "./theme";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

// index.html already painted the stored choice before first paint; this subscribes so that
// a choice of "system" keeps following the OS after mount.
initTheme();

// HashRouter, not BrowserRouter: routing must not depend on the path the bundle happens
// to be served from. The desktop shell loads `…/index.html` (and production serves from
// `tauri://localhost`), where a path-based router matches no route and renders a silent
// blank page. Routing off the fragment behaves identically under http://, tauri:// and
// file://, and survives a reload on a nested route.
createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <AppRoutes />
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
