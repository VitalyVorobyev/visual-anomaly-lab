import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router";

import { App } from "./App";
import { EchoRoute } from "./routes/EchoRoute";
import { HealthRoute } from "./routes/HealthRoute";
import { NotFoundRoute } from "./routes/NotFoundRoute";
import "./styles.css";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

// HashRouter, not BrowserRouter: routing must not depend on the path the bundle happens
// to be served from. The desktop shell loads `…/index.html` (and production serves from
// `tauri://localhost`), where a path-based router matches no route and renders a silent
// blank page. Routing off the fragment behaves identically under http://, tauri:// and
// file://, and survives a reload on a nested route.
createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <Routes>
          <Route element={<App />}>
            <Route index element={<HealthRoute />} />
            <Route path="echo" element={<EchoRoute />} />
            {/* Never fail to an empty document again. */}
            <Route path="*" element={<NotFoundRoute />} />
          </Route>
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
