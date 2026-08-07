import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router";

import { App } from "./App";
import { DatasetRoute } from "./routes/DatasetRoute";
import { DatasetsRoute } from "./routes/DatasetsRoute";
import { EchoRoute } from "./routes/EchoRoute";
import { ExperimentRoute } from "./routes/ExperimentRoute";
import { ExperimentSampleRoute } from "./routes/ExperimentSampleRoute";
import { ExperimentsRoute } from "./routes/ExperimentsRoute";
import { HealthRoute } from "./routes/HealthRoute";
import { ImportRoute } from "./routes/ImportRoute";
import { NotFoundRoute } from "./routes/NotFoundRoute";
import { SampleRoute } from "./routes/SampleRoute";
import { SplitsRoute } from "./routes/SplitsRoute";
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
            <Route index element={<DatasetsRoute />} />
            <Route path="import" element={<ImportRoute />} />
            <Route path="datasets/:datasetId" element={<DatasetRoute />} />
            <Route path="datasets/:datasetId/splits" element={<SplitsRoute />} />
            <Route path="datasets/:datasetId/samples/:sampleId" element={<SampleRoute />} />
            <Route path="experiments" element={<ExperimentsRoute />} />
            <Route path="experiments/:experimentId" element={<ExperimentRoute />} />
            <Route
              path="experiments/:experimentId/samples/:sampleId"
              element={<ExperimentSampleRoute />}
            />
            <Route path="health" element={<HealthRoute />} />
            {/* Kept as a debugging aid for the WebSocket path, off the navigation. */}
            <Route path="echo" element={<EchoRoute />} />
            {/* Never fail to an empty document again. */}
            <Route path="*" element={<NotFoundRoute />} />
          </Route>
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
