// Self-hosted rather than fetched: the desktop shell serves from tauri://localhost with no
// network guarantee, so a webfont request is a font that sometimes does not arrive.
import "@fontsource-variable/ibm-plex-sans/wght.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "@fontsource/ibm-plex-mono/latin-600.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, StrictMode, Suspense } from "react";
import { createRoot } from "react-dom/client";
import { HashRouter, Route, Routes } from "react-router";

import { App, CanvasLayout, ReadingLayout, WorkspaceLayout } from "./App";
import { CompareRoute } from "./routes/CompareRoute";
import { CompareSampleRoute } from "./routes/compare/CompareSampleRoute";
import { AnnotationQueueRoute } from "./routes/AnnotationQueueRoute";
import { DatasetRoute } from "./routes/DatasetRoute";
import { DatasetsRoute } from "./routes/DatasetsRoute";
import { EchoRoute } from "./routes/EchoRoute";
import { ExperimentRoute } from "./routes/ExperimentRoute";
import { ExperimentSampleRoute } from "./routes/ExperimentSampleRoute";
import { CreateExperimentRoute, ExperimentsRoute } from "./routes/ExperimentsRoute";
import { HealthRoute } from "./routes/HealthRoute";
import { ImportRoute } from "./routes/ImportRoute";
import { NotFoundRoute } from "./routes/NotFoundRoute";
import { SampleRoute } from "./routes/SampleRoute";
import { SplitsRoute } from "./routes/SplitsRoute";
import { RegionPreparationRoute } from "./routes/RegionPreparationRoute";
import "./styles.css";
import { initTheme } from "./theme";

const queryClient = new QueryClient();
const AnnotationEditorRoute = lazy(async () => {
  const module = await import("./routes/AnnotationEditorRoute");
  return { default: module.AnnotationEditorRoute };
});

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
        <Routes>
          <Route element={<App />}>
            {/* Read rather than looked at: capped at a readable measure. */}
            <Route element={<ReadingLayout />}>
              <Route index element={<DatasetsRoute />} />
              <Route path="import" element={<ImportRoute />} />
              <Route path="datasets/:datasetId/splits" element={<SplitsRoute />} />
              <Route path="datasets/:datasetId/samples/:sampleId" element={<SampleRoute />} />
              <Route path="experiments" element={<ExperimentsRoute />} />
              <Route path="experiments/new" element={<CreateExperimentRoute />} />
              <Route
                path="datasets/:datasetId/experiments"
                element={<ExperimentsRoute />}
              />
              <Route
                path="datasets/:datasetId/experiments/new"
                element={<CreateExperimentRoute />}
              />
              <Route path="compare" element={<CompareRoute />} />
              <Route path="experiments/:experimentId" element={<ExperimentRoute />} />
              <Route path="health" element={<HealthRoute />} />
              {/* Kept as a debugging aid for the WebSocket path, off the navigation. */}
              <Route path="echo" element={<EchoRoute />} />
              {/* Never fail to an empty document again. */}
              <Route path="*" element={<NotFoundRoute />} />
            </Route>

            {/* Dataset work owns the viewport: its data surface is the one scroll region. */}
            <Route element={<WorkspaceLayout />}>
              <Route path="datasets/:datasetId" element={<DatasetRoute />} />
              <Route
                path="datasets/:datasetId/annotate"
                element={<AnnotationQueueRoute />}
              />
              <Route
                path="datasets/:datasetId/prepare"
                element={<RegionPreparationRoute />}
              />
            </Route>

            {/* Annotation is a flush workbench: its rails own every edge of the viewport. */}
            <Route element={<CanvasLayout flush />}>
              <Route
                path="datasets/:datasetId/annotate/:sampleId/:imageId"
                element={
                  <Suspense
                    fallback={
                      <div className="grid min-h-0 flex-1 place-items-center bg-canvas text-sm text-fg-muted">
                        Loading annotation editor…
                      </div>
                    }
                  >
                    <AnnotationEditorRoute />
                  </Suspense>
                }
              />
            </Route>

            {/* The image is the content: full window width, and the viewport's height. */}
            <Route element={<CanvasLayout />}>
              <Route
                path="experiments/:experimentId/samples/:sampleId"
                element={<ExperimentSampleRoute />}
              />
              {/* The same gesture, N runs wide: one sample under every compared method. */}
              <Route path="compare/samples/:sampleId" element={<CompareSampleRoute />} />
            </Route>
          </Route>
        </Routes>
      </HashRouter>
    </QueryClientProvider>
  </StrictMode>,
);
