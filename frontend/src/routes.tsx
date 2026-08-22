/**
 * The route table: where every screen's layout and scroll ownership is visible at a glance.
 *
 * Extracted from `main.tsx` so it can be rendered under a `MemoryRouter` in a test. The
 * dataset branch in particular makes a claim about react-router's ranking -- that declaring
 * `datasets/:datasetId` as a layout route does not swallow the sample viewer or the
 * annotation editor, which are declared under other layouts -- and a claim about ranking is
 * worth asserting rather than believing.
 */

import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router";

import { App, CanvasLayout, ReadingLayout } from "./App";
import { CompareRoute } from "./routes/CompareRoute";
import { CompareSampleRoute } from "./routes/compare/CompareSampleRoute";
import { AnnotationQueueRoute } from "./routes/AnnotationQueueRoute";
import { DatasetLayout } from "./routes/dataset/DatasetLayout";
import { DatasetRoute } from "./routes/DatasetRoute";
import { DatasetsRoute } from "./routes/datasets/DatasetsRoute";
import { EchoRoute } from "./routes/EchoRoute";
import { ExperimentRoute } from "./routes/ExperimentRoute";
import { ExperimentSampleRoute } from "./routes/ExperimentSampleRoute";
import {
  CreateExperimentRoute,
  DatasetCreateExperimentRoute,
  DatasetExperimentsRoute,
  ExperimentsRoute,
} from "./routes/ExperimentsRoute";
import { HealthRoute } from "./routes/HealthRoute";
import { ImportRoute } from "./routes/ImportRoute";
import { NotFoundRoute } from "./routes/NotFoundRoute";
import { SampleRoute } from "./routes/SampleRoute";
import { SplitsRoute } from "./routes/SplitsRoute";
import { RegionPreparationRoute } from "./routes/RegionPreparationRoute";

const AnnotationEditorRoute = lazy(async () => {
  const module = await import("./routes/AnnotationEditorRoute");
  return { default: module.AnnotationEditorRoute };
});

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<App />}>
        {/* Read rather than looked at: capped at a readable measure. */}
        <Route element={<ReadingLayout />}>
          <Route index element={<DatasetsRoute />} />
          <Route path="import" element={<ImportRoute />} />
          <Route path="experiments" element={<ExperimentsRoute />} />
          <Route path="experiments/new" element={<CreateExperimentRoute />} />
          <Route path="compare" element={<CompareRoute />} />
          <Route path="experiments/:experimentId" element={<ExperimentRoute />} />
          <Route path="health" element={<HealthRoute />} />
          {/* Kept as a debugging aid for the WebSocket path, off the navigation. */}
          <Route path="echo" element={<EchoRoute />} />
          {/* Never fail to an empty document again. */}
          <Route path="*" element={<NotFoundRoute />} />
        </Route>

        {/*
         * One dataset, five tabs. The layout owns the identity band and the section strip
         * exactly once and does not scroll; each child owns exactly one scroll region below
         * it. The two dataset screens that are not tabs -- the sample viewer and the
         * annotation editor -- stay under their own layouts above and below: a branch is
         * matched only at its leaf, so `…/annotate` cannot claim `…/annotate/1/2`.
         */}
        <Route path="datasets/:datasetId" element={<DatasetLayout />}>
          <Route index element={<DatasetRoute />} />
          <Route path="annotate" element={<AnnotationQueueRoute />} />
          <Route path="prepare" element={<RegionPreparationRoute />} />
          <Route path="splits" element={<SplitsRoute />} />
          <Route path="experiments" element={<DatasetExperimentsRoute />} />
          <Route path="experiments/new" element={<DatasetCreateExperimentRoute />} />
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

        {/* Flush as well: the sample viewer's own band and rail own their edges. */}
        <Route element={<CanvasLayout flush />}>
          <Route path="datasets/:datasetId/samples/:sampleId" element={<SampleRoute />} />
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
  );
}
