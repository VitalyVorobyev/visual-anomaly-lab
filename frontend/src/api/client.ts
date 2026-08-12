/**
 * The typed HTTP client, and readable names for the schemas it speaks.
 *
 * `generated.ts` is produced from the backend's OpenAPI schema by
 * `scripts/gen-api-types.sh` and committed, so `tsc` and CI work without a running
 * backend. Regenerate it whenever a route or schema changes; the diff *is* the API
 * contract changing.
 *
 * Every type below is an alias into that file rather than a hand-written interface. A
 * field the backend renames becomes a type error here, which is the whole point (ADR-0012).
 */

import createClient from "openapi-fetch";

import { resolveApiBaseUrl } from "./baseUrl";
import type { components, paths } from "./generated";

export const apiBaseUrl = resolveApiBaseUrl();

export const api = createClient<paths>({ baseUrl: apiBaseUrl });

type Schemas = components["schemas"];

export type HealthResponse = Schemas["HealthResponse"];

export type Label = Schemas["Label"];
export type LabelSource = Schemas["LabelSource"];
export type Subset = Schemas["Subset"];
export type Channel = Schemas["Channel"];

export type BulkLabelFilter = Schemas["BulkLabelFilter"];
export type BulkLabelRequest = Schemas["BulkLabelRequest"];

export type DatasetSummary = Schemas["DatasetSummary"];
export type DatasetDetail = Schemas["DatasetDetail"];
export type DatasetDeletionPreview = Schemas["DatasetDeletionPreview"];
export type DatasetDeletionResult = Schemas["DatasetDeletionResult"];
export type ReferencePackCatalog = Schemas["ReferencePackCatalog"];
export type ReferencePackInfo = Schemas["ReferencePackInfo"];
export type RegisterReferencePacksParams = Schemas["RegisterReferencePacksParams"];
export type SampleSummary = Schemas["SampleSummary"];
export type SamplePage = Schemas["SamplePage"];
export type ImageSummary = Schemas["ImageSummary"];

export type AnnotationDraft = Schemas["AnnotationDraft"];
export type AnnotationDocument = Schemas["AnnotationDocument-Output"];
export type AnnotationDocumentInput = Schemas["AnnotationDocument-Input"];
export type AnnotationLabel = Schemas["AnnotationLabel"];
export type AnnotationRevision = Schemas["AnnotationRevision"];
export type AnnotationPoint = Schemas["AnnotationPoint"];
export type PolygonShape = Schemas["PolygonShape-Output"];
export type BitmapShape = Schemas["BitmapShape-Output"];
export type AnnotationShape = PolygonShape | BitmapShape;
export type AssistPoint = Schemas["AssistPoint"];
export type AssistBox = Schemas["AssistBox"];
export type SegmentAssistRequest = Schemas["SegmentAssistRequest"];
export type SegmentAssistResponse = Schemas["SegmentAssistResponse"];
export type SegmentCandidate = Schemas["SegmentCandidate"];
export type SegmentAssistCapability = Schemas["SegmentAssistCapability"];

export type ModelAssetCatalog = Schemas["ModelAssetCatalog"];
export type ModelAssetInfo = Schemas["ModelAssetInfo"];

export type JobSummary = Schemas["JobSummary"];
export type JobDetail = Schemas["JobDetail"];
export type JobStatus = Schemas["JobStatus"];
export type JobKind = Schemas["JobKind"];
export type JobMetrics = Schemas["JobMetrics"];
export type MetricSeries = Schemas["MetricSeries"];
export type MetricPoint = Schemas["MetricPoint"];

export type AdapterInfo = Schemas["AdapterInfo"];
/**
 * A model used in both directions gets two schemas: `-Output` requires everything the
 * server always emits, `-Input` allows a client to omit anything with a default. The
 * review screen reads an Output and posts it straight back, which type-checks because
 * every Output is a valid Input.
 */
export type Manifest = Schemas["Manifest-Output"];
export type ManifestSample = Schemas["ManifestSample-Output"];
export type ManifestImage = Schemas["ManifestImage-Output"];
export type ManifestWarning = Schemas["ManifestWarning-Output"];
export type ChannelMapping = Schemas["ChannelMapping"];
export type WarningCode = Schemas["WarningCode"];
export type CommitResponse = Schemas["CommitResponse"];

export type SplitDetail = Schemas["SplitDetail"];
export type SplitParams = Schemas["SplitParams-Output"];
export type SplitParamsInput = Schemas["SplitParams-Input"];
export type SubsetComposition = Schemas["SubsetComposition"];

export type ImageTier = Schemas["ImageTier"];

export type ModelDescription = Schemas["ModelDescription"];
export type Capabilities = Schemas["Capabilities"];
export type MethodCatalog = Schemas["MethodCatalog"];
export type ExperimentSummary = Schemas["ExperimentSummary"];
export type ExperimentDetail = Schemas["ExperimentDetail"];
export type ExperimentDeletionPreview = Schemas["ExperimentDeletionPreview"];
export type ExperimentDeletionResult = Schemas["ExperimentDeletionResult"];
export type TrainingState = Schemas["TrainingState"];
export type ExperimentStatus = Schemas["ExperimentStatus"];
export type ExperimentSort = Schemas["ExperimentSort"];
export type MetricSummary = Schemas["MetricSummary"];
export type ResultsPage = Schemas["ResultsPage"];
export type SampleVerdict = Schemas["SampleVerdict"];
export type ThresholdReport = Schemas["ThresholdReport"];
export type ImageScore = Schemas["ImageScore"];
export type MapScale = Schemas["MapScale"];
export type SamplePreview = Schemas["SamplePreview"];
export type ArtifactListing = Schemas["ArtifactListing"];
export type ArtifactGroup = Schemas["ArtifactGroup"];
export type MapRender = Schemas["MapRender"];
export type DiagnosticIndex = Schemas["DiagnosticIndex"];
export type DiagnosticEntry = Schemas["DiagnosticEntry"];
export type DiagnosticKind = Schemas["DiagnosticKind"];
export type DiagnosticScope = Schemas["DiagnosticScope"];
export type DiagnosticOrigin = Schemas["DiagnosticOrigin"];
export type DiagnoseResponse = Schemas["DiagnoseResponse"];
export type PruneResult = Schemas["PruneResult"];
export type PruneScope = Schemas["PruneScope"];
export type Curve = Schemas["Curve"];
export type CurveSet = Schemas["CurveSet"];

export type ComparisonReport = Schemas["ComparisonReport"];
export type ComparedRun = Schemas["ComparedRun"];
export type ComparedSample = Schemas["ComparedSample"];
export type OperatingPoint = Schemas["OperatingPoint"];
export type ConfusionCounts = Schemas["ConfusionCounts"];

/**
 * Turn an openapi-fetch result into a value or an exception.
 *
 * The client returns `{ data, error }` rather than throwing, which is right for a
 * library and wrong for a TanStack Query `queryFn` — Query decides between its success
 * and error states by whether the function threw. Every hook funnels through here so
 * that decision is made the same way once.
 */
export function unwrap<T>(result: { data?: T; error?: unknown }, what: string): T {
  if (result.error !== undefined || result.data === undefined) {
    throw new Error(describeError(result.error, what));
  }
  return result.data;
}

function describeError(error: unknown, what: string): string {
  if (error && typeof error === "object" && "detail" in error) {
    const { detail } = error as { detail?: unknown };
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "reason" in detail) {
      const { reason } = detail as { reason?: unknown };
      if (typeof reason === "string") return reason;
    }
    // FastAPI's validation errors are a list of per-field objects; the first one is
    // almost always the one worth showing.
    if (Array.isArray(detail) && detail.length > 0) {
      const first = detail[0] as { msg?: unknown };
      if (typeof first?.msg === "string") return first.msg;
    }
  }
  return `The backend did not return ${what}.`;
}
