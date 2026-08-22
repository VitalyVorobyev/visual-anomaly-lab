/**
 * Server-state boundary for the annotation editor.
 *
 * Draft responses carry their concurrency token in an HTTP header, so the editor keeps
 * the value beside the typed body. A save can never silently overwrite another window:
 * the backend rejects a stale token and this hook surfaces that conflict as an error.
 *
 * A dataset annotates either each image or each whole sample (ADR-0036), and the two use
 * different routes, different ETag namespaces and different cache keys. That branch lives
 * *here*, in one `DraftTarget`, rather than in the editor: the screen is the same screen
 * either way, and duplicating the fork through every call site is how the two halves drift.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  AnnotationDocument,
  AnnotationDocumentInput,
  AnnotationLabel,
  AnnotationRevision,
  AnnotationScope,
  AnnotationScopeState,
  SegmentAssistCapability,
  SegmentAssistRequest,
  SegmentAssistResponse,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

/** Which document the editor is editing: one photograph, or one whole part. */
export type DraftTarget =
  | { scope: "image"; imageId: number }
  | { scope: "sample"; sampleId: number };

/** A draft reduced to what the editor actually needs, whichever scope produced it. */
export interface DraftEnvelope {
  document: AnnotationDocument;
  version: number;
  etag: string;
}

function targetKey(target: DraftTarget) {
  return target.scope === "sample"
    ? queryKeys.annotationSampleDraft(target.sampleId)
    : queryKeys.annotationDraft(target.imageId);
}

function requireEtag(response: Response): string {
  const etag = response.headers.get("etag");
  if (!etag) throw new Error("The backend omitted the annotation concurrency token.");
  return etag;
}

export function useAnnotationLabels(datasetId: number | undefined) {
  return useQuery<AnnotationLabel[]>({
    queryKey: queryKeys.annotationLabels(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/annotation-labels", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the annotation taxonomy",
      ),
    enabled: datasetId !== undefined,
  });
}

export function useAnnotationScope(datasetId: number | undefined) {
  return useQuery<AnnotationScopeState>({
    queryKey: queryKeys.annotationScope(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/annotation-scope", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the annotation scope",
      ),
    enabled: datasetId !== undefined,
  });
}

export function useSetAnnotationScope(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<AnnotationScopeState, Error, AnnotationScope>({
    mutationFn: async (scope) =>
      unwrap(
        await api.PUT("/api/datasets/{dataset_id}/annotation-scope", {
          params: { path: { dataset_id: datasetId } },
          body: { scope },
        }),
        "the annotation scope",
      ),
    onSuccess: (state) => {
      queryClient.setQueryData(queryKeys.annotationScope(datasetId), state);
      // The scope decides which routes the editor calls, and it travels on the dataset
      // detail every screen reads.
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}

/** Open the draft for this target, creating it from the newest truth if none is open. */
export function useEditorDraft(target: DraftTarget | undefined) {
  return useQuery<DraftEnvelope>({
    queryKey: target ? targetKey(target) : ["annotations", "none"],
    queryFn: async () => {
      if (!target) throw new Error("no annotation target");
      if (target.scope === "sample") {
        const result = await api.POST("/api/samples/{sample_id}/annotations/draft", {
          params: { path: { sample_id: target.sampleId } },
        });
        const draft = unwrap(result, "the annotation draft");
        return {
          document: draft.document,
          version: draft.version,
          etag: requireEtag(result.response),
        };
      }
      const result = await api.POST("/api/images/{image_id}/annotations/draft", {
        params: { path: { image_id: target.imageId } },
      });
      const draft = unwrap(result, "the annotation draft");
      return {
        document: draft.document,
        version: draft.version,
        etag: requireEtag(result.response),
      };
    },
    enabled: target !== undefined,
    staleTime: Infinity,
  });
}

export function useSaveDraft(target: DraftTarget) {
  const queryClient = useQueryClient();
  return useMutation<DraftEnvelope, Error, { document: AnnotationDocumentInput; etag: string }>({
    mutationFn: async ({ document, etag }) => {
      // The two calls are spelled out rather than folded into one `result`: their response
      // bodies are different types, and a union of them widens `draft` into something
      // neither route actually returns.
      if (target.scope === "sample") {
        const result = await api.PUT("/api/samples/{sample_id}/annotations/draft", {
          params: { path: { sample_id: target.sampleId }, header: { "If-Match": etag } },
          body: document,
        });
        const draft = unwrap(result, "the saved annotation draft");
        return {
          document: draft.document,
          version: draft.version,
          etag: requireEtag(result.response),
        };
      }
      const result = await api.PUT("/api/images/{image_id}/annotations/draft", {
        params: { path: { image_id: target.imageId }, header: { "If-Match": etag } },
        body: document,
      });
      const draft = unwrap(result, "the saved annotation draft");
      return {
        document: draft.document,
        version: draft.version,
        etag: requireEtag(result.response),
      };
    },
    onSuccess: (saved) => {
      queryClient.setQueryData(targetKey(target), saved);
    },
  });
}

/**
 * Freeze the draft.
 *
 * Always a list, because a sample-scoped completion writes one immutable revision per
 * image of the sample — the fan-out that keeps every consumer below it image-keyed.
 */
export function useCompleteDraft(target: DraftTarget, imageIds: readonly number[]) {
  const queryClient = useQueryClient();
  return useMutation<AnnotationRevision[], Error, string>({
    mutationFn: async (etag) => {
      if (target.scope === "sample") {
        return unwrap(
          await api.POST("/api/samples/{sample_id}/annotations/complete", {
            params: { path: { sample_id: target.sampleId }, header: { "If-Match": etag } },
          }),
          "the completed annotation revisions",
        );
      }
      const revision = unwrap(
        await api.POST("/api/images/{image_id}/annotations/complete", {
          params: { path: { image_id: target.imageId }, header: { "If-Match": etag } },
        }),
        "the completed annotation revision",
      );
      return [revision];
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: targetKey(target) });
      for (const imageId of imageIds) {
        void queryClient.invalidateQueries({ queryKey: queryKeys.annotationRevisions(imageId) });
      }
    },
  });
}

export function useSegmentAssistCapability() {
  return useQuery<SegmentAssistCapability>({
    queryKey: queryKeys.segmentAssist(),
    queryFn: async () =>
      unwrap(await api.GET("/api/segment-assist"), "the contour-assistance capability"),
  });
}

export function useSegmentAssist(imageId: number) {
  return useMutation<SegmentAssistResponse, Error, SegmentAssistRequest>({
    mutationFn: async (body) =>
      unwrap(
        await api.POST("/api/images/{image_id}/segment-assist", {
          params: { path: { image_id: imageId } },
          body,
        }),
        "the contour suggestions",
      ),
  });
}
