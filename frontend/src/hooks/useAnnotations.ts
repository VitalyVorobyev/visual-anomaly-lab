/** Server-state boundary for the annotation editor.
 *
 * Draft responses carry their concurrency token in an HTTP header, so the editor keeps
 * the value beside the typed body. A save can never silently overwrite another window:
 * the backend rejects a stale token and this hook surfaces that conflict as an error.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  AnnotationDocumentInput,
  AnnotationDraft,
  AnnotationLabel,
  AnnotationRevision,
  SegmentAssistCapability,
  SegmentAssistRequest,
  SegmentAssistResponse,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

export interface DraftEnvelope {
  draft: AnnotationDraft;
  etag: string;
}

function withEtag(
  result: { data?: AnnotationDraft; error?: unknown; response: Response },
  what: string,
): DraftEnvelope {
  const draft = unwrap(result, what);
  const etag = result.response.headers.get("etag");
  if (!etag) throw new Error("The backend omitted the annotation concurrency token.");
  return { draft, etag };
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

export function useAnnotationDraft(imageId: number | undefined) {
  return useQuery<DraftEnvelope>({
    queryKey: queryKeys.annotationDraft(imageId ?? -1),
    queryFn: async () => {
      const result = await api.POST("/api/images/{image_id}/annotations/draft", {
        params: { path: { image_id: imageId as number } },
      });
      return withEtag(result, "the annotation draft");
    },
    enabled: imageId !== undefined,
    staleTime: Infinity,
  });
}

export function useSaveAnnotation(imageId: number) {
  const queryClient = useQueryClient();
  return useMutation<DraftEnvelope, Error, { document: AnnotationDocumentInput; etag: string }>({
    mutationFn: async ({ document, etag }) => {
      const result = await api.PUT("/api/images/{image_id}/annotations/draft", {
        params: {
          path: { image_id: imageId },
          header: { "If-Match": etag },
        },
        body: document,
      });
      const draft = unwrap(result, "the saved annotation draft");
      const nextEtag = result.response.headers.get("etag");
      if (!nextEtag) throw new Error("The backend omitted the annotation concurrency token.");
      return { draft, etag: nextEtag };
    },
    onSuccess: (saved) => {
      queryClient.setQueryData(queryKeys.annotationDraft(imageId), saved);
    },
  });
}

export function useCompleteAnnotation(imageId: number) {
  const queryClient = useQueryClient();
  return useMutation<AnnotationRevision, Error, string>({
    mutationFn: async (etag) =>
      unwrap(
        await api.POST("/api/images/{image_id}/annotations/complete", {
          params: {
            path: { image_id: imageId },
            header: { "If-Match": etag },
          },
        }),
        "the completed annotation revision",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.annotationDraft(imageId) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.annotationRevisions(imageId) });
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
