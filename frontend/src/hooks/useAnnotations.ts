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

import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, apiError, unwrap } from "../api/client";
import type {
  AnnotationDocument,
  AnnotationDocumentInput,
  AnnotationLabel,
  AnnotationRevision,
  AnnotationScope,
  AnnotationScopeState,
  CopyRegionsResult,
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
  /** Null until the first save has created the draft. */
  version: number | null;
  /** Null until the first save has created the draft; see `useEditorDraft`. */
  etag: string | null;
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

/**
 * Recolour a class.
 *
 * The colour belongs to the *label*, not to the session: a scratch should look like a scratch on
 * every screen and for everyone reading the dataset. The route has existed since the taxonomy
 * did ("Rename or recolour a class without changing its stable key") and simply had no caller —
 * which is why, until the tint bug was fixed, the palette was both unreachable and unused.
 */
export function useUpdateAnnotationLabel(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<AnnotationLabel, Error, AnnotationLabel>({
    mutationFn: async (label) =>
      unwrap(
        await api.PUT("/api/datasets/{dataset_id}/annotation-labels/{key}", {
          params: { path: { dataset_id: datasetId, key: label.key } },
          body: { name: label.name, color: label.color, position: label.position },
        }),
        "the annotation label",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.annotationLabels(datasetId) });
    },
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

/**
 * What the editor opens on: the draft in progress, or the document one would start from.
 *
 * `etag === null` is the load-bearing state, not a placeholder. It says "nothing is persisted
 * yet", which is what tells the first save to create rather than update. Reading is a plain GET
 * that writes nothing — it used to be a POST issued from this query function, which is how
 * merely browsing the queue left a draft row behind every image and made `annotation_scope`
 * permanently unreachable.
 */
async function fetchImageDraft(imageId: number): Promise<DraftEnvelope> {
  const result = await api.GET("/api/images/{image_id}/annotations/draft", {
    params: { path: { image_id: imageId } },
  });
  const state = unwrap(result, "the annotation draft");
  return {
    document: state.document,
    version: state.version,
    etag: state.persisted ? requireEtag(result.response) : null,
  };
}

async function fetchSampleDraft(sampleId: number): Promise<DraftEnvelope> {
  const result = await api.GET("/api/samples/{sample_id}/annotations/draft", {
    params: { path: { sample_id: sampleId } },
  });
  const state = unwrap(result, "the annotation draft");
  return {
    document: state.document,
    version: state.version,
    etag: state.persisted ? requireEtag(result.response) : null,
  };
}

export function useEditorDraft(target: DraftTarget | undefined) {
  return useQuery<DraftEnvelope>({
    queryKey: target ? targetKey(target) : ["annotations", "none"],
    queryFn: async () => {
      if (!target) throw new Error("no annotation target");
      return target.scope === "sample"
        ? fetchSampleDraft(target.sampleId)
        : fetchImageDraft(target.imageId);
    },
    enabled: target !== undefined,
    staleTime: Infinity,
  });
}

/**
 * Every channel of the part, read ahead of being asked for.
 *
 * Two things fall out of it, and both used to be impossible. Switching channel resolves from
 * the cache instead of blanking the editor while a request runs — which is only affordable
 * now that reading a draft is a read; under the previous design this would have persisted a
 * row per channel per part visited. And the copy dialog can say what each channel already
 * holds, so "copy to dark" is a decision rather than a guess.
 *
 * Sample scope has nothing to prefetch: one document already covers every channel.
 */
export function useSiblingDrafts(imageIds: readonly number[], enabled: boolean) {
  return useQueries({
    queries: imageIds.map((imageId) => ({
      queryKey: queryKeys.annotationDraft(imageId),
      queryFn: () => fetchImageDraft(imageId),
      enabled,
      staleTime: Infinity,
    })),
    combine: (results) => {
      const shapeCounts = new Map<number, number>();
      results.forEach((result, index) => {
        const imageId = imageIds[index];
        if (imageId !== undefined && result.data) {
          shapeCounts.set(imageId, result.data.document.shapes.length);
        }
      });
      return shapeCounts;
    },
  });
}

/**
 * Persist the document, creating the draft on the first save.
 *
 * The create is `POST` with `If-None-Match: *` rather than an upsert, and that is a correctness
 * choice: an upsert hands a second window holding the same seed the *first* window's saved draft
 * together with a currently-valid token for a document it never read, and its next save then
 * overwrites work no precondition could protect. Create-only turns that into a 412.
 */
export function useSaveDraft(target: DraftTarget) {
  const queryClient = useQueryClient();
  return useMutation<
    DraftEnvelope,
    Error,
    { document: AnnotationDocumentInput; etag: string | null }
  >({
    mutationFn: async ({ document, etag }) => {
      // The four calls are spelled out rather than folded together: their response bodies are
      // different types, and a union of them widens `draft` into something no route returns.
      if (target.scope === "sample") {
        if (etag === null) {
          const result = await api.POST("/api/samples/{sample_id}/annotations/draft", {
            params: {
              path: { sample_id: target.sampleId },
              header: { "If-None-Match": "*" },
            },
            body: document,
          });
          const draft = unwrap(result, "the new annotation draft");
          return {
            document: draft.document,
            version: draft.version,
            etag: requireEtag(result.response),
          };
        }
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
      if (etag === null) {
        const result = await api.POST("/api/images/{image_id}/annotations/draft", {
          params: {
            path: { image_id: target.imageId },
            header: { "If-None-Match": "*" },
          },
          body: document,
        });
        const draft = unwrap(result, "the new annotation draft");
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
 * Throw the draft away without completing it.
 *
 * `force` sends `If-Match: *`, which RFC 9110 defines as matching any current representation.
 * It is offered only after a 412 has told the user the draft moved under them, so discarding
 * someone else's work is always a second, informed click.
 */
export function useDiscardDraft(target: DraftTarget) {
  const queryClient = useQueryClient();
  return useMutation<void, Error, { etag: string | null; force?: boolean }>({
    mutationFn: async ({ etag, force = false }) => {
      const ifMatch = force || etag === null ? "*" : etag;
      const result =
        target.scope === "sample"
          ? await api.DELETE("/api/samples/{sample_id}/annotations/draft", {
              params: { path: { sample_id: target.sampleId }, header: { "If-Match": ifMatch } },
            })
          : await api.DELETE("/api/images/{image_id}/annotations/draft", {
              params: { path: { image_id: target.imageId }, header: { "If-Match": ifMatch } },
            });
      if (result.error !== undefined) throw apiError(result, "the annotation draft");
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: targetKey(target) });
    },
  });
}

/**
 * Put this channel's regions on its siblings too.
 *
 * Image scope only, and appending only. The token guards the *source*: a copy is one action
 * that writes to several channels, so an editor that has fallen behind must not be allowed to
 * spread a stale document across all of them. The targets need no token because appending
 * cannot destroy what is already there.
 */
export function useCopyRegions(imageId: number) {
  const queryClient = useQueryClient();
  return useMutation<
    CopyRegionsResult,
    Error,
    { etag: string; targetImageIds: readonly number[] }
  >({
    mutationFn: async ({ etag, targetImageIds }) =>
      unwrap(
        await api.POST("/api/images/{image_id}/annotations/copy-regions", {
          params: { path: { image_id: imageId }, header: { "If-Match": etag } },
          body: { target_image_ids: [...targetImageIds] },
        }),
        "the copied regions",
      ),
    onSuccess: (result) => {
      // Each target's draft moved on the server, and its cached envelope still carries the
      // version before the copy — an editor opening it with that token would collect a 412.
      for (const target of result.targets) {
        void queryClient.invalidateQueries({ queryKey: queryKeys.annotationDraft(target.image_id) });
      }
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
    onSuccess: (revisions) => {
      // Written, not invalidated. Invalidating refetched the draft while this screen's observer
      // was still mounted, and while that query function was a POST the refetch *recreated* the
      // draft completion had just consumed — one orphan per unit of finished work, and the
      // single largest source of the rows migration 016 had to clear. A completed target seeds
      // from the revision just written, which is exactly what the server would now return.
      const completed = revisions[0];
      if (completed) {
        queryClient.setQueryData<DraftEnvelope>(targetKey(target), {
          document: completed.document,
          version: null,
          etag: null,
        });
      }
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
