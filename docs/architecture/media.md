# Media and thumbnail cache

Source images are whatever the dataset supplies — BMPs of a few megabytes, multi-megapixel JPEGs.
Resolution and format are per-image data, recorded at import. Serving source files to a browser grid would
move hundreds of megabytes per screen, so the media layer serves **three tiers**:

| Tier | Size | Format | Used by |
| --- | --- | --- | --- |
| `thumb` | 256 px long edge | WebP, q80 | dataset browser grid, ranked lists |
| `preview` | 1024 px long edge | WebP, q85 | sample viewer, side-by-side channel comparison |
| `full` | native | lossless PNG, on demand | pixel-peeping, anomaly-map overlay inspection |

The **full tier is lossless** because it is used to judge defects and align anomaly-map overlays, where
compression artifacts could be mistaken for surface features.

**Cache layout:** `data/thumbnails/{thumb,preview}/{image_id}.webp`. Keying by `image_id` alone is safe
**because imported files are immutable**: paths are recorded once, `sha256` is stored at import, and
`verify` ([import](import.md)) detects drift, so there is no invalidation to build. Deleting a dataset
removes exactly these image-id cache files after its database transaction; it never derives a deletion
target from a source path.

**Only `thumb` and `preview` are cached.** A cached `full` tier would cost about a megabyte per image to
avoid re-rendering something looked at once, so it is rendered per request and kept off the wire by its
`ETag`.

**Generation** is lazy — the first request for a cached tier renders and stores it — plus a post-import
**prewarm job** ([jobs](jobs.md)) that generates all thumbs up front. Responses carry an `ETag` derived from
the image `sha256` plus tier, and `Cache-Control: immutable`.

**Bit depth is transparent.** Decoding normalizes 8-bit grayscale and 24-bit sources to one in-memory
representation, and the tier renderer is bit-depth agnostic, so no call site special-cases either.

**The ground-truth outline is served in two frames.** `GET /api/images/{image_id}/mask` draws the annotated
region's contour as a transparent PNG at the source's own size — an outline, because a fill hides the
pixels the reader is judging the map against. `frame=prepared&experiment_id=N` draws the same mask
projected through that run's pinned region transform, at the prepared size, which is what a
**diagnostics** pane needs: diagnostics are prepared-frame, and a source-frame outline over one is off by
exactly the crop and letterbox ([diagnostics](diagnostics.md)). The contour is traced after projection, and
the `ETag` carries the frame and the pinned manifest digest so the two never answer for each other out of a
cache.

**A supervised run's label map is drawn for a gallery tile.**
`GET /api/experiments/{id}/images/{iid}/label-map?colours=…[&truth=true]` paints the method's label map
(a faint fill and a solid border) or the truth over the pinned classes (a dashed border only) as a
transparent PNG, with the sample page's rule (`labelPaint.ts`; its two alphas are mirrored in
`media/overlay.py`). Class `i` is the `i`-th of `colours`, which the client sends from the design
system's palette, so the server keeps no copy of it; fewer colours than pinned classes is a 422. The
plane is strided — never resampled — to the `thumb` tier's 256 px long edge before its borders are
traced, so the response is bounded whatever the image's size. It is revalidated rather than immutable,
because re-scoring or completing an annotation changes it: the `ETag` is the drawn plane's digest plus
the colours, and a match is a 304 without encoding.

**A detection run's boxes are drawn for a gallery tile.**
`GET /api/experiments/{id}/images/{iid}/box-map?colours=match,false_positive,missed[&predictions=false][&truth=false]`
draws one image's detections that the subset's cut keeps (solid) and its true boxes of a pinned class
(dashed) as an SVG laid out at the source's size, so the tile stretches it exactly as it stretches the
thumbnail; the stroke does not scale with it, and no text is written. Each box takes the tone of its
verdict at IoU 0.5 ([evaluation](evaluation.md#object-detection)), and the three tones arrive from the
client — read from the theme's custom properties — so the server keeps no copy of them. The response is
bounded by construction (at most 100 detections an image) and revalidated by the drawing's digest.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
