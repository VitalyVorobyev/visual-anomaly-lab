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

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
