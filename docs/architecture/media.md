# Media and thumbnail cache

Source images are whatever the dataset supplies — the showcase tree holds 1280×1024 BMPs of roughly
3.9 MB, the public reference datasets hold multi-megapixel JPEGs. Nothing below depends on which:
resolution and format are per-image data, recorded at import. Serving source files directly to a browser
grid would move hundreds of megabytes per screen and stall the UI, so the media layer serves
**three tiers**:

| Tier | Size | Format | Used by |
| --- | --- | --- | --- |
| `thumb` | 256 px long edge | WebP, q80 | dataset browser grid, ranked lists |
| `preview` | 1024 px long edge | WebP, q85 | sample viewer, side-by-side channel comparison |
| `full` | native 1280×1024 | lossless PNG, on demand | pixel-peeping, anomaly-map overlay inspection |

WebP at these quality levels is roughly two orders of magnitude smaller than the source BMP with no
perceptible loss at the display sizes involved. The **full tier is lossless** because it is used to judge
defects and to align anomaly-map overlays, where JPEG-style artifacts could be mistaken for surface features.

**Cache layout:** `data/thumbnails/{thumb,preview}/{image_id}.webp`. Keying by `image_id` alone is safe
**because imported files are immutable**: paths are recorded once, `sha256` is stored at import, and `verify`
([import](import.md)) detects any drift. There is no invalidation problem to solve, so none is built.

**Only `thumb` and `preview` are cached.** A cached `full` tier costs roughly 1.2 MB per image — most of a
gigabyte for one dataset — to avoid re-rendering something that is looked at once, so it is rendered per
request and kept off the wire by its `ETag` instead.

**Generation** is lazy — the first request for a cached tier renders and stores it — with a post-import
**pre-warm job** (reusing the job system, [the job system](jobs.md)) that generates all thumbs up front so the first browse is
smooth. Responses carry an `ETag` derived from the image `sha256` plus tier, and `Cache-Control: immutable`,
so the WebView re-fetches nothing.

**8-bit grayscale BMPs are handled transparently.** Decoding normalizes to a common in-memory representation
and the tier renderer is bit-depth agnostic, so the mixed 24-bit / 8-bit reference data requires no special
casing at any call site.

---

[← the handbook](README.md) · [why it is shaped this way](../adr/README.md)
